"""Gym adapter that exposes BIL observations to EquiDiff's rollout stack."""

from __future__ import annotations

from typing import Any, Optional

import gym
import numpy as np
from gym import spaces
from scipy.spatial.transform import Rotation

from robomimic_mimicgen.utils.mimicgen_env import MimicGenRolloutEnv
from robomimic_mimicgen.utils.obs_utils import build_obs_dict


CALIBRATION_OBS_KEY = "bil_body_to_site_rotation"


class BILMimicGenWrapper(gym.Env):
    """Use BIL's modern-MuJoCo environment inside EquiDiff wrappers.

    The reset and seed behavior follows ``RobomimicImageWrapper``.  The only
    added observation is the measured EEF-body to controller-site rotation,
    which prevents a silent 90-degree orientation-frame error.
    """

    metadata = {"render.modes": ["rgb_array"]}

    def __init__(
        self,
        env: Any,
        shape_meta: dict,
        init_state: Optional[np.ndarray | dict] = None,
        render_obs_key: str = "agentview_image",
        render_height: int = 256,
        render_width: int = 256,
        eef_correction_enabled: bool = False,
        eef_correction_position_threshold_m: float = 0.01,
        eef_correction_rotation_threshold_deg: float = 10.0,
        eef_correction_max_per_chunk: int = 1,
        eef_replan_enabled: bool = False,
        eef_replan_position_threshold_m: float = 0.01,
        eef_replan_rotation_threshold_deg: float = 10.0,
    ):
        super().__init__()
        self.env = env
        self.task_name = env.name
        self.task_prefix = self.task_name.rsplit("_D", 1)[0]
        self.shape_meta = shape_meta
        self.init_state = init_state
        self.render_obs_key = render_obs_key
        self.render_height = int(render_height)
        self.render_width = int(render_width)
        self.seed_state_map = {}
        self._seed = None
        self.has_reset_before = False
        self.eef_correction_enabled = bool(eef_correction_enabled)
        self.eef_replan_enabled = bool(eef_replan_enabled)
        if self.eef_correction_enabled and self.eef_replan_enabled:
            raise ValueError("EEF correction and EEF replan are mutually exclusive")
        self.eef_correction_position_threshold_m = float(
            eef_correction_position_threshold_m
        )
        self.eef_correction_rotation_threshold_rad = np.deg2rad(
            float(eef_correction_rotation_threshold_deg)
        )
        self.eef_correction_max_per_chunk = int(eef_correction_max_per_chunk)
        self.eef_replan_position_threshold_m = float(
            eef_replan_position_threshold_m
        )
        self.eef_replan_rotation_threshold_rad = np.deg2rad(
            float(eef_replan_rotation_threshold_deg)
        )
        if self.eef_correction_position_threshold_m < 0:
            raise ValueError("EEF correction position threshold must be non-negative")
        if self.eef_correction_rotation_threshold_rad < 0:
            raise ValueError("EEF correction rotation threshold must be non-negative")
        if self.eef_correction_max_per_chunk <= 0:
            raise ValueError("EEF correction max-per-chunk must be positive")
        if self.eef_replan_position_threshold_m < 0:
            raise ValueError("EEF replan position threshold must be non-negative")
        if self.eef_replan_rotation_threshold_rad < 0:
            raise ValueError("EEF replan rotation threshold must be non-negative")
        self._reset_eef_correction_state()

        self.action_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=tuple(shape_meta["action"]["shape"]),
            dtype=np.float32,
        )
        observation_spaces = {}
        for key, value in shape_meta["obs"].items():
            observation_spaces[key] = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=tuple(value["shape"]),
                dtype=np.float32,
            )
        self.observation_space = spaces.Dict(observation_spaces)

    def _body_to_site_rotation(self, raw_obs: dict) -> np.ndarray:
        body_rotation = Rotation.from_quat(
            np.asarray(raw_obs["robot0_eef_quat"], dtype=np.float64)
        ).as_matrix()
        if hasattr(self.env, "get_eef_tracking_state"):
            site_rotation = self.env.get_eef_tracking_state()["actual_ori"]
        else:
            robosuite_env = self.env.env
            controller = robosuite_env.robots[0].controller
            site_id = robosuite_env.sim.model.site_name2id(controller.eef_name)
            site_rotation = np.asarray(
                robosuite_env.sim.data.site_xmat[site_id].reshape(3, 3)
            ).copy()
        return (body_rotation.T @ site_rotation).reshape(9).astype(np.float32)

    def get_observation(self, raw_obs=None):
        if isinstance(self.env, MimicGenRolloutEnv):
            if raw_obs is None:
                raw_obs = self.env.get_observation()
        else:
            # EnvRobosuite's observation layout is image-policy oriented.
            # Read the identical simulator state through BIL's canonical
            # observation builder instead of altering environment dynamics.
            raw_obs = build_obs_dict(
                self.env.env,
                task_prefix=self.task_prefix,
            )
        extended = dict(raw_obs)
        extended[CALIBRATION_OBS_KEY] = self._body_to_site_rotation(raw_obs)

        result = {}
        for key, space in self.observation_space.spaces.items():
            if key not in extended:
                raise KeyError(
                    f"Environment {self.task_name} did not produce observation {key!r}"
                )
            value = np.asarray(extended[key], dtype=space.dtype)
            if value.shape != space.shape:
                raise ValueError(
                    f"Observation {key!r} has shape {value.shape}; expected {space.shape}"
                )
            result[key] = value
        return result

    def seed(self, seed=None):
        if seed is not None and int(seed) < 0:
            raise ValueError(f"Seed must be non-negative, got {seed}")
        self._seed = None if seed is None else int(seed)
        return [self._seed]

    def reset(self):
        self._reset_eef_correction_state()
        if self.init_state is not None:
            if not self.has_reset_before:
                self.env.reset()
                self.has_reset_before = True
            if isinstance(self.init_state, dict):
                state = dict(self.init_state)
            else:
                state = {"states": np.asarray(self.init_state).copy()}
            raw_obs = self.env.reset_to(state)
        elif self._seed is not None:
            seed = self._seed
            if seed in self.seed_state_map:
                raw_obs = self.env.reset_to(
                    {"states": np.asarray(self.seed_state_map[seed]).copy()}
                )
            else:
                np.random.seed(seed)
                raw_obs = self.env.reset()
                self.seed_state_map[seed] = np.asarray(
                    self.env.get_state()["states"]
                ).copy()
            self._seed = None
        else:
            raw_obs = self.env.reset()
        return self.get_observation(raw_obs)

    def _reset_eef_correction_state(self):
        self._eef_correction_plan = None
        self._eef_correction_plan_index = 0
        self._eef_corrections_this_chunk = 0
        self._eef_correction_stats = {
            "checks": 0,
            "corrections": 0,
            "improved_corrections": 0,
            "extra_control_steps": 0,
            "pre_position_error_sum_m": 0.0,
            "pre_rotation_error_sum_rad": 0.0,
            "triggered_pre_position_error_sum_m": 0.0,
            "triggered_pre_rotation_error_sum_rad": 0.0,
            "post_position_error_sum_m": 0.0,
            "post_rotation_error_sum_rad": 0.0,
            "max_pre_position_error_m": 0.0,
            "max_pre_rotation_error_rad": 0.0,
        }
        self._eef_replan_stats = {
            "checks": 0,
            "replans": 0,
            "discarded_actions": 0,
            "pre_position_error_sum_m": 0.0,
            "pre_rotation_error_sum_rad": 0.0,
            "triggered_position_error_sum_m": 0.0,
            "triggered_rotation_error_sum_rad": 0.0,
            "max_pre_position_error_m": 0.0,
            "max_pre_rotation_error_rad": 0.0,
        }

    def set_eef_correction_plan(
        self,
        future_eef_pos: np.ndarray,
        future_eef_rot: np.ndarray,
        correction_actions: np.ndarray,
    ) -> None:
        """Install one predicted achieved-EEF trajectory for the next chunk.

        Poses remain in BIL's observed EEF body frame for error measurement.
        ``correction_actions`` are already calibrated absolute OSC site-frame
        targets with shape ``(Ta, 7)``.
        """
        positions = np.asarray(future_eef_pos, dtype=np.float64)
        rotations = np.asarray(future_eef_rot, dtype=np.float64)
        actions = np.asarray(correction_actions, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[-1] != 3:
            raise ValueError(
                f"Future EEF positions must have shape (Ta, 3), got {positions.shape}"
            )
        expected_rot = (positions.shape[0], 3, 3)
        expected_action = (positions.shape[0], 7)
        if rotations.shape != expected_rot:
            raise ValueError(
                f"Future EEF rotations must have shape {expected_rot}, got {rotations.shape}"
            )
        if actions.shape != expected_action:
            raise ValueError(
                f"Correction actions must have shape {expected_action}, got {actions.shape}"
            )
        if not (
            np.isfinite(positions).all()
            and np.isfinite(rotations).all()
            and np.isfinite(actions).all()
        ):
            raise ValueError("EEF correction plan contains NaN or Inf")
        self._eef_correction_plan = {
            "positions": positions.copy(),
            "rotations": rotations.copy(),
            "actions": actions.copy(),
        }
        self._eef_correction_plan_index = 0
        self._eef_corrections_this_chunk = 0

    def set_eef_replan_plan(
        self,
        future_eef_pos: np.ndarray,
        future_eef_rot: np.ndarray,
    ) -> None:
        """Install achieved-EEF predictions used to interrupt a stale chunk."""
        positions = np.asarray(future_eef_pos, dtype=np.float64)
        rotations = np.asarray(future_eef_rot, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[-1] != 3:
            raise ValueError(
                f"Future EEF positions must have shape (Ta, 3), got {positions.shape}"
            )
        expected_rot = (positions.shape[0], 3, 3)
        if rotations.shape != expected_rot:
            raise ValueError(
                f"Future EEF rotations must have shape {expected_rot}, got {rotations.shape}"
            )
        if not (np.isfinite(positions).all() and np.isfinite(rotations).all()):
            raise ValueError("EEF replan plan contains NaN or Inf")
        self._eef_correction_plan = {
            "positions": positions.copy(),
            "rotations": rotations.copy(),
            "actions": None,
        }
        self._eef_correction_plan_index = 0
        self._eef_corrections_this_chunk = 0

    @staticmethod
    def _eef_pose_error(observation, target_position, target_rotation):
        actual_position = np.asarray(
            observation["robot0_eef_pos"], dtype=np.float64
        )
        actual_rotation = Rotation.from_quat(
            np.asarray(observation["robot0_eef_quat"], dtype=np.float64)
        ).as_matrix()
        position_error = float(np.linalg.norm(target_position - actual_position))
        rotation_error = float(
            Rotation.from_matrix(actual_rotation.T @ target_rotation).magnitude()
        )
        return position_error, rotation_error

    def _record_eef_correction_check(self, position_error, rotation_error):
        stats = self._eef_correction_stats
        stats["checks"] += 1
        stats["pre_position_error_sum_m"] += position_error
        stats["pre_rotation_error_sum_rad"] += rotation_error
        stats["max_pre_position_error_m"] = max(
            stats["max_pre_position_error_m"], position_error
        )
        stats["max_pre_rotation_error_rad"] = max(
            stats["max_pre_rotation_error_rad"], rotation_error
        )

    def _record_eef_replan_check(self, position_error, rotation_error):
        stats = self._eef_replan_stats
        stats["checks"] += 1
        stats["pre_position_error_sum_m"] += position_error
        stats["pre_rotation_error_sum_rad"] += rotation_error
        stats["max_pre_position_error_m"] = max(
            stats["max_pre_position_error_m"], position_error
        )
        stats["max_pre_rotation_error_rad"] = max(
            stats["max_pre_rotation_error_rad"], rotation_error
        )

    @property
    def eef_correction_stats(self):
        stats = dict(self._eef_correction_stats)
        checks = max(stats["checks"], 1)
        corrections = max(stats["corrections"], 1)
        return {
            "checks": float(stats["checks"]),
            "corrections": float(stats["corrections"]),
            "extra_control_steps": float(stats["extra_control_steps"]),
            "trigger_rate": float(stats["corrections"] / checks),
            "improved_fraction": float(
                stats["improved_corrections"] / corrections
            ),
            "pre_position_error_mean_m": float(
                stats["pre_position_error_sum_m"] / checks
            ),
            "pre_rotation_error_mean_rad": float(
                stats["pre_rotation_error_sum_rad"] / checks
            ),
            "triggered_pre_position_error_mean_m": float(
                stats["triggered_pre_position_error_sum_m"] / corrections
            ),
            "triggered_pre_rotation_error_mean_rad": float(
                stats["triggered_pre_rotation_error_sum_rad"] / corrections
            ),
            "post_position_error_mean_m": float(
                stats["post_position_error_sum_m"] / corrections
            ),
            "post_rotation_error_mean_rad": float(
                stats["post_rotation_error_sum_rad"] / corrections
            ),
            "max_pre_position_error_m": float(
                stats["max_pre_position_error_m"]
            ),
            "max_pre_rotation_error_rad": float(
                stats["max_pre_rotation_error_rad"]
            ),
        }

    @property
    def eef_replan_stats(self):
        stats = dict(self._eef_replan_stats)
        checks = max(stats["checks"], 1)
        replans = max(stats["replans"], 1)
        return {
            "checks": float(stats["checks"]),
            "replans": float(stats["replans"]),
            "discarded_actions": float(stats["discarded_actions"]),
            "trigger_rate": float(stats["replans"] / checks),
            "pre_position_error_mean_m": float(
                stats["pre_position_error_sum_m"] / checks
            ),
            "pre_rotation_error_mean_rad": float(
                stats["pre_rotation_error_sum_rad"] / checks
            ),
            "triggered_position_error_mean_m": float(
                stats["triggered_position_error_sum_m"] / replans
            ),
            "triggered_rotation_error_mean_rad": float(
                stats["triggered_rotation_error_sum_rad"] / replans
            ),
            "max_pre_position_error_m": float(
                stats["max_pre_position_error_m"]
            ),
            "max_pre_rotation_error_rad": float(
                stats["max_pre_rotation_error_rad"]
            ),
        }

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (7,):
            raise ValueError(
                "BIL MimicGen wrapper expects EquiDiff's inverse-transformed "
                f"7D absolute OSC action, got {action.shape}"
            )
        raw_obs, reward, done, info = self.env.step(action)
        observation = self.get_observation(raw_obs)
        info = dict(info)

        plan = self._eef_correction_plan
        plan_index = self._eef_correction_plan_index
        replan_enabled = bool(getattr(self, "eef_replan_enabled", False))
        can_check = (
            (self.eef_correction_enabled or replan_enabled)
            and plan is not None
            and plan_index < plan["positions"].shape[0]
        )
        if can_check:
            target_position = plan["positions"][plan_index]
            target_rotation = plan["rotations"][plan_index]
            pre_position_error, pre_rotation_error = self._eef_pose_error(
                observation, target_position, target_rotation
            )
            if replan_enabled:
                self._record_eef_replan_check(
                    pre_position_error, pre_rotation_error
                )
                position_threshold = self.eef_replan_position_threshold_m
                rotation_threshold = self.eef_replan_rotation_threshold_rad
            else:
                self._record_eef_correction_check(
                    pre_position_error, pre_rotation_error
                )
                position_threshold = self.eef_correction_position_threshold_m
                rotation_threshold = self.eef_correction_rotation_threshold_rad
            should_correct = (
                not done
                and (
                    replan_enabled
                    or self._eef_corrections_this_chunk
                    < self.eef_correction_max_per_chunk
                )
                and (
                    pre_position_error > position_threshold
                    or pre_rotation_error > rotation_threshold
                )
            )
            if replan_enabled:
                info.update({
                    "eef_replan_checked": True,
                    "eef_replan_triggered": bool(should_correct),
                    "eef_replan_pre_position_error_m": pre_position_error,
                    "eef_replan_pre_rotation_error_rad": pre_rotation_error,
                })
                if should_correct:
                    stats = self._eef_replan_stats
                    stats["replans"] += 1
                    stats["discarded_actions"] += (
                        plan["positions"].shape[0] - plan_index - 1
                    )
                    stats["triggered_position_error_sum_m"] += pre_position_error
                    stats["triggered_rotation_error_sum_rad"] += pre_rotation_error
                    # MultiStepWrapper consumes this private control signal and
                    # returns the latest observation to the policy immediately.
                    info["truncate_action_chunk"] = True
            else:
                info.update({
                    "eef_correction_checked": True,
                    "eef_correction_triggered": bool(should_correct),
                    "eef_correction_pre_position_error_m": pre_position_error,
                    "eef_correction_pre_rotation_error_rad": pre_rotation_error,
                })
            if should_correct and self.eef_correction_enabled:
                correction_action = plan["actions"][plan_index]
                corrected_raw_obs, corrected_reward, corrected_done, corrected_info = (
                    self.env.step(correction_action)
                )
                observation = self.get_observation(corrected_raw_obs)
                post_position_error, post_rotation_error = self._eef_pose_error(
                    observation, target_position, target_rotation
                )
                reward = max(float(reward), float(corrected_reward))
                done = bool(done or corrected_done)
                info.update(dict(corrected_info))
                info.update({
                    "eef_correction_post_position_error_m": post_position_error,
                    "eef_correction_post_rotation_error_rad": post_rotation_error,
                })
                stats = self._eef_correction_stats
                stats["corrections"] += 1
                stats["extra_control_steps"] += 1
                stats["triggered_pre_position_error_sum_m"] += pre_position_error
                stats["triggered_pre_rotation_error_sum_rad"] += pre_rotation_error
                stats["post_position_error_sum_m"] += post_position_error
                stats["post_rotation_error_sum_rad"] += post_rotation_error
                position_was_trigger = (
                    pre_position_error
                    > self.eef_correction_position_threshold_m
                )
                rotation_was_trigger = (
                    pre_rotation_error
                    > self.eef_correction_rotation_threshold_rad
                )
                if (
                    (not position_was_trigger
                     or post_position_error < pre_position_error)
                    and (not rotation_was_trigger
                         or post_rotation_error < pre_rotation_error)
                ):
                    stats["improved_corrections"] += 1
                self._eef_corrections_this_chunk += 1

        if plan is not None:
            self._eef_correction_plan_index += 1
        return observation, reward, done, info

    def render(self, mode="rgb_array"):
        if mode != "rgb_array":
            raise ValueError(f"Unsupported render mode {mode!r}")
        current = self.env
        visited = set()
        sim = None
        while id(current) not in visited:
            visited.add(id(current))
            candidate = getattr(current, "sim", None)
            if candidate is not None:
                sim = candidate
                break
            nested = getattr(current, "env", None)
            if nested is None or nested is current:
                break
            current = nested
        if sim is None:
            raise RuntimeError("Could not locate MuJoCo simulator for rendering")

        # MuJoCo 2.3 can expose both visual geoms (group 1) and the colored
        # collision meshes (group 0) through robosuite's offscreen context.
        # Hide only collision alpha while capturing this frame, then restore
        # it before the next simulator step. This changes pixels only.
        collision_ids = np.flatnonzero(np.asarray(sim.model.geom_group) == 0)
        original_rgba = np.asarray(sim.model.geom_rgba[collision_ids]).copy()
        try:
            for geom_id in collision_ids:
                sim.model.geom_rgba[geom_id, 3] = 0.0
            return self.env.render(
                mode="rgb_array",
                height=self.render_height,
                width=self.render_width,
                camera_name="agentview",
            )
        finally:
            for geom_id, rgba in zip(collision_ids, original_rgba):
                sim.model.geom_rgba[geom_id] = rgba

    def close(self):
        close = getattr(self.env, "close", None)
        if close is not None:
            close()
