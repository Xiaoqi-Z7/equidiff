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

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (7,):
            raise ValueError(
                "BIL MimicGen wrapper expects EquiDiff's inverse-transformed "
                f"7D absolute OSC action, got {action.shape}"
            )
        raw_obs, reward, done, info = self.env.step(action)
        return self.get_observation(raw_obs), reward, done, info

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
