"""BIL checkpoint adapter for EquiDiff's evaluation policy contract.

This module intentionally contains inference glue only.  The BIL network and
checkpoint stay unchanged; observations arrive from EquiDiff with shape
``B x To x ...`` and actions leave as EquiDiff absolute actions with shape
``B x Ta x 10`` (position, rotation-6D, gripper).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn

CALIBRATION_OBS_KEY = "bil_body_to_site_rotation"


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """PyTorch3D-compatible matrix-to-6D conversion used by EquiDiff."""
    return matrix[..., :2, :].clone().reshape(*matrix.shape[:-2], 6)


def checkpoint_with_inference_steps(
    checkpoint: dict,
    num_inference_steps: Optional[int],
) -> tuple[dict, int, int, int]:
    """Return a checkpoint copy with an optional inference-step override.

    The flow schedule is constructed while the policy is restored, so the
    checkpoint config must be changed before ``algo_factory`` runs.  The
    original checkpoint object is never mutated.
    """
    stored_config = checkpoint.get("config")
    if isinstance(stored_config, str):
        config = json.loads(stored_config)
    elif isinstance(stored_config, dict):
        config = copy.deepcopy(stored_config)
    else:
        raise TypeError(
            "Checkpoint config must be a JSON string or dictionary, got "
            f"{type(stored_config).__name__}"
        )

    try:
        flow_config = config["algo"]["flow"]
        num_training_steps = int(flow_config["num_training_steps"])
        checkpoint_num_inference_steps = int(
            flow_config["num_inference_steps"]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "Checkpoint does not contain valid algo.flow training and "
            "inference step counts"
        ) from error

    effective_num_inference_steps = (
        checkpoint_num_inference_steps
        if num_inference_steps is None
        else int(num_inference_steps)
    )
    if not 1 <= effective_num_inference_steps <= num_training_steps:
        raise ValueError(
            "num_inference_steps must satisfy 1 <= steps <= "
            f"num_training_steps ({num_training_steps}), got "
            f"{effective_num_inference_steps}"
        )

    checkpoint_copy = dict(checkpoint)
    if num_inference_steps is not None:
        flow_config["num_inference_steps"] = effective_num_inference_steps
        checkpoint_copy["config"] = (
            json.dumps(config) if isinstance(stored_config, str) else config
        )
    return (
        checkpoint_copy,
        checkpoint_num_inference_steps,
        effective_num_inference_steps,
        num_training_steps,
    )


def checkpoint_with_action_horizon(
    checkpoint: dict,
    action_horizon: Optional[int],
) -> tuple[dict, int, int, int]:
    """Return a checkpoint copy with an optional rollout action horizon."""
    stored_config = checkpoint.get("config")
    if isinstance(stored_config, str):
        config = json.loads(stored_config)
    elif isinstance(stored_config, dict):
        config = copy.deepcopy(stored_config)
    else:
        raise TypeError(
            "Checkpoint config must be a JSON string or dictionary, got "
            f"{type(stored_config).__name__}"
        )

    try:
        horizon_config = config["algo"]["horizon"]
        prediction_horizon = int(horizon_config["prediction_horizon"])
        checkpoint_action_horizon = int(horizon_config["action_horizon"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "Checkpoint does not contain valid algo.horizon prediction and "
            "action horizons"
        ) from error

    effective_action_horizon = (
        checkpoint_action_horizon
        if action_horizon is None
        else int(action_horizon)
    )
    if not 1 <= effective_action_horizon <= prediction_horizon:
        raise ValueError(
            "action_horizon must satisfy 1 <= steps <= "
            f"prediction_horizon ({prediction_horizon}), got "
            f"{effective_action_horizon}"
        )

    checkpoint_copy = dict(checkpoint)
    if action_horizon is not None:
        horizon_config["action_horizon"] = effective_action_horizon
        checkpoint_copy["config"] = (
            json.dumps(config) if isinstance(stored_config, str) else config
        )
    return (
        checkpoint_copy,
        checkpoint_action_horizon,
        effective_action_horizon,
        prediction_horizon,
    )


class BILCheckpointPolicy(nn.Module):
    """Expose a BIL checkpoint through EquiDiff's ``predict_action`` API."""

    def __init__(
        self,
        checkpoint_path: str,
        device: str | torch.device = "cuda",
        task_description: Optional[str] = None,
        num_inference_steps: Optional[int] = None,
        action_horizon: Optional[int] = None,
        expected_prediction_target: str = "action",
    ):
        super().__init__()
        checkpoint_path = str(Path(checkpoint_path).expanduser().resolve())
        requested_device = torch.device(device)
        if requested_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")

        # These imports register BIL checkpoint factories and pull in the
        # diffusion stack.  Keep them local to construction so importing this
        # adapter in an environment worker never imports diffusers or
        # transformers.  Policy construction only happens in the evaluator's
        # parent process.
        import robomimic_ext.algo  # noqa: F401
        import robomimic_ext.config  # noqa: F401
        from robomimic.utils.file_utils import maybe_dict_from_checkpoint
        from robomimic_ext.utils.file_utils import policy_from_checkpoint

        checkpoint = maybe_dict_from_checkpoint(ckpt_path=checkpoint_path)
        (
            checkpoint,
            self.checkpoint_num_inference_steps,
            self.num_inference_steps,
            self.num_training_steps,
        ) = checkpoint_with_inference_steps(
            checkpoint,
            num_inference_steps,
        )
        (
            checkpoint,
            self.checkpoint_action_horizon,
            effective_action_horizon,
            checkpoint_prediction_horizon,
        ) = checkpoint_with_action_horizon(
            checkpoint,
            action_horizon,
        )
        rollout_policy, checkpoint = policy_from_checkpoint(
            device=requested_device,
            ckpt_dict=checkpoint,
            verbose=False,
        )
        self._rollout_policy = rollout_policy
        self.bil_policy = rollout_policy.policy
        self.checkpoint = checkpoint
        self.checkpoint_path = checkpoint_path
        self.task_description = task_description
        self.obs_normalization_stats = rollout_policy.obs_normalization_stats

        self.prediction_target = str(
            getattr(self.bil_policy, "prediction_target", "action")
        )
        if self.prediction_target != expected_prediction_target:
            raise ValueError(
                f"Checkpoint predicts {self.prediction_target!r}, expected "
                f"{expected_prediction_target!r}: {checkpoint_path}"
            )

        self.action_type = str(getattr(self.bil_policy, "action_type", "delta"))
        if self.action_type not in ("delta", "absolute"):
            raise ValueError(
                "Checkpoint has unsupported action type "
                f"{self.action_type!r}: {checkpoint_path}"
            )

        if self.bil_policy.num_arms != 1:
            raise NotImplementedError(
                "The MimicGen EquiDiff adapter currently supports one arm; "
                f"checkpoint has {self.bil_policy.num_arms}."
            )
        if not self.bil_policy.use_gripper:
            raise NotImplementedError("EquiDiff MimicGen actions require a gripper channel")

        self.observation_horizon = int(self.bil_policy.To)
        self.action_horizon = int(self.bil_policy.Ta)
        self.prediction_horizon = int(self.bil_policy.Tp)
        if self.action_horizon != effective_action_horizon:
            raise RuntimeError(
                "Loaded policy action horizon does not match the requested "
                f"value: {self.action_horizon} != {effective_action_horizon}"
            )
        if self.prediction_horizon != checkpoint_prediction_horizon:
            raise RuntimeError(
                "Loaded policy prediction horizon changed unexpectedly: "
                f"{self.prediction_horizon} != {checkpoint_prediction_horizon}"
            )
        self.expected_obs_shapes = {
            key: tuple(shape)
            for key, shape in self.bil_policy.obs_shapes.items()
        }

        config = checkpoint.get("config", {})
        if isinstance(config, str):
            config = json.loads(config)
        self.checkpoint_config = config
        self.task_name = self._infer_task_name(checkpoint, config)

        # Register one scalar so EquiDiff can query ``device`` and ``dtype``
        # in exactly the same way as for its native nn.Module policies.
        self.register_buffer(
            "_device_anchor", torch.empty(0, device=requested_device), persistent=False
        )

    @staticmethod
    def _infer_task_name(checkpoint: dict, config: dict) -> Optional[str]:
        env_metadata = checkpoint.get("env_metadata") or {}
        env_name = env_metadata.get("env_name")
        if env_name:
            return str(env_name)
        experiment = config.get("experiment", {}) if isinstance(config, dict) else {}
        env_name = experiment.get("env")
        return None if env_name is None else str(env_name)

    @property
    def device(self) -> torch.device:
        return self._device_anchor.device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.bil_policy.nets.parameters()).dtype

    @property
    def shape_meta(self) -> dict:
        """Observation/action metadata consumed by the EquiDiff runner."""
        obs = {
            key: {"shape": list(shape), "type": "low_dim"}
            for key, shape in self.expected_obs_shapes.items()
        }
        # Delta-trained BIL checkpoints predict an absolute pose expressed in
        # robosuite's observed EEF body frame. Their EquiDiff rollout action
        # therefore needs the measured body-to-controller-site rotation.
        # Controller-absolute checkpoints are supervised directly with OSC
        # site-frame goals and must not receive this transform a second time.
        if self.action_type == "delta":
            obs[CALIBRATION_OBS_KEY] = {"shape": [9], "type": "low_dim"}
        return {
            "action": {"shape": [10]},
            "obs": obs,
        }

    def reset(self) -> None:
        """The batched adapter is stateless across EquiDiff action chunks."""

    def set_normalizer(self, normalizer) -> None:
        raise RuntimeError("BIL observation normalization is stored in its checkpoint")

    def set_inference_object_noise(
        self,
        enabled: bool,
        position_std: Optional[float] = None,
        position_clip: Optional[float] = None,
        rotation_std: Optional[float] = None,
        rotation_clip: Optional[float] = None,
    ) -> None:
        """Forward rollout-time object-pose noise to the BIL policy.

        The underlying BIL implementation perturbs only valid entries in the
        raw ``object`` pose observation. Robot EEF observations and actions
        remain noise-free.
        """
        setter = getattr(self.bil_policy, "set_inference_object_noise", None)
        if setter is None:
            if enabled:
                raise TypeError(
                    f"Policy {type(self.bil_policy).__name__} does not support "
                    "inference object noise"
                )
            return
        setter(
            enabled=enabled,
            position_std=position_std,
            position_clip=position_clip,
            rotation_std=rotation_std,
            rotation_clip=rotation_clip,
        )

    def _prepare_obs(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        missing = set(self.expected_obs_shapes).difference(obs_dict)
        if missing:
            raise KeyError(f"EquiDiff observation is missing BIL keys: {sorted(missing)}")

        prepared = {}
        batch_size = None
        for key, expected_shape in self.expected_obs_shapes.items():
            value = obs_dict[key]
            if not isinstance(value, torch.Tensor):
                value = torch.as_tensor(value)
            value = value.to(device=self.device, dtype=torch.float32)
            if value.ndim < 2 or tuple(value.shape[2:]) != expected_shape:
                raise ValueError(
                    f"Observation {key!r} must have shape B x To x {expected_shape}; "
                    f"got {tuple(value.shape)}"
                )
            if value.shape[1] < self.observation_horizon:
                raise ValueError(
                    f"Observation {key!r} has To={value.shape[1]}, but checkpoint "
                    f"requires To={self.observation_horizon}"
                )
            if batch_size is None:
                batch_size = value.shape[0]
            elif value.shape[0] != batch_size:
                raise ValueError("All observation keys must share the same batch size")
            prepared[key] = value[:, -self.observation_horizon :]

        if self.obs_normalization_stats is not None:
            for key, stats in self.obs_normalization_stats.items():
                mean = torch.as_tensor(
                    stats["mean"], device=self.device, dtype=prepared[key].dtype
                )
                std = torch.as_tensor(
                    stats["std"], device=self.device, dtype=prepared[key].dtype
                )
                prepared[key] = (prepared[key] - mean) / std
        return prepared

    @torch.inference_mode()
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        prepared = self._prepare_obs(obs_dict)
        # Match BIL's standard rollout order: checkpoint normalization first,
        # followed by optional inference-time object-pose noise. Keeping this
        # policy-side leaves EquiDiff's simulator state untouched.
        prepared = self.bil_policy._add_inference_noise_to_object(prepared)
        batch_size = next(iter(prepared.values())).shape[0]

        task_description = self.task_description
        if task_description is not None and batch_size > 1:
            task_description = [task_description] * batch_size

        trajectory = self.bil_policy._get_action_trajectory(
            obs_dict=prepared,
            goal_dict=None,
            task_description=task_description,
        )

        predicted_rotation = trajectory["act_rot"][:, 0]
        if self.action_type == "delta":
            calibration = obs_dict[CALIBRATION_OBS_KEY]
            if not isinstance(calibration, torch.Tensor):
                calibration = torch.as_tensor(calibration)
            calibration = calibration.to(
                device=self.device, dtype=predicted_rotation.dtype
            )
            if (
                calibration.ndim != 3
                or calibration.shape[0] != batch_size
                or calibration.shape[1] < 1
                or calibration.shape[2] != 9
            ):
                raise ValueError(
                    "Calibration observation must have shape B x To x 9; "
                    f"got {tuple(calibration.shape)}"
                )
            calibration = calibration[:, -1].reshape(batch_size, 3, 3)

            # The original delta-action training path constructs its target
            # from the observed EEF body frame. EquiDiff's absolute OSC action
            # controls the EEF site frame, so retain the historical calibrated
            # conversion for these checkpoints.
            site_rotation = torch.einsum(
                "btij,bjk->btik", predicted_rotation, calibration
            )
        else:
            # Controller-absolute labels are the OSC controller's own
            # goal_ori matrices. They already live in the EEF site frame.
            site_rotation = predicted_rotation
        action = torch.cat(
            (
                trajectory["act_pos"][:, 0],
                matrix_to_rotation_6d(site_rotation),
                trajectory["act_grip"][:, 0],
            ),
            dim=-1,
        )
        expected_shape = (batch_size, self.action_horizon, 10)
        if tuple(action.shape) != expected_shape:
            raise RuntimeError(
                f"Adapter produced {tuple(action.shape)}, expected {expected_shape}"
            )
        if not torch.isfinite(action).all():
            raise RuntimeError("BIL adapter produced NaN or Inf actions")
        return {"action": action}


class BILEEFPlanner(BILCheckpointPolicy):
    """Load an independent future-EEF checkpoint without producing actions."""

    def __init__(
        self,
        checkpoint_path: str,
        device: str | torch.device = "cuda",
        task_description: Optional[str] = None,
        num_inference_steps: Optional[int] = None,
        sampling_seed: int = 0,
    ):
        super().__init__(
            checkpoint_path=checkpoint_path,
            device=device,
            task_description=task_description,
            num_inference_steps=num_inference_steps,
            action_horizon=None,
            expected_prediction_target="future_eef",
        )
        if self.action_horizon != self.prediction_horizon:
            raise ValueError(
                "EEF planner checkpoint must expose its complete prediction "
                f"horizon, got Ta={self.action_horizon}, "
                f"Tp={self.prediction_horizon}"
            )
        self.sampling_seed = int(sampling_seed)
        self._prediction_index = 0

    def reset(self) -> None:
        self._prediction_index = 0

    def reference_to_action(
        self,
        obs_dict: Dict[str, torch.Tensor],
        reference_dict: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Convert predicted EEF-body poses to absolute OSC site targets.

        MimicGen observations expose the EEF body orientation while
        robosuite's OSC controller targets its EEF site. Position origins
        coincide in the current observation/controller convention; rotation
        requires the measured body-to-site calibration used by ActionFlow.
        """
        positions = reference_dict["future_eef_pos"]
        body_rotations = reference_dict["future_eef_rot"]
        grip = reference_dict["act_grip"]
        calibration = obs_dict[CALIBRATION_OBS_KEY]
        if not isinstance(calibration, torch.Tensor):
            calibration = torch.as_tensor(calibration)
        calibration = calibration.to(
            device=body_rotations.device,
            dtype=body_rotations.dtype,
        )
        batch_size = body_rotations.shape[0]
        if (
            calibration.ndim != 3
            or calibration.shape[0] != batch_size
            or calibration.shape[1] < 1
            or calibration.shape[2] != 9
        ):
            raise ValueError(
                "Calibration observation must have shape B x To x 9; "
                f"got {tuple(calibration.shape)}"
            )
        calibration = calibration[:, -1].reshape(batch_size, 3, 3)
        site_rotations = torch.einsum(
            "btij,bjk->btik", body_rotations, calibration
        )
        action = torch.cat(
            (positions, matrix_to_rotation_6d(site_rotations), grip), dim=-1
        )
        expected = (batch_size, self.prediction_horizon, 10)
        if tuple(action.shape) != expected:
            raise RuntimeError(
                f"Planner OSC targets have {tuple(action.shape)}, expected "
                f"{expected}"
            )
        return action

    @torch.inference_mode()
    def predict_reference(
        self, obs_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Return world-frame EEF body poses and expert-style grip commands."""
        prepared = self._prepare_obs(obs_dict)
        batch_size = next(iter(prepared.values())).shape[0]
        task_description = self.task_description
        if task_description is not None and batch_size > 1:
            task_description = [task_description] * batch_size

        cuda_devices = []
        if self.device.type == "cuda":
            cuda_devices = [
                self.device.index
                if self.device.index is not None
                else torch.cuda.current_device()
            ]
        # Planner sampling must not advance ActionFlow's global RNG stream;
        # otherwise a nominally log-only ablation would change later actions.
        with torch.random.fork_rng(devices=cuda_devices):
            seed = self.sampling_seed + self._prediction_index
            torch.manual_seed(seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            trajectory = self.bil_policy._get_action_trajectory(
                obs_dict=prepared,
                goal_dict=None,
                task_description=task_description,
            )
        self._prediction_index += 1
        result = {
            "future_eef_pos": trajectory["act_pos"][:, 0],
            "future_eef_rot": trajectory["act_rot"][:, 0],
            "act_grip": trajectory["act_grip"][:, 0],
        }
        expected_pos = (batch_size, self.prediction_horizon, 3)
        expected_rot = (batch_size, self.prediction_horizon, 3, 3)
        expected_grip = (batch_size, self.prediction_horizon, 1)
        if tuple(result["future_eef_pos"].shape) != expected_pos:
            raise RuntimeError(
                f"Planner positions have {tuple(result['future_eef_pos'].shape)}, "
                f"expected {expected_pos}"
            )
        if tuple(result["future_eef_rot"].shape) != expected_rot:
            raise RuntimeError(
                f"Planner rotations have {tuple(result['future_eef_rot'].shape)}, "
                f"expected {expected_rot}"
            )
        if tuple(result["act_grip"].shape) != expected_grip:
            raise RuntimeError(
                f"Planner gripper has {tuple(result['act_grip'].shape)}, "
                f"expected {expected_grip}"
            )
        if not all(torch.isfinite(value).all() for value in result.values()):
            raise RuntimeError("EEF planner produced NaN or Inf values")
        return result
