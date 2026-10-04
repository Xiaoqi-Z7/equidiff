"""BIL checkpoint adapter for EquiDiff's evaluation policy contract.

This module intentionally contains inference glue only.  The BIL network and
checkpoint stay unchanged; observations arrive from EquiDiff with shape
``B x To x ...`` and actions leave as EquiDiff absolute actions with shape
``B x Ta x 10`` (position, rotation-6D, gripper).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn

# Checkpoint factories are registered by import side effects.
import robomimic_ext.algo  # noqa: F401
import robomimic_ext.config  # noqa: F401
from robomimic_ext.utils.file_utils import policy_from_checkpoint


CALIBRATION_OBS_KEY = "bil_body_to_site_rotation"


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """PyTorch3D-compatible matrix-to-6D conversion used by EquiDiff."""
    return matrix[..., :2, :].clone().reshape(*matrix.shape[:-2], 6)


class BILCheckpointPolicy(nn.Module):
    """Expose a BIL checkpoint through EquiDiff's ``predict_action`` API."""

    def __init__(
        self,
        checkpoint_path: str,
        device: str | torch.device = "cuda",
        task_description: Optional[str] = None,
    ):
        super().__init__()
        checkpoint_path = str(Path(checkpoint_path).expanduser().resolve())
        requested_device = torch.device(device)
        if requested_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")

        rollout_policy, checkpoint = policy_from_checkpoint(
            device=requested_device,
            ckpt_path=checkpoint_path,
            verbose=False,
        )
        self._rollout_policy = rollout_policy
        self.bil_policy = rollout_policy.policy
        self.checkpoint = checkpoint
        self.checkpoint_path = checkpoint_path
        self.task_description = task_description
        self.obs_normalization_stats = rollout_policy.obs_normalization_stats

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
        # This value is used only by the adapter, not by the trained network.
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

        calibration = obs_dict[CALIBRATION_OBS_KEY]
        if not isinstance(calibration, torch.Tensor):
            calibration = torch.as_tensor(calibration)
        calibration = calibration.to(
            device=self.device, dtype=trajectory["act_rot"].dtype
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

        # BIL predicts the robosuite EEF body frame. Absolute OSC commands in
        # EquiDiff control an EEF site frame, so apply the measured transform.
        body_rotation = trajectory["act_rot"][:, 0]
        site_rotation = torch.einsum(
            "btij,bjk->btik", body_rotation, calibration
        )
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
