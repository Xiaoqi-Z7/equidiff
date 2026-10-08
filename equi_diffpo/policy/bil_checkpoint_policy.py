"""BIL checkpoint adapter for EquiDiff's evaluation policy contract.

This module intentionally contains inference glue only.  The BIL network and
checkpoint stay unchanged; observations arrive from EquiDiff with shape
``B x To x ...`` and actions leave as EquiDiff absolute actions with shape
``B x Ta x 10`` (position, rotation-6D, gripper).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn

CALIBRATION_OBS_KEY = "bil_body_to_site_rotation"


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """PyTorch3D-compatible matrix-to-6D conversion used by EquiDiff."""
    return matrix[..., :2, :].clone().reshape(*matrix.shape[:-2], 6)


def quaternion_multiply_xyzw(
    left: torch.Tensor, right: torch.Tensor
) -> torch.Tensor:
    """Hamilton product for quaternions stored in robosuite ``xyzw`` order."""
    left_xyz, left_w = left[..., :3], left[..., 3:4]
    right_xyz, right_w = right[..., :3], right[..., 3:4]
    left_xyz, right_xyz = torch.broadcast_tensors(left_xyz, right_xyz)
    left_w, right_w = torch.broadcast_tensors(left_w, right_w)
    xyz = (
        left_w * right_xyz
        + right_w * left_xyz
        + torch.linalg.cross(left_xyz, right_xyz, dim=-1)
    )
    w = left_w * right_w - (left_xyz * right_xyz).sum(dim=-1, keepdim=True)
    return torch.cat((xyz, w), dim=-1)


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

        # These imports register BIL checkpoint factories and pull in the
        # diffusion stack.  Keep them local to construction so importing this
        # adapter in an environment worker never imports diffusers or
        # transformers.  Policy construction only happens in the evaluator's
        # parent process.
        import robomimic_ext.algo  # noqa: F401
        import robomimic_ext.config  # noqa: F401
        from robomimic_ext.utils.file_utils import policy_from_checkpoint

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
        self.symmetry_object_index: Optional[int] = None
        self.symmetry_axis: Optional[str] = None
        self.symmetry_angle_degrees: Optional[float] = None

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

    def set_object_symmetry_intervention(
        self,
        object_index: Optional[int],
        axis: str = "z",
        angle_degrees: float = 180.0,
    ) -> None:
        """Configure a fixed local-frame rotation of one policy object pose.

        The intervention changes only the BIL ``object`` observation. For an
        object world rotation ``R`` and local symmetry rotation ``S``, the
        adapter presents ``R @ S`` to the policy while leaving the simulator
        state, object position, and every other observation unchanged.
        """
        if object_index is None:
            self.symmetry_object_index = None
            self.symmetry_axis = None
            self.symmetry_angle_degrees = None
            return
        if object_index < 0:
            raise ValueError(f"object_index must be non-negative, got {object_index}")
        if axis not in ("x", "y", "z"):
            raise ValueError(f"axis must be one of x, y, z; got {axis!r}")
        if not math.isfinite(angle_degrees):
            raise ValueError(f"angle_degrees must be finite, got {angle_degrees}")
        self.symmetry_object_index = int(object_index)
        self.symmetry_axis = axis
        self.symmetry_angle_degrees = float(angle_degrees)

    def _apply_object_symmetry_intervention(
        self, obs_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        object_index = getattr(self, "symmetry_object_index", None)
        if object_index is None:
            return obs_dict
        if "object" not in obs_dict:
            raise KeyError("Object symmetry intervention requires an 'object' observation")

        objects = obs_dict["object"]
        if objects.shape[-1] % 7 != 0:
            raise ValueError(
                f"Expected concatenated 7D object poses, got shape {objects.shape}."
            )
        num_objects = objects.shape[-1] // 7
        if object_index >= num_objects:
            raise IndexError(
                f"Symmetry object index {object_index} is out of range for "
                f"{num_objects} object poses"
            )

        object_poses = objects.clone().reshape(*objects.shape[:-1], num_objects, 7)
        quaternion = object_poses[..., object_index, 3:7]
        norm = torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
        valid = norm > 1e-8
        safe_quaternion = quaternion / norm.clamp_min(1e-8)

        half_angle = math.radians(self.symmetry_angle_degrees) / 2.0
        local_rotation = torch.zeros(
            4, device=objects.device, dtype=objects.dtype
        )
        local_rotation[("x", "y", "z").index(self.symmetry_axis)] = math.sin(
            half_angle
        )
        local_rotation[3] = math.cos(half_angle)
        rotated = quaternion_multiply_xyzw(safe_quaternion, local_rotation)
        rotated = rotated / torch.linalg.vector_norm(
            rotated, dim=-1, keepdim=True
        ).clamp_min(1e-8)
        # Match the canonical quaternion convention used by build_obs_dict().
        rotated = torch.where(rotated[..., 3:4] < 0, -rotated, rotated)
        object_poses[..., object_index, 3:7] = torch.where(
            valid, rotated, quaternion
        )

        intervened = dict(obs_dict)
        intervened["object"] = object_poses.reshape_as(objects)
        return intervened

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

        # Apply the causal pose intervention in raw observation coordinates,
        # before the checkpoint's normalization statistics.
        prepared = self._apply_object_symmetry_intervention(prepared)

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
