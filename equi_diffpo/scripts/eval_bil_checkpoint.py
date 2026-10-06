"""Evaluate a BIL checkpoint in EquiDiff's vectorized MimicGen pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path

import numpy as np
import torch

from equi_diffpo.env.robomimic.bil_mimicgen_wrapper import (
    BILMimicGenWrapper,
)
from equi_diffpo.env_runner.robomimic_image_runner import RobomimicImageRunner


MIMICGEN_MAX_STEPS = {
    "Stack_D1": 400,
    "StackThree_D1": 400,
    "Square_D2": 400,
    "Threading_D2": 400,
    "Coffee_D2": 400,
    "ThreePieceAssembly_D2": 500,
    "HammerCleanup_D1": 500,
    "MugCleanup_D1": 500,
    "Kitchen_D1": 800,
    "NutAssembly_D0": 500,
    "PickPlace_D0": 1000,
    "CoffeePreparation_D1": 800,
}


def organize_local_videos(log_data: dict, output_dir: Path) -> list[dict]:
    """Give rollout videos stable seed/status names and write local manifests."""
    pattern = re.compile(r"^(train|test)/sim_video_(-?\d+)$")
    rows = []
    for key, video in log_data.items():
        match = pattern.match(key)
        if match is None:
            continue
        split, seed_text = match.groups()
        reward_key = f"{split}/sim_max_reward_{seed_text}"
        if reward_key not in log_data:
            raise KeyError(f"Missing reward for video {key}: {reward_key}")
        source_value = getattr(video, "_path", None)
        if not source_value:
            raise ValueError(f"Video object for {key} does not expose a local path")
        source = Path(source_value).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Video for {key} not found: {source}")

        reward = float(log_data[reward_key])
        successful = reward >= 1.0 - 1e-8
        status = "success" if successful else "failure"
        score_suffix = "" if successful else f"_score_{reward:.3f}"
        target = source.with_name(
            f"{split}_seed_{seed_text}_{status}{score_suffix}{source.suffix}"
        )
        if source != target:
            source.replace(target)
            try:
                video._path = str(target)
            except AttributeError:
                pass
        rows.append({
            "split": split,
            "seed": int(seed_text),
            "max_reward": reward,
            "successful": successful,
            "status": status,
            "video": str(target),
        })

    rows.sort(key=lambda row: (row["split"], row["seed"]))
    if not rows:
        return rows

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "video_manifest.json"
    json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    csv_path = output_dir / "video_manifest.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"video_manifest_json={json_path}")
    print(f"video_manifest_csv={csv_path}")
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--task",
        default=None,
        help="MimicGen task name (for example Square_D2); inferred when possible.",
    )
    parser.add_argument("--task-description", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0, help="Policy sampling seed")
    parser.add_argument("--n-train", type=int, default=6)
    parser.add_argument("--n-train-vis", type=int, default=2)
    parser.add_argument("--train-start-idx", type=int, default=0)
    parser.add_argument("--n-test", type=int, default=50)
    parser.add_argument("--n-test-vis", type=int, default=4)
    parser.add_argument("--test-start-seed", type=int, default=100000)
    parser.add_argument("--n-envs", type=int, default=28)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override EquiDiff's task-specific horizon.",
    )
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--crf", type=int, default=22)
    parser.add_argument("--tqdm-interval-sec", type=float, default=1.0)
    parser.add_argument(
        "--inference-object-noise",
        "--inference_object_noise",
        dest="inference_object_noise",
        action="store_true",
        help="Add clipped Gaussian noise only to valid BIL object poses.",
    )
    parser.add_argument(
        "--noise-position-std", "--noise_position_std",
        dest="noise_position_std", type=float, default=0.005 / 2.795,
    )
    parser.add_argument(
        "--noise-position-clip", "--noise_position_clip",
        dest="noise_position_clip", type=float, default=0.005,
    )
    parser.add_argument(
        "--noise-rotation-std", "--noise_rotation_std",
        dest="noise_rotation_std", type=float, default=5.0 / 1.96,
        help="Rotation noise standard deviation in degrees.",
    )
    parser.add_argument(
        "--noise-rotation-clip", "--noise_rotation_clip",
        dest="noise_rotation_clip", type=float, default=5.0,
        help="Rotation noise clipping magnitude in degrees.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Keep BIL's checkpoint stack out of multiprocessing spawn workers.  A
    # spawned AsyncVectorEnv worker re-imports this entrypoint as ``__mp_main__``
    # before it executes the worker target.  Importing the policy at module
    # scope therefore made every environment process recursively import
    # diffusers and transformers from the shared filesystem even though only
    # the parent process performs policy inference.
    from equi_diffpo.policy.bil_checkpoint_policy import BILCheckpointPolicy

    policy = BILCheckpointPolicy(
        checkpoint_path=args.checkpoint,
        device=args.device,
        task_description=args.task_description,
    )
    policy.set_inference_object_noise(
        enabled=args.inference_object_noise,
        position_std=args.noise_position_std,
        position_clip=args.noise_position_clip,
        rotation_std=args.noise_rotation_std,
        rotation_clip=args.noise_rotation_clip,
    )
    task_name = args.task or policy.task_name
    if task_name is None:
        raise ValueError("Could not infer task from checkpoint; pass --task")
    if args.task is not None and policy.task_name not in (None, args.task):
        raise ValueError(
            f"Checkpoint task is {policy.task_name!r}, but --task is {args.task!r}"
        )
    max_steps = args.max_steps
    if max_steps is None:
        try:
            max_steps = MIMICGEN_MAX_STEPS[task_name]
        except KeyError as error:
            raise ValueError(
                f"No EquiDiff max_steps mapping for task {task_name!r}; "
                "pass --max-steps explicitly"
            ) from error

    print(
        json.dumps(
            {
                "checkpoint": policy.checkpoint_path,
                "runner": (
                    "equi_diffpo.env_runner.robomimic_image_runner."
                    "RobomimicImageRunner"
                ),
                "task": task_name,
                "device": str(policy.device),
                "To": policy.observation_horizon,
                "Ta": policy.action_horizon,
                "Tp": policy.prediction_horizon,
                "n_envs": args.n_envs,
                "n_test": args.n_test,
                "test_start_seed": args.test_start_seed,
                "max_steps": max_steps,
                "inference_object_noise": args.inference_object_noise,
                "noise_position_std": args.noise_position_std,
                "noise_position_clip": args.noise_position_clip,
                "noise_rotation_std_degrees": args.noise_rotation_std,
                "noise_rotation_clip_degrees": args.noise_rotation_clip,
            },
            indent=2,
        )
    )

    runner = None
    try:
        runner = RobomimicImageRunner(
            output_dir=args.output_dir,
            dataset_path=args.dataset,
            shape_meta=policy.shape_meta,
            n_train=args.n_train,
            n_train_vis=args.n_train_vis,
            train_start_idx=args.train_start_idx,
            n_test=args.n_test,
            n_test_vis=args.n_test_vis,
            test_start_seed=args.test_start_seed,
            max_steps=max_steps,
            n_obs_steps=policy.observation_horizon,
            n_action_steps=policy.action_horizon,
            fps=args.fps,
            crf=args.crf,
            n_envs=args.n_envs,
            abs_action=True,
            tqdm_interval_sec=args.tqdm_interval_sec,
            wrapper_factory=BILMimicGenWrapper,
        )
        log_data = runner.run(policy)
        organize_local_videos(log_data, Path(args.output_dir).expanduser().resolve())
        metrics = {
            key: float(value)
            for key, value in log_data.items()
            if isinstance(value, (int, float, np.integer, np.floating))
        }
        metrics_path = Path(args.output_dir).expanduser().resolve() / "metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(metrics, indent=2, sort_keys=True))
        print(f"metrics_path={metrics_path}")
    finally:
        if runner is not None:
            runner.env.close()


if __name__ == "__main__":
    main()
