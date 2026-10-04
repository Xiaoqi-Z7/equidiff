# BIL checkpoints in the EquiDiff MimicGen test pipeline

This integration branch is based on upstream EquiDiff commit
`e40abb003b25071d4e5b01bfa9933dc16cd32c67`. The parent BIL repository pins
the exact integration commit through its `equidiff` submodule.
The entry point directly instantiates EquiDiff's upstream
`RobomimicImageRunner`. Its spawned vector environments, `run()` loop,
`MultiStepWrapper`, action conversion, seeded reset protocol, maximum episode
reward, and `test/mean_score` metric are therefore shared with native policies.

The compatibility layer makes only the changes needed by BIL:

- load a BIL / robomimic checkpoint;
- expose BIL's object poses and object IDs to the policy;
- run BIL inference over EquiDiff's environment batch;
- convert BIL EEF-body rotations to the OSC controller site frame;
- emit EquiDiff absolute actions as position + rotation-6D + gripper;
- keep EquiDiff's `EnvUtils.create_env_from_metadata()` environment factory;
- replace only the observation wrapper so BIL can read canonical object poses.

The runner intentionally has no BIL-specific environment factory. This keeps
environment construction, episode resets, vector workers, stepping, and metric
aggregation on EquiDiff's code path; the injected wrapper only translates
observations for BIL and leaves simulator dynamics unchanged.

Both EquiDiff and BIL use MuJoCo 2.3.2 and the same robosuite commit. The
`mujoco_py` reference in robomimic is only a stale exception-type import; it is
not the simulation backend. The compatibility shim maps that exception to
`mujoco.FatalError`; it does not replace or modify the simulator.

## Environment

From the BIL repository root:

```bash
git submodule update --init --recursive
pixi install -e bil-mimicgen
```

The `bil-mimicgen` feature includes the additional EquiDiff rollout
dependencies (`gym`, `dill`, `av`, and zarr 2). No separate EquiDiff conda
environment is required for BIL evaluation.

## Full EquiDiff-protocol evaluation

```bash
MUJOCO_GL=egl \
PYTHONPATH="$PWD:$PWD/equidiff" \
pixi run -e bil-mimicgen python -m equi_diffpo.scripts.eval_bil_checkpoint \
  --checkpoint /absolute/path/to/model.pth \
  --dataset "$PWD/data/mimicgen/square_d2.hdf5" \
  --output-dir "$PWD/results_equidiff_bil/square_seed0" \
  --task Square_D2 \
  --device cuda \
  --seed 0 \
  --n-train 6 \
  --n-test 50 \
  --test-start-seed 100000 \
  --n-envs 28 \
  --max-steps 400
```

`--task` can be omitted when the task name is stored in the checkpoint. The
runner reads `To`, `Ta`, and `Tp` from the checkpoint and rejects a horizon
mismatch instead of silently changing evaluation behavior.

By default video recording is disabled to reduce GPU and renderer memory. Use
`--n-test-vis N` or `--n-train-vis N` to enable it for the first `N` episodes.
Numerical results are written to `<output-dir>/metrics.json`.

For a language-conditioned checkpoint, pass the exact training instruction
with `--task-description`. The adapter repeats it across the vector batch.

## Quick smoke test

The following tests the complete two-worker path without claiming a meaningful
success rate:

```bash
MUJOCO_GL=egl \
PYTHONPATH="$PWD:$PWD/equidiff" \
pixi run -e bil-mimicgen python -m equi_diffpo.scripts.eval_bil_checkpoint \
  --checkpoint /absolute/path/to/model.pth \
  --dataset "$PWD/data/mimicgen/square_d2.hdf5" \
  --output-dir /tmp/bil_equidiff_smoke \
  --device cpu \
  --n-train 0 \
  --n-test 2 \
  --n-envs 2 \
  --max-steps 8
```

## Fair comparison checklist

Use the same EquiDiff runner settings for all methods:

- task and raw MimicGen dataset;
- 6 train initializations and 50 test seeds;
- test seeds starting at 100000;
- 28 environment workers;
- the checkpoint's observation/action horizons and EquiDiff's task-specific
  episode limit (400, 500, 800, or 1000 steps);
- absolute OSC control;
- policy sampling seed and number of repeated evaluation runs.

The body-to-site rotation is measured inside every worker and passed as an
adapter-only observation. It is not an input to the learned BIL network.
