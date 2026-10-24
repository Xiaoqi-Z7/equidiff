import os
import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import h5py
import math
import dill
import wandb.sdk.data_types.video as wv
from equi_diffpo.gym_util.async_vector_env import AsyncVectorEnv
from equi_diffpo.gym_util.sync_vector_env import SyncVectorEnv
from equi_diffpo.gym_util.multistep_wrapper import MultiStepWrapper
from equi_diffpo.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from equi_diffpo.model.common.rotation_transformer import RotationTransformer

from equi_diffpo.policy.base_image_policy import BaseImagePolicy
from equi_diffpo.common.pytorch_util import dict_apply
from equi_diffpo.env_runner.base_image_runner import BaseImageRunner
from equi_diffpo.env.robomimic.robomimic_image_wrapper import RobomimicImageWrapper
from equi_diffpo.common.mujoco_py_compat import (
    install_mujoco_py_exception_shim,
    install_robosuite_egl_cleanup_shim,
)

install_mujoco_py_exception_shim()
install_robosuite_egl_cleanup_shim()

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils


def create_env(env_meta, shape_meta, enable_render=True):
    modality_mapping = collections.defaultdict(list)
    for key, attr in shape_meta['obs'].items():
        modality_mapping[attr.get('type', 'low_dim')].append(key)
    ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False, 
        render_offscreen=enable_render,
        use_image_obs=enable_render, 
    )
    return env


class RobomimicImageRunner(BaseImageRunner):
    """
    Robomimic envs already enforces number of steps.
    """

    def __init__(self, 
            output_dir,
            dataset_path,
            shape_meta:dict,
            n_train=10,
            n_train_vis=3,
            train_start_idx=0,
            n_test=22,
            n_test_vis=6,
            test_start_seed=10000,
            max_steps=400,
            n_obs_steps=2,
            n_action_steps=8,
            render_obs_key='agentview_image',
            fps=10,
            crf=22,
            past_action=False,
            abs_action=False,
            tqdm_interval_sec=5.0,
            n_envs=None,
            wrapper_factory=None,
            eef_correction=False,
            eef_correction_position_threshold_m=0.01,
            eef_correction_rotation_threshold_deg=10.0,
            eef_correction_max_per_chunk=1,
            eef_replan=False,
            eef_replan_position_threshold_m=0.01,
            eef_replan_rotation_threshold_deg=10.0,
        ):
        super().__init__(output_dir)

        if n_envs is None:
            n_envs = n_train + n_test
        if wrapper_factory is None:
            wrapper_factory = RobomimicImageWrapper
        if eef_correction and not abs_action:
            raise ValueError("EEF correction requires absolute OSC actions")
        if eef_correction and eef_replan:
            raise ValueError("EEF correction and EEF replan are mutually exclusive")

        wrapper_kwargs = {}
        if eef_correction:
            wrapper_kwargs = {
                "eef_correction_enabled": True,
                "eef_correction_position_threshold_m": float(
                    eef_correction_position_threshold_m
                ),
                "eef_correction_rotation_threshold_deg": float(
                    eef_correction_rotation_threshold_deg
                ),
                "eef_correction_max_per_chunk": int(
                    eef_correction_max_per_chunk
                ),
            }
        elif eef_replan:
            wrapper_kwargs = {
                "eef_replan_enabled": True,
                "eef_replan_position_threshold_m": float(
                    eef_replan_position_threshold_m
                ),
                "eef_replan_rotation_threshold_deg": float(
                    eef_replan_rotation_threshold_deg
                ),
            }

        # assert n_obs_steps <= n_action_steps
        dataset_path = os.path.expanduser(dataset_path)
        robosuite_fps = 20
        steps_per_render = max(robosuite_fps // fps, 1)

        # read from dataset
        env_meta = FileUtils.get_env_metadata_from_dataset(
            dataset_path)
        # disable object state observation
        env_meta['env_kwargs']['use_object_obs'] = False

        rotation_transformer = None
        if abs_action:
            env_meta['env_kwargs']['controller_configs']['control_delta'] = False
            rotation_transformer = RotationTransformer('axis_angle', 'rotation_6d')

        def env_fn():
            robomimic_env = create_env(
                env_meta=env_meta,
                shape_meta=shape_meta,
            )
            # Robosuite's hard reset causes excessive memory consumption.
            # Disabled to run more envs.
            # https://github.com/ARISE-Initiative/robosuite/blob/92abf5595eddb3a845cd1093703e5a3ccd01e77e/robosuite/environments/base.py#L247-L248
            robomimic_env.env.hard_reset = False
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    wrapper_factory(
                        env=robomimic_env,
                        shape_meta=shape_meta,
                        init_state=None,
                        render_obs_key=render_obs_key,
                        **wrapper_kwargs,
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec='h264',
                        input_pix_fmt='rgb24',
                        crf=crf,
                        thread_type='FRAME',
                        thread_count=1
                    ),
                    file_path=None,
                    steps_per_render=steps_per_render
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps
            )
        
        # For each process the OpenGL context can only be initialized once
        # Since AsyncVectorEnv uses fork to create worker process,
        # a separate env_fn that does not create OpenGL context (enable_render=False)
        # is needed to initialize spaces.
        def dummy_env_fn():
            robomimic_env = create_env(
                env_meta=env_meta,
                shape_meta=shape_meta,
                enable_render=False,
            )
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    wrapper_factory(
                        env=robomimic_env,
                        shape_meta=shape_meta,
                        init_state=None,
                        render_obs_key=render_obs_key,
                        **wrapper_kwargs,
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec='h264',
                        input_pix_fmt='rgb24',
                        crf=crf,
                        thread_type='FRAME',
                        thread_count=1
                    ),
                    file_path=None,
                    steps_per_render=steps_per_render
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps
            )

        env_fns = [env_fn] * n_envs
        env_seeds = list()
        env_prefixs = list()
        env_init_fn_dills = list()

        # train
        with h5py.File(dataset_path, 'r') as f:
            for i in range(n_train):
                train_idx = train_start_idx + i
                enable_render = i < n_train_vis
                init_state = f[f'data/demo_{train_idx}/states'][0]

                def init_fn(env, init_state=init_state, 
                    enable_render=enable_render):
                    # setup rendering
                    # video_wrapper
                    assert isinstance(env.env, VideoRecordingWrapper)
                    env.env.video_recoder.stop()
                    env.env.file_path = None
                    if enable_render:
                        filename = pathlib.Path(output_dir).joinpath(
                            'media', wv.util.generate_id() + ".mp4")
                        filename.parent.mkdir(parents=False, exist_ok=True)
                        filename = str(filename)
                        env.env.file_path = filename

                    # switch to init_state reset
                    assert isinstance(env.env.env, wrapper_factory)
                    env.env.env.init_state = init_state

                env_seeds.append(train_idx)
                env_prefixs.append('train/')
                env_init_fn_dills.append(dill.dumps(init_fn))
        
        # test
        for i in range(n_test):
            seed = test_start_seed + i
            enable_render = i < n_test_vis

            def init_fn(env, seed=seed, 
                enable_render=enable_render):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', wv.util.generate_id() + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # switch to seed reset
                assert isinstance(env.env.env, wrapper_factory)
                env.env.env.init_state = None
                env.seed(seed)

            env_seeds.append(seed)
            env_prefixs.append('test/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        env = AsyncVectorEnv(env_fns, dummy_env_fn=dummy_env_fn)

        self.env_meta = env_meta
        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.rotation_transformer = rotation_transformer
        self.abs_action = abs_action
        self.tqdm_interval_sec = tqdm_interval_sec
        self.eef_correction = bool(eef_correction)
        self.eef_replan = bool(eef_replan)
        self.max_rewards = {}
        for prefix in self.env_prefixs:
            self.max_rewards[prefix] = 0

    def run(self, policy: BaseImagePolicy):
        device = policy.device
        dtype = policy.dtype
        env = self.env
        
        # plan for rollout
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        # allocate data
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits
        all_eef_correction_stats = [None] * n_inits
        all_eef_replan_stats = [None] * n_inits

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0,this_n_active_envs)
            
            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]]*n_diff)
            assert len(this_init_fns) == n_envs

            # init envs
            env.call_each('run_dill_function', 
                args_list=[(x,) for x in this_init_fns])

            # start rollout
            obs = env.reset()
            past_action = None
            policy.reset()

            env_name = self.env_meta['env_name']
            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval {env_name}Image {chunk_idx+1}/{n_chunks}", 
                leave=False, mininterval=self.tqdm_interval_sec)
            
            done = False
            while not done:
                # create obs dict
                np_obs_dict = dict(obs)
                if self.past_action and (past_action is not None):
                    # TODO: not tested
                    np_obs_dict['past_action'] = past_action[
                        :,-(self.n_obs_steps-1):].astype(np.float32)
                
                # device transfer
                obs_dict = dict_apply(np_obs_dict, 
                    lambda x: torch.from_numpy(x).to(
                        device=device))

                # run policy
                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)

                # device_transfer
                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())

                action = np_action_dict['action']
                if not np.all(np.isfinite(action)):
                    print(action)
                    raise RuntimeError("Nan or Inf action")
                
                # step env
                env_action = action
                if self.abs_action:
                    env_action = self.undo_transform_action(action)

                if self.eef_correction or self.eef_replan:
                    required = {
                        "future_eef_pos",
                        "future_eef_rot",
                    }
                    if self.eef_correction:
                        required.add("eef_correction_action")
                    missing = required.difference(np_action_dict)
                    if missing:
                        raise RuntimeError(
                            "EEF correction/replan requires a joint Future EEF "
                            f"checkpoint; missing outputs: {sorted(missing)}"
                        )
                    future_eef_pos = np_action_dict["future_eef_pos"]
                    future_eef_rot = np_action_dict["future_eef_rot"]
                    expected_pos = (n_envs, action.shape[1], 3)
                    expected_rot = (n_envs, action.shape[1], 3, 3)
                    if future_eef_pos.shape != expected_pos:
                        raise RuntimeError(
                            f"Future EEF positions have shape {future_eef_pos.shape}, "
                            f"expected {expected_pos}"
                        )
                    if future_eef_rot.shape != expected_rot:
                        raise RuntimeError(
                            f"Future EEF rotations have shape {future_eef_rot.shape}, "
                            f"expected {expected_rot}"
                        )
                    if self.eef_correction:
                        correction_action = self.undo_transform_action(
                            np_action_dict["eef_correction_action"]
                        )
                        expected_action = (n_envs, action.shape[1], 7)
                        if correction_action.shape != expected_action:
                            raise RuntimeError(
                                "Correction actions have shape "
                                f"{correction_action.shape}, expected {expected_action}"
                            )
                        env.call_each(
                            "set_eef_correction_plan",
                            args_list=[
                                (
                                    future_eef_pos[i],
                                    future_eef_rot[i],
                                    correction_action[i],
                                )
                                for i in range(n_envs)
                            ],
                        )
                    else:
                        env.call_each(
                            "set_eef_replan_plan",
                            args_list=[
                                (future_eef_pos[i], future_eef_rot[i])
                                for i in range(n_envs)
                            ],
                        )

                obs, reward, done, info = env.step(env_action)
                done = np.all(done)
                past_action = action

                # update pbar
                executed_steps = None
                if isinstance(info, dict):
                    executed_steps = info.get("executed_action_steps")
                elif isinstance(info, (list, tuple)):
                    executed_steps = [
                        item["executed_action_steps"]
                        for item in info
                        if isinstance(item, dict)
                        and "executed_action_steps" in item
                    ]
                progress_steps = (
                    action.shape[1]
                    if executed_steps is None or len(np.atleast_1d(executed_steps)) == 0
                    else int(np.max(executed_steps))
                )
                pbar.update(progress_steps)
            pbar.close()

            # collect data for this round
            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]
            if self.eef_correction:
                all_eef_correction_stats[this_global_slice] = env.call(
                    "get_attr", "eef_correction_stats"
                )[this_local_slice]
            if self.eef_replan:
                all_eef_replan_stats[this_global_slice] = env.call(
                    "get_attr", "eef_replan_stats"
                )[this_local_slice]
        # clear out video buffer
        _ = env.reset()
        
        # log
        max_rewards = collections.defaultdict(list)
        feedback_metrics = collections.defaultdict(list)
        log_data = dict()
        # results reported in the paper are generated using the commented out line below
        # which will only report and average metrics from first n_envs initial condition and seeds
        # fortunately this won't invalidate our conclusion since
        # 1. This bug only affects the variance of metrics, not their mean
        # 2. All baseline methods are evaluated using the same code
        # to completely reproduce reported numbers, uncomment this line:
        # for i in range(len(self.env_fns)):
        # and comment out this line
        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)
            log_data[prefix+f'sim_max_reward_{seed}'] = max_reward
            if self.eef_correction:
                episode_stats = all_eef_correction_stats[i]
                if episode_stats is None:
                    raise RuntimeError(
                        f"Missing EEF correction statistics for rollout {i}"
                    )
                for key, value in episode_stats.items():
                    numeric_value = float(value)
                    log_data[
                        prefix + f"eef_correction/{key}_{seed}"
                    ] = numeric_value
                    feedback_metrics[(prefix, f"eef_correction/{key}")].append(
                        numeric_value
                    )
            if self.eef_replan:
                episode_stats = all_eef_replan_stats[i]
                if episode_stats is None:
                    raise RuntimeError(
                        f"Missing EEF replan statistics for rollout {i}"
                    )
                for key, value in episode_stats.items():
                    numeric_value = float(value)
                    log_data[prefix + f"eef_replan/{key}_{seed}"] = numeric_value
                    feedback_metrics[(prefix, f"eef_replan/{key}")].append(
                        numeric_value
                    )

            # visualize sim
            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix+f'sim_video_{seed}'] = sim_video
        
        # log aggregate metrics
        for prefix, value in max_rewards.items():
            name = prefix+'mean_score'
            value = np.mean(value)
            log_data[name] = value
            self.max_rewards[prefix] = max(self.max_rewards[prefix], value)
            log_data[prefix+'max_score'] = self.max_rewards[prefix]

        for (prefix, key), values in feedback_metrics.items():
            log_data[prefix + key + "_mean"] = float(np.mean(values))

        return log_data

    def undo_transform_action(self, action):
        raw_shape = action.shape
        if raw_shape[-1] == 20:
            # dual arm
            action = action.reshape(-1,2,10)

        d_rot = action.shape[-1] - 4
        pos = action[...,:3]
        rot = action[...,3:3+d_rot]
        gripper = action[...,[-1]]
        rot = self.rotation_transformer.inverse(rot)
        uaction = np.concatenate([
            pos, rot, gripper
        ], axis=-1)

        if raw_shape[-1] == 20:
            # dual arm
            uaction = uaction.reshape(*raw_shape[:-1], 14)

        return uaction
