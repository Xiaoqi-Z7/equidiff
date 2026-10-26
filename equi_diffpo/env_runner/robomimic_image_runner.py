import os
import json
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


def eef_pose_error(actual_position, actual_quaternion_xyzw,
        predicted_position, predicted_rotation):
    """Return translation (m) and geodesic rotation (rad) errors."""
    actual_position = np.asarray(actual_position, dtype=np.float64)
    actual_quaternion = np.asarray(
        actual_quaternion_xyzw, dtype=np.float64
    )
    quaternion_norm = np.linalg.norm(actual_quaternion)
    if quaternion_norm <= 1e-12:
        raise ValueError("Observed EEF quaternion has zero norm")
    x, y, z, w = actual_quaternion / quaternion_norm
    actual_rotation = np.asarray([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])
    predicted_position = np.asarray(predicted_position, dtype=np.float64)
    predicted_rotation = np.asarray(predicted_rotation, dtype=np.float64)
    relative_rotation = predicted_rotation.T @ actual_rotation
    rotation_error = float(np.arccos(np.clip(
        (np.trace(relative_rotation) - 1.0) / 2.0,
        -1.0,
        1.0,
    )))
    position_error = float(np.linalg.norm(
        actual_position - predicted_position
    ))
    return position_error, rotation_error


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
        ):
        super().__init__(output_dir)

        if n_envs is None:
            n_envs = n_train + n_test
        if wrapper_factory is None:
            wrapper_factory = RobomimicImageWrapper

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
                        render_obs_key=render_obs_key
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
                        render_obs_key=render_obs_key
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
        self.max_rewards = {}
        for prefix in self.env_prefixs:
            self.max_rewards[prefix] = 0

    def run(self, policy: BaseImagePolicy, eef_planner=None,
            eef_replan_config=None, planner_servo_config=None):
        device = policy.device
        dtype = policy.dtype
        env = self.env
        adaptive_replan = eef_replan_config is not None
        planner_pose_servo = planner_servo_config is not None
        if adaptive_replan and planner_pose_servo:
            raise ValueError(
                "Adaptive deviation replanning and planner pose servo are "
                "mutually exclusive"
            )
        if (adaptive_replan or planner_pose_servo) and eef_planner is None:
            raise ValueError("EEF replanning/servo requires an EEF planner")
        if (adaptive_replan or planner_pose_servo) and self.past_action:
            raise NotImplementedError(
                "EEF replanning/servo does not support past_action input"
            )
        
        # plan for rollout
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        # allocate data
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits
        all_eef_deviation_records = [list() for _ in range(n_inits)]

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
            if eef_planner is not None:
                eef_planner.reset()
            chunk_eef_records = [list() for _ in range(n_envs)]
            episode_steps = [0 for _ in range(n_envs)]
            plan_index = 0

            env_name = self.env_meta['env_name']
            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval {env_name}Image {chunk_idx+1}/{n_chunks}", 
                leave=False, mininterval=self.tqdm_interval_sec)
            
            done = False
            if adaptive_replan:
                chunk_eef_records = self._run_adaptive_replan_batch(
                    policy=policy,
                    eef_planner=eef_planner,
                    obs=obs,
                    pbar=pbar,
                    config=eef_replan_config,
                )
                done = True
            elif planner_pose_servo:
                chunk_eef_records = self._run_planner_pose_servo_batch(
                    policy=policy,
                    eef_planner=eef_planner,
                    obs=obs,
                    pbar=pbar,
                    config=planner_servo_config,
                )
                done = True
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
                    reference_dict = (
                        eef_planner.predict_reference(obs_dict)
                        if eef_planner is not None
                        else None
                    )

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

                obs, reward, done, info = env.step(env_action)
                if reference_dict is not None:
                    np_reference_dict = dict_apply(
                        reference_dict,
                        lambda x: x.detach().to('cpu').numpy(),
                    )
                    per_env_step_observations = env.call(
                        'get_attr', 'last_step_observations'
                    )
                    for env_idx in range(this_n_active_envs):
                        step_observations = per_env_step_observations[env_idx]
                        num_executed = min(
                            len(step_observations),
                            action.shape[1],
                            np_reference_dict['future_eef_pos'].shape[1],
                        )
                        for horizon_idx in range(num_executed):
                            actual = step_observations[horizon_idx]
                            predicted_pos = np_reference_dict[
                                'future_eef_pos'
                            ][env_idx, horizon_idx]
                            predicted_rot = np_reference_dict[
                                'future_eef_rot'
                            ][env_idx, horizon_idx]
                            position_error, rotation_error = eef_pose_error(
                                actual['robot0_eef_pos'],
                                actual['robot0_eef_quat'],
                                predicted_pos,
                                predicted_rot,
                            )
                            chunk_eef_records[env_idx].append({
                                'episode_step': int(episode_steps[env_idx] + horizon_idx),
                                'plan_index': int(plan_index),
                                'horizon_index': int(horizon_idx),
                                'position_error_m': position_error,
                                'rotation_error_rad': rotation_error,
                                'predicted_position': predicted_pos.tolist(),
                                'actual_position': np.asarray(
                                    actual['robot0_eef_pos'], dtype=np.float64
                                ).tolist(),
                                'planner_gripper_command': float(
                                    np_reference_dict['act_grip'][env_idx, horizon_idx, 0]
                                ),
                                'actionflow_gripper_command': float(
                                    action[env_idx, horizon_idx, -1]
                                ),
                            })
                        episode_steps[env_idx] += num_executed
                done = np.all(done)
                past_action = action
                plan_index += 1

                # update pbar
                pbar.update(action.shape[1])
            pbar.close()

            # collect data for this round
            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]
            for local_idx in range(this_n_active_envs):
                all_eef_deviation_records[start + local_idx].extend(
                    chunk_eef_records[local_idx]
                )
        # clear out video buffer
        _ = env.reset()
        
        # log
        max_rewards = collections.defaultdict(list)
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

            # visualize sim
            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix+f'sim_video_{seed}'] = sim_video

        if eef_planner is not None:
            output_path = pathlib.Path(self.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            records_path = output_path / 'eef_deviation_records.jsonl'
            flat_records = []
            with records_path.open('w', encoding='utf-8') as file:
                for rollout_idx, records in enumerate(all_eef_deviation_records):
                    reward = float(np.max(all_rewards[rollout_idx]))
                    successful = reward >= 1.0 - 1e-8
                    for record in records:
                        row = {
                            'split': self.env_prefixs[rollout_idx].rstrip('/'),
                            'seed': int(self.env_seeds[rollout_idx]),
                            'successful': successful,
                            'max_reward': reward,
                            **record,
                        }
                        flat_records.append(row)
                        file.write(json.dumps(row) + '\n')

            summary = {
                'records_path': str(records_path),
                'num_records': len(flat_records),
                'num_episodes': n_inits,
                'adaptive_replan': adaptive_replan,
                'planner_pose_servo': planner_pose_servo,
                'replan_config': eef_replan_config,
                'planner_servo_config': planner_servo_config,
                'total_replans': int(sum(
                    bool(row.get('replan_triggered', False))
                    for row in flat_records
                )),
                'total_discarded_actions': int(sum(
                    int(row.get('discarded_actions', 0))
                    for row in flat_records
                )),
                'groups': {},
            }
            if planner_pose_servo:
                terminal_servo_rows = [
                    row for row in flat_records
                    if row.get('servo_target_reached', False)
                    or row.get('servo_target_timeout', False)
                    or row.get('servo_no_progress', False)
                ]
                summary['servo'] = {
                    'targets_reached': int(sum(
                        bool(row.get('servo_target_reached', False))
                        for row in flat_records
                    )),
                    'target_timeouts': int(sum(
                        bool(row.get('servo_target_timeout', False))
                        for row in flat_records
                    )),
                    'no_progress_events': int(sum(
                        bool(row.get('servo_no_progress', False))
                        for row in flat_records
                    )),
                    'mean_steps_per_terminal_target': (
                        float(np.mean([
                            row['servo_attempt'] for row in terminal_servo_rows
                        ])) if terminal_servo_rows else None
                    ),
                }
            for group_name, desired_success in (
                ('success', True), ('failure', False), ('all', None)
            ):
                rows = [
                    row for row in flat_records
                    if desired_success is None
                    or row['successful'] is desired_success
                ]
                pos = np.asarray(
                    [row['position_error_m'] for row in rows], dtype=np.float64
                )
                rot = np.asarray(
                    [row['rotation_error_rad'] for row in rows], dtype=np.float64
                )
                summary['groups'][group_name] = {
                    'count': int(len(rows)),
                    'position_error_m_mean': float(pos.mean()) if len(pos) else None,
                    'position_error_m_p50': float(np.percentile(pos, 50)) if len(pos) else None,
                    'position_error_m_p90': float(np.percentile(pos, 90)) if len(pos) else None,
                    'position_error_m_p95': float(np.percentile(pos, 95)) if len(pos) else None,
                    'rotation_error_rad_mean': float(rot.mean()) if len(rot) else None,
                    'rotation_error_rad_p50': float(np.percentile(rot, 50)) if len(rot) else None,
                    'rotation_error_rad_p90': float(np.percentile(rot, 90)) if len(rot) else None,
                    'rotation_error_rad_p95': float(np.percentile(rot, 95)) if len(rot) else None,
                }
            summary_path = output_path / 'eef_deviation_summary.json'
            summary_path.write_text(
                json.dumps(summary, indent=2) + '\n', encoding='utf-8'
            )
            log_data['eef_planner/deviation_records'] = len(flat_records)
            log_data['eef_planner/total_replans'] = summary['total_replans']
            log_data['eef_planner/total_discarded_actions'] = summary[
                'total_discarded_actions'
            ]
            log_data['eef_planner/episodes_with_replan'] = len({
                (row['split'], row['seed']) for row in flat_records
                if row.get('replan_triggered', False)
            })
            if planner_pose_servo:
                log_data['eef_servo/targets_reached'] = summary[
                    'servo'
                ]['targets_reached']
                log_data['eef_servo/target_timeouts'] = summary[
                    'servo'
                ]['target_timeouts']
                log_data['eef_servo/no_progress_events'] = summary[
                    'servo'
                ]['no_progress_events']
                mean_servo_steps = summary['servo'][
                    'mean_steps_per_terminal_target'
                ]
                if mean_servo_steps is not None:
                    log_data[
                        'eef_servo/mean_steps_per_terminal_target'
                    ] = mean_servo_steps
            if flat_records:
                log_data['eef_planner/position_error_m_mean'] = summary[
                    'groups'
                ]['all']['position_error_m_mean']
                log_data['eef_planner/rotation_error_rad_mean'] = summary[
                    'groups'
                ]['all']['rotation_error_rad_mean']
        
        # log aggregate metrics
        for prefix, value in max_rewards.items():
            name = prefix+'mean_score'
            value = np.mean(value)
            log_data[name] = value
            self.max_rewards[prefix] = max(self.max_rewards[prefix], value)
            log_data[prefix+'max_score'] = self.max_rewards[prefix]

        return log_data

    def _run_planner_pose_servo_batch(self, policy, eef_planner, obs, pbar,
            config):
        """Track planner waypoints with repeated absolute OSC targets."""
        env = self.env
        n_envs = len(self.env_fns)
        device = policy.device
        position_tolerance = float(config['position_tolerance_m'])
        rotation_tolerance = float(config['rotation_tolerance_rad'])
        max_steps_per_target = int(config['max_steps_per_target'])
        no_progress_limit = int(config['no_progress_steps'])
        max_replans = int(config['max_replans_per_episode'])
        grip_source = str(config['gripper_source'])
        progress_position_epsilon = float(
            config.get('progress_position_epsilon_m', 1e-4)
        )
        progress_rotation_epsilon = float(
            config.get('progress_rotation_epsilon_rad', np.radians(0.1))
        )

        done = np.zeros(n_envs, dtype=bool)
        needs_plan = np.ones(n_envs, dtype=bool)
        servo_action_plans = [None] * n_envs
        actionflow_plans = [None] * n_envs
        reference_plans = [None] * n_envs
        target_cursors = np.zeros(n_envs, dtype=np.int64)
        servo_attempts = np.zeros(n_envs, dtype=np.int64)
        episode_steps = np.zeros(n_envs, dtype=np.int64)
        no_progress_counts = np.zeros(n_envs, dtype=np.int64)
        best_position_errors = np.full(n_envs, np.inf)
        best_rotation_errors = np.full(n_envs, np.inf)
        replan_counts = np.zeros(n_envs, dtype=np.int64)
        next_plan_indices = np.zeros(n_envs, dtype=np.int64)
        current_plan_indices = np.full(n_envs, -1, dtype=np.int64)
        records = [list() for _ in range(n_envs)]

        while not np.all(done):
            planning_indices = np.flatnonzero(np.logical_and(~done, needs_plan))
            if len(planning_indices):
                subset_obs = {
                    key: value[planning_indices]
                    for key, value in obs.items()
                }
                obs_dict = dict_apply(
                    subset_obs,
                    lambda value: torch.from_numpy(value).to(device=device),
                )
                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)
                    reference_dict = eef_planner.predict_reference(obs_dict)
                    planner_actions = eef_planner.reference_to_action(
                        obs_dict, reference_dict
                    )
                action_batch = action_dict['action'].detach().to('cpu').numpy()
                reference_batch = dict_apply(
                    reference_dict,
                    lambda value: value.detach().to('cpu').numpy(),
                )
                planner_action_batch = planner_actions.detach().to('cpu').numpy()
                num_targets = min(
                    action_batch.shape[1], planner_action_batch.shape[1]
                )
                planner_action_batch = planner_action_batch[:, :num_targets].copy()
                action_batch = action_batch[:, :num_targets]
                if grip_source == 'actionflow':
                    planner_action_batch[..., -1:] = action_batch[..., -1:]
                elif grip_source != 'planner':
                    raise ValueError(
                        "Planner servo gripper source must be 'actionflow' "
                        f"or 'planner', got {grip_source!r}"
                    )
                if not np.all(np.isfinite(planner_action_batch)):
                    raise RuntimeError("NaN or Inf planner servo target")

                for subset_idx, env_idx in enumerate(planning_indices):
                    servo_action_plans[env_idx] = planner_action_batch[subset_idx]
                    actionflow_plans[env_idx] = action_batch[subset_idx]
                    reference_plans[env_idx] = {
                        key: value[subset_idx, :num_targets]
                        for key, value in reference_batch.items()
                    }
                    target_cursors[env_idx] = 0
                    servo_attempts[env_idx] = 0
                    no_progress_counts[env_idx] = 0
                    best_position_errors[env_idx] = np.inf
                    best_rotation_errors[env_idx] = np.inf
                    needs_plan[env_idx] = False
                    current_plan_indices[env_idx] = next_plan_indices[env_idx]
                    next_plan_indices[env_idx] += 1

            first_plan = next(
                servo_action_plans[index] for index in range(n_envs)
                if servo_action_plans[index] is not None
            )
            step_actions = np.zeros(
                (n_envs, 1, first_plan.shape[-1]), dtype=first_plan.dtype
            )
            active_indices = np.flatnonzero(~done)
            executed_targets = {}
            for env_idx in active_indices:
                target_idx = int(target_cursors[env_idx])
                plan = servo_action_plans[env_idx]
                if plan is None or target_idx >= len(plan):
                    raise RuntimeError(
                        f"Environment {env_idx} has no planner servo target"
                    )
                step_actions[env_idx, 0] = plan[target_idx]
                executed_targets[env_idx] = target_idx

            env_action = (
                self.undo_transform_action(step_actions)
                if self.abs_action else step_actions
            )
            obs, reward, step_done, info = env.step(env_action)
            per_env_observations = env.call(
                'get_attr', 'last_step_observations'
            )

            for env_idx in active_indices:
                target_idx = executed_targets[env_idx]
                step_observations = per_env_observations[env_idx]
                if len(step_observations) != 1:
                    raise RuntimeError(
                        "Planner pose servo requires exactly one executed "
                        f"control step, got {len(step_observations)}"
                    )
                actual = step_observations[0]
                reference = reference_plans[env_idx]
                predicted_position = reference['future_eef_pos'][target_idx]
                predicted_rotation = reference['future_eef_rot'][target_idx]
                position_error, rotation_error = eef_pose_error(
                    actual['robot0_eef_pos'],
                    actual['robot0_eef_quat'],
                    predicted_position,
                    predicted_rotation,
                )
                servo_attempts[env_idx] += 1
                reached = (
                    position_error <= position_tolerance
                    and rotation_error <= rotation_tolerance
                )
                improved = (
                    position_error
                    < best_position_errors[env_idx] - progress_position_epsilon
                    or rotation_error
                    < best_rotation_errors[env_idx] - progress_rotation_epsilon
                )
                best_position_errors[env_idx] = min(
                    best_position_errors[env_idx], position_error
                )
                best_rotation_errors[env_idx] = min(
                    best_rotation_errors[env_idx], rotation_error
                )
                no_progress_counts[env_idx] = (
                    0 if improved else no_progress_counts[env_idx] + 1
                )
                attempt_at_decision = int(servo_attempts[env_idx])
                no_progress_at_decision = int(
                    no_progress_counts[env_idx]
                )
                timed_out = (
                    not reached
                    and servo_attempts[env_idx] >= max_steps_per_target
                )
                no_progress = (
                    not reached
                    and no_progress_limit > 0
                    and no_progress_counts[env_idx] >= no_progress_limit
                )
                target_failed = timed_out or no_progress
                remaining_targets = (
                    len(servo_action_plans[env_idx]) - target_idx
                )
                can_replan = (
                    target_failed
                    and not bool(step_done[env_idx])
                    and replan_counts[env_idx] < max_replans
                )
                replan_suppressed = (
                    target_failed
                    and not bool(step_done[env_idx])
                    and replan_counts[env_idx] >= max_replans
                )

                if can_replan:
                    replan_counts[env_idx] += 1
                    needs_plan[env_idx] = True
                elif reached or target_failed:
                    target_cursors[env_idx] += 1
                    servo_attempts[env_idx] = 0
                    no_progress_counts[env_idx] = 0
                    best_position_errors[env_idx] = np.inf
                    best_rotation_errors[env_idx] = np.inf
                    if target_cursors[env_idx] >= len(
                        servo_action_plans[env_idx]
                    ):
                        needs_plan[env_idx] = True

                records[env_idx].append({
                    'episode_step': int(episode_steps[env_idx]),
                    'plan_index': int(current_plan_indices[env_idx]),
                    'horizon_index': int(target_idx),
                    'servo_attempt': attempt_at_decision,
                    'position_error_m': position_error,
                    'rotation_error_rad': rotation_error,
                    'predicted_position': predicted_position.tolist(),
                    'actual_position': np.asarray(
                        actual['robot0_eef_pos'], dtype=np.float64
                    ).tolist(),
                    'planner_gripper_command': float(
                        reference['act_grip'][target_idx, 0]
                    ),
                    'actionflow_gripper_command': float(
                        actionflow_plans[env_idx][target_idx, -1]
                    ),
                    'executed_gripper_command': float(
                        servo_action_plans[env_idx][target_idx, -1]
                    ),
                    'servo_target_reached': bool(reached),
                    'servo_target_timeout': bool(timed_out),
                    'servo_no_progress': bool(no_progress),
                    'servo_no_progress_count': no_progress_at_decision,
                    'replan_triggered': bool(can_replan),
                    'replan_count': int(replan_counts[env_idx]),
                    'discarded_actions': int(
                        remaining_targets if can_replan else 0
                    ),
                    'replan_suppressed_max': bool(replan_suppressed),
                    'gripper_source': grip_source,
                })
                episode_steps[env_idx] += 1

            done = np.logical_or(done, np.asarray(step_done, dtype=bool))
            pbar.update(1)

        return records

    def _run_adaptive_replan_batch(self, policy, eef_planner, obs, pbar,
            config):
        """Execute control steps individually and replan per environment.

        Vector environments remain simulator-step synchronized, while every
        environment owns an independent action/reference chunk and cursor.
        Triggering invalidates only that environment's unexecuted actions.
        The next plan is generated from its latest observation.
        """
        env = self.env
        n_envs = len(self.env_fns)
        device = policy.device
        position_threshold = float(config['position_threshold_m'])
        rotation_threshold = float(config['rotation_threshold_rad'])
        consecutive_steps = int(config['consecutive_steps'])
        max_replans = int(config['max_replans_per_episode'])

        done = np.zeros(n_envs, dtype=bool)
        needs_plan = np.ones(n_envs, dtype=bool)
        action_plans = [None] * n_envs
        reference_plans = [None] * n_envs
        cursors = np.zeros(n_envs, dtype=np.int64)
        episode_steps = np.zeros(n_envs, dtype=np.int64)
        violation_streaks = np.zeros(n_envs, dtype=np.int64)
        replan_counts = np.zeros(n_envs, dtype=np.int64)
        next_plan_indices = np.zeros(n_envs, dtype=np.int64)
        current_plan_indices = np.full(n_envs, -1, dtype=np.int64)
        records = [list() for _ in range(n_envs)]

        while not np.all(done):
            planning_indices = np.flatnonzero(np.logical_and(~done, needs_plan))
            if len(planning_indices):
                subset_obs = {
                    key: value[planning_indices]
                    for key, value in obs.items()
                }
                obs_dict = dict_apply(
                    subset_obs,
                    lambda value: torch.from_numpy(value).to(device=device),
                )
                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)
                    reference_dict = eef_planner.predict_reference(obs_dict)
                action_batch = dict_apply(
                    action_dict,
                    lambda value: value.detach().to('cpu').numpy(),
                )['action']
                reference_batch = dict_apply(
                    reference_dict,
                    lambda value: value.detach().to('cpu').numpy(),
                )
                if not np.all(np.isfinite(action_batch)):
                    raise RuntimeError(
                        "NaN or Inf action in adaptive replanning"
                    )
                for subset_idx, env_idx in enumerate(planning_indices):
                    action_plans[env_idx] = action_batch[subset_idx]
                    reference_plans[env_idx] = {
                        key: value[subset_idx]
                        for key, value in reference_batch.items()
                    }
                    cursors[env_idx] = 0
                    needs_plan[env_idx] = False
                    current_plan_indices[env_idx] = next_plan_indices[env_idx]
                    next_plan_indices[env_idx] += 1

            first_plan = next(
                action_plans[index] for index in range(n_envs)
                if action_plans[index] is not None
            )
            step_actions = np.zeros(
                (n_envs, 1, first_plan.shape[-1]), dtype=first_plan.dtype
            )
            active_indices = np.flatnonzero(~done)
            executed_cursors = {}
            for env_idx in active_indices:
                cursor = int(cursors[env_idx])
                plan = action_plans[env_idx]
                if plan is None or cursor >= len(plan):
                    raise RuntimeError(
                        f"Environment {env_idx} has no executable action plan"
                    )
                step_actions[env_idx, 0] = plan[cursor]
                executed_cursors[env_idx] = cursor

            env_action = (
                self.undo_transform_action(step_actions)
                if self.abs_action else step_actions
            )
            obs, reward, step_done, info = env.step(env_action)
            per_env_observations = env.call(
                'get_attr', 'last_step_observations'
            )

            for env_idx in active_indices:
                cursor = executed_cursors[env_idx]
                step_observations = per_env_observations[env_idx]
                if len(step_observations) != 1:
                    raise RuntimeError(
                        "Adaptive replanning requires exactly one executed "
                        f"control step, got {len(step_observations)}"
                    )
                actual = step_observations[0]
                reference = reference_plans[env_idx]
                predicted_position = reference['future_eef_pos'][cursor]
                predicted_rotation = reference['future_eef_rot'][cursor]
                position_error, rotation_error = eef_pose_error(
                    actual['robot0_eef_pos'],
                    actual['robot0_eef_quat'],
                    predicted_position,
                    predicted_rotation,
                )
                threshold_violated = (
                    position_error > position_threshold
                    or rotation_error > rotation_threshold
                )
                violation_streaks[env_idx] = (
                    violation_streaks[env_idx] + 1
                    if threshold_violated else 0
                )
                streak_at_decision = int(violation_streaks[env_idx])
                remaining_actions = len(action_plans[env_idx]) - cursor - 1
                reached_trigger_streak = (
                    streak_at_decision >= consecutive_steps
                )
                can_replan = (
                    reached_trigger_streak
                    and remaining_actions > 0
                    and not bool(step_done[env_idx])
                    and replan_counts[env_idx] < max_replans
                )
                replan_suppressed = (
                    reached_trigger_streak
                    and remaining_actions > 0
                    and not bool(step_done[env_idx])
                    and replan_counts[env_idx] >= max_replans
                )
                if can_replan:
                    replan_counts[env_idx] += 1
                    needs_plan[env_idx] = True
                    violation_streaks[env_idx] = 0
                else:
                    cursors[env_idx] += 1
                    if cursors[env_idx] >= len(action_plans[env_idx]):
                        needs_plan[env_idx] = True

                records[env_idx].append({
                    'episode_step': int(episode_steps[env_idx]),
                    'plan_index': int(current_plan_indices[env_idx]),
                    'horizon_index': int(cursor),
                    'position_error_m': position_error,
                    'rotation_error_rad': rotation_error,
                    'predicted_position': predicted_position.tolist(),
                    'actual_position': np.asarray(
                        actual['robot0_eef_pos'], dtype=np.float64
                    ).tolist(),
                    'planner_gripper_command': float(
                        reference['act_grip'][cursor, 0]
                    ),
                    'actionflow_gripper_command': float(
                        action_plans[env_idx][cursor, -1]
                    ),
                    'threshold_violated': bool(threshold_violated),
                    'violation_streak': streak_at_decision,
                    'replan_triggered': bool(can_replan),
                    'replan_count': int(replan_counts[env_idx]),
                    'discarded_actions': int(
                        remaining_actions if can_replan else 0
                    ),
                    'replan_suppressed_max': bool(replan_suppressed),
                })
                episode_steps[env_idx] += 1

            done = np.logical_or(done, np.asarray(step_done, dtype=bool))
            pbar.update(1)

        return records

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
