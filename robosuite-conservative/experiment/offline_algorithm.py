"""
Set of functions and classes that are modified versions of existing ones in rlkit
"""
import abc
import os

from rlkit.core import logger, eval_util
import gtimer as gt
from rlkit.core.rl_algorithm import _get_epoch_timings
from rlkit.samplers.data_collector import DataCollector
from rlkit.samplers.data_collector import PathCollector
from models.utils.buffer import buffer_to, torchify_buffer
from models.combo.combo_trainer import merge_dict
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt


class OfflineBaseRLAlgorithm(object, metaclass=abc.ABCMeta):
    """
    Base object of offline RL algorithm.
    """
    def __init__(
            self,
            agent,
            trainer,
            evaluation_env,
            evaluation_data_collector: DataCollector,
    ):
        self.agent = agent
        self.trainer = trainer
        self.eval_env = evaluation_env
        self.eval_data_collector = evaluation_data_collector
        self._model_start_epoch = 0
        self._agent_start_epoch = 0
        self._transfer_start_epoch = 0
        self._env_wrapper = evaluation_env.process_images

    def train(self, start_epoch=0):
        self._model_start_epoch = start_epoch
        self._agent_start_epoch = start_epoch
        self._transfer_start_epoch = start_epoch
        self._train()

    def _train(self):
        """
        Train model.
        """
        raise NotImplementedError('_train must implemented by inherited class')

    def _end_epoch(self, epoch, phase='agent'):
        if phase == 'agent':
            self._log_agent_stats(epoch)
        elif phase == 'model':
            self._log_model_stats(epoch)
        elif phase == 'transfer':
            self._log_transfer_stats(epoch)
        else:
            raise NotImplementedError

        self.eval_data_collector.end_epoch(epoch)
        self.trainer.end_epoch(epoch)

    def _get_snapshot(self):
        pass

    def _log_transfer_stats(self, epoch):
        pass

    def _log_model_stats(self, epoch):
        pass

    def _log_agent_stats(self, epoch):
        pass


class OfflineBatchRLAlgorithm(OfflineBaseRLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(
            self,
            device,
            dataset,
            agent,
            trainer,
            replay_buffer,
            evaluation_env,
            evaluation_data_collector: PathCollector,
            batch_size,
            eval_max_path_length,
            log_snapshot_interval,
            num_agent_epochs,
            num_eval_steps_per_epoch,
            num_trains_per_train_loop,
            num_train_loops_per_epoch=1,
            **kwargs
    ):
        super().__init__(
            agent,
            trainer,
            evaluation_env,
            evaluation_data_collector
        )

        self.device = device
        self.dataset = dataset
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.eval_max_path_length = eval_max_path_length
        self.log_snapshot_interval = log_snapshot_interval
        self.num_agent_epochs = num_agent_epochs
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch

    def _train(self):

        # Add dataset to memory
        for idx in range(len(self.dataset)):
            epi = self.dataset.get_episode(idx)
            epi = self._preprocess_episode(epi)
            self.replay_buffer.add_path(epi)

        # Agent training phase
        for epoch in gt.timed_for(
                range(self._agent_start_epoch, self.num_agent_epochs),
                save_itrs=True,
        ):

            self.eval_data_collector.collect_new_paths(
                self.eval_max_path_length,
                self.num_eval_steps_per_epoch,
                discard_incomplete_paths=True,
            )
            gt.stamp('evaluation sampling', unique=False)

            for _ in range(self.num_train_loops_per_epoch):
                self.training_mode(True)
                for __ in range(self.num_trains_per_train_loop):
                    train_data = self.replay_buffer.random_batch(self.batch_size)
                    self.trainer.train(train_data)
                gt.stamp('training', unique=False)
                self.training_mode(False)

            self._end_epoch(epoch, phase='agent')

    def _log_agent_stats(self, epoch):
        # if (epoch % self.log_snapshot_interval) == 0:
        #     policy_snapshot = dict(agent=self.agent)
        #     logger.save_model_params(policy_snapshot, epoch)

        eval_paths = self.eval_data_collector.get_epoch_paths()

        logger.log("Agent epoch {} finished".format(epoch), with_timestamp=False)

        """
        Agent Trainer
        """
        logger.record_dict(self.trainer.get_diagnostics('agent'))

        """
        Evaluation
        """
        logger.record_dict(self.eval_data_collector.get_diagnostics(), prefix='eval/')

        if hasattr(self.eval_env, 'get_diagnostics'):
            logger.record_dict(self.eval_env.get_diagnostics(eval_paths), prefix='eval/')
        logger.record_dict(
            get_custom_generic_path_information(eval_paths, self.eval_max_path_length),
            prefix='eval/',
        )

        """
        Misc
        """
        gt.stamp('logging', unique=False)
        logger.record_dict(_get_epoch_timings())
        logger.record_tabular('Agent Epoch', epoch)
        logger.dump_tabular(file='agent.csv')

    def _preprocess_episode(self, episode):
        data = {}
        num_steps = len(episode['terminals'])
        for key in ['actions', 'rewards', 'terminals']:
            values = []
            for step in range(1, num_steps):
                values.append(episode[key][step])
            data[key] = values

        obses = []
        for step in range(num_steps):
            obs = {
                'robot_states': self.eval_env._process_state(episode['robot_states'][step]),
                'object_states': self.eval_env._process_object(episode['object_states'][step])
            }
            obses.append(obs)
        next_obses = obses[1:]
        obses = obses[:-1]

        data['observations'] = obses
        data['next_observations'] = next_obses
        data['agent_infos'] = [[] for _ in range(num_steps-1)]
        data['env_infos'] = [[] for _ in range(num_steps-1)]

        return data

    def to(self, device):
        for net in self.trainer.networks:
            net.to(device)

    def training_mode(self, mode):
        for net in self.trainer.networks:
            net.train(mode)


class OfflineBatchIMGRLAlgorithm(OfflineBaseRLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(
            self,
            device,
            dataset,
            agent,
            trainer,
            evaluation_env,
            evaluation_data_collector: PathCollector,
            batch_size,
            eval_max_path_length,
            visualize_policy_interval,
            log_snapshot_interval,
            num_agent_epochs,
            num_eval_steps_per_epoch,
            num_trains_per_train_loop,
            num_train_loops_per_epoch=1,
            **kwargs
    ):
        super().__init__(
            agent,
            trainer,
            evaluation_env,
            evaluation_data_collector
        )

        self.device = device
        self.dataset = dataset
        self.data_loader = DataLoader(dataset, batch_size, shuffle=True, num_workers=3,
                                      collate_fn=self._preprocess_episode)
        self.batch_size = batch_size
        self.eval_max_path_length = eval_max_path_length
        self.visualize_policy_interval = visualize_policy_interval
        self.log_snapshot_interval = log_snapshot_interval
        self.num_agent_epochs = num_agent_epochs
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch

    def _train(self):

        # Agent training phase
        for epoch in gt.timed_for(
                range(self._agent_start_epoch, self.num_agent_epochs),
                save_itrs=True,
        ):

            self.eval_data_collector.collect_new_paths(
                self.eval_max_path_length,
                self.num_eval_steps_per_epoch,
                discard_incomplete_paths=True,
            )
            gt.stamp('evaluation sampling', unique=False)

            self.training_mode(True)
            for _ in range(self.num_train_loops_per_epoch):
                self.trainer.train_agent(self.num_trains_per_train_loop, self.data_loader)
                gt.stamp('agent training', unique=False)
            self.training_mode(False)

            self._end_epoch(epoch, phase='agent')

    def _log_agent_stats(self, epoch):
        if (epoch % self.log_snapshot_interval) == 0:
            policy_snapshot = dict(agent=self.agent)
            logger.save_model_params(policy_snapshot, epoch)

        eval_paths = self.eval_data_collector.get_epoch_paths()
        if (epoch % self.visualize_policy_interval) == 0:
            num_epi = len(eval_paths)
            imgs = []
            for idx in range(200):  # log first 200 steps
                for path in eval_paths:
                    imgs.append(path['observations'][idx]['image_obses'])
            imgs = np.stack(imgs, axis=0)
            imgs = buffer_to(imgs, 'cpu')
            imgs = imgs.permute(0, 3, 1, 2) / 255.0
            save_trj = os.path.join(logger.get_snapshot_dir(), 'trj_'+str(epoch)+'.png')
            save_image(imgs, save_trj, nrow=num_epi)

        logger.log("Agent epoch {} finished".format(epoch), with_timestamp=False)

        """
        Agent Trainer
        """
        logger.record_dict(self.trainer.get_diagnostics('agent'))

        """
        Evaluation
        """
        logger.record_dict(self.eval_data_collector.get_diagnostics(), prefix='eval/')

        if hasattr(self.eval_env, 'get_diagnostics'):
            logger.record_dict(self.eval_env.get_diagnostics(eval_paths), prefix='eval/')
        logger.record_dict(
            get_custom_generic_path_information(eval_paths, self.eval_max_path_length),
            prefix='eval/',
        )

        """
        Misc
        """
        gt.stamp('logging', unique=False)
        logger.record_dict(_get_epoch_timings())
        logger.record_tabular('Agent Epoch', epoch)
        logger.dump_tabular(file='agent.csv')

    def _preprocess_episode(self, episodes):

        data = {}
        for key in ['actions', 'rewards', 'terminals']:
            values = []
            for epi in episodes:
                values.append(epi[key][1:])
            values = np.concatenate(values, axis=0)
            data[key] = torchify_buffer(values)

        for key in ['robot_states', 'visual_images']:
            values, next_values = [], []
            for epi in episodes:
                if key == 'visual_images':
                    epi['image_obses'] = self._env_wrapper(epi[key])
                    n_key = 'image_obses'
                else:
                    n_key = key
                values.append(epi[n_key][:-1])
                next_values.append(epi[n_key][1:])

            values = np.concatenate(values, axis=0)
            next_values = np.concatenate(next_values, axis=0)
            data[n_key] = torchify_buffer(values)
            data['next_'+n_key] = torchify_buffer(next_values)

        return data

    def to(self, device):
        for net in self.trainer.networks:
            net.to(device)

    def training_mode(self, mode):
        for net in self.trainer.networks:
            net.train(mode)


class OfflineBatchLatentRLAlgorithm(OfflineBaseRLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(
            self,
            device,
            dataset,
            agent,
            trainer,
            evaluation_env,
            evaluation_data_collector: PathCollector,
            batch_size,
            eval_max_path_length,
            log_snapshot_interval,
            visualize_model_interval,
            visualize_policy_interval,
            num_model_epochs,
            num_agent_epochs,
            samples_per_epoch,
            num_eval_steps_per_epoch,
            num_trains_per_train_loop,
            num_train_loops_per_epoch=1,
            min_num_batch_before_training=0
    ):
        super().__init__(
            agent,
            trainer,
            evaluation_env,
            evaluation_data_collector
        )

        self.device = device
        self.dataset = dataset
        self.data_loader = DataLoader(dataset, batch_size, shuffle=True, num_workers=3,
                                      collate_fn=self._preprocess_episode)
        self.batch_size = batch_size
        self.eval_max_path_length = eval_max_path_length
        self.log_snapshot_interval = log_snapshot_interval
        self.visualize_model_interval = visualize_model_interval
        self.visualize_policy_interval = visualize_policy_interval
        self.num_model_epochs = num_model_epochs
        self.num_agent_epochs = num_agent_epochs
        self.samples_per_epoch = samples_per_epoch
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.min_num_batch_before_training = min_num_batch_before_training

        self.agent.to(self.device)

    def _train(self):

        # Model training phase
        self.trainer.model_training_phase()
        for epoch in gt.timed_for(
                range(self._model_start_epoch, self.num_model_epochs),
                save_itrs=True,
        ):
            self.trainer.fit_model(self.data_loader)
            gt.stamp('model training', unique=False)

            self._end_epoch(epoch, phase='model')

        # Initialize latent buffer and Add samples
        self.trainer.agent_training_phase()
        self._process_dataset_to_real_buffer()

        self.agent.sample_mode()
        for _ in range(self.min_num_batch_before_training):
            self.trainer.sample_data(self.samples_per_epoch, self.data_loader)

        # Agent training phase
        for epoch in gt.timed_for(
                range(self._agent_start_epoch, self.num_agent_epochs),
                save_itrs=True,
        ):
            self.agent.eval_mode()
            self.eval_data_collector.collect_new_paths(
                self.eval_max_path_length,
                self.num_eval_steps_per_epoch,
                discard_incomplete_paths=True,
            )
            gt.stamp('evaluation sampling', unique=False)

            for _ in range(self.num_train_loops_per_epoch):
                self.agent.train_mode()
                self.trainer.train_agent(self.num_trains_per_train_loop)

                self.agent.sample_mode()
                self.trainer.sample_data(self.samples_per_epoch, self.data_loader)

                gt.stamp('agent training', unique=False)

            self._end_epoch(epoch, phase='agent')

    def load_model(self, path):
        try:
            model_snapshot = logger.load_model_params(path)
            self.agent.load_model_snapshot(model_snapshot)
            self.trainer.load_opt_snapshot(model_snapshot, ['model_opt'])
        except:
            print("Load {} Failure. Continue training.".format(path))

    def _log_agent_stats(self, epoch):
        eval_paths = self.eval_data_collector.get_epoch_paths()
        if (epoch % self.visualize_policy_interval) == 0:
            num_epi = len(eval_paths)
            imgs = []
            for idx in range(200):  # log first 200 steps
                for path in eval_paths:
                    imgs.append(path['observations'][idx]['image_obses'])
            imgs = np.stack(imgs, axis=0)
            imgs = buffer_to(imgs, 'cpu')
            imgs = imgs.permute(0, 3, 1, 2) / 255.0
            save_trj = os.path.join(logger.get_snapshot_dir(), 'trj_'+str(epoch)+'.png')
            save_image(imgs, save_trj, nrow=num_epi)

        logger.log("Agent epoch {} finished".format(epoch), with_timestamp=False)

        """
        Agent Trainer
        """
        logger.record_dict(self.trainer.get_diagnostics('agent'))

        """
        Evaluation
        """
        logger.record_dict(self.eval_data_collector.get_diagnostics(), prefix='eval/')

        if hasattr(self.eval_env, 'get_diagnostics'):
            logger.record_dict(self.eval_env.get_diagnostics(eval_paths), prefix='eval/')
        logger.record_dict(
            get_custom_generic_path_information(eval_paths, self.eval_max_path_length),
            prefix='eval/',
        )

        """
        Misc
        """
        gt.stamp('logging', unique=False)
        logger.record_dict(_get_epoch_timings())
        logger.record_tabular('Agent Epoch', epoch)
        logger.dump_tabular(file='agent.csv')

    def _log_model_stats(self, epoch):
        if (epoch % self.log_snapshot_interval) == 0:
            model_snapshot = self.agent.get_model_snapshot()
            opt_snapshot = self.trainer.get_opt_snapshot()
            model_snapshot = merge_dict(model_snapshot, opt_snapshot)
            logger.save_model_params(model_snapshot, epoch)

        if (epoch % self.visualize_model_interval) == 0:
            epi = self.dataset.get_episode(np.random.randint(len(self.dataset)))
            epi = buffer_to(self._preprocess_episode([epi]), self.device)
            recons, rolls = self.trainer.visualize_model(epi)
            save_recon = os.path.join(logger.get_snapshot_dir(), 'recon_'+str(epoch)+'.png')
            save_roll = os.path.join(logger.get_snapshot_dir(), 'roll_'+str(epoch)+'.png')
            save_image(recons, save_recon, nrow=3)
            save_image(rolls, save_roll, nrow=2)

        logger.log("Model epoch {} finished".format(epoch), with_timestamp=False)

        """
        Model Trainer
        """
        logger.record_dict(self.trainer.get_diagnostics('model'))

        gt.stamp('logging', unique=False)
        logger.record_dict(_get_epoch_timings())
        logger.record_tabular('Epoch', epoch)
        logger.dump_tabular(file='model.csv')

    def _process_dataset_to_real_buffer(self):
        for epi_idx in range(len(self.dataset)):
            episode = self.dataset.get_episode(epi_idx)
            episode = buffer_to(self._preprocess_episode([episode]), self.device)
            self.trainer.process_episode_to_real_buffer(episode)

    def _preprocess_episode(self, episodes):
        data = {}
        for key in ['robot_states', 'actions', 'rewards', 'terminals']:
            values = []
            for epi in episodes:
                values.append(epi[key])
            values = np.array(values)
            # [B, T, ...] to [T, B, ...]
            data[key] = torchify_buffer(values).transpose(1, 0)

        images = []
        for epi in episodes:
            imgs = self._env_wrapper(epi['visual_images'])
            images.append(imgs)
        images = np.array(images)
        data['image_obses'] = torchify_buffer(images).transpose(1, 0)

        return data


def rollout(
        env,
        agent,
        max_path_length=np.inf,
        render=False,
        render_kwargs=None,
        video_writer=None,
):
    """
    Custom rollout function that extends the basic rlkit functionality in the following ways:
    - Allows for automatic video writing if @video_writer is specified

    Added args:
        video_writer (imageio.get_writer): If specified, will write image frames to this writer

    The following is pulled directly from the rlkit rollout(...) function docstring:

    The following value for the following keys will be a 2D array, with the
    first dimension corresponding to the time dimension.
     - observations
     - actions
     - rewards
     - next_observations
     - terminals

    The next two elements will be lists of dictionaries, with the index into
    the list being the index into the time
     - agent_infos
     - env_infos
    """
    if render_kwargs is None:
        render_kwargs = {}
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    o = env.reset()
    agent.reset()
    next_o = None
    path_length = 0

    # Only render if specified AND there's no video writer
    if render and video_writer is None:
        env.render(**render_kwargs)

    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)
        next_o, r, d, env_info = env.step(a)
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)

        # Grab image data to write to video writer if specified
        if video_writer is not None:
            # We need to directly grab full observations so we can get image data
            full_obs = env._get_observation()

            # Grab image data (assume relevant camera name is the first in the env camera array)
            img = full_obs[env.camera_names[0] + "_image"]

            # Write to video writer
            video_writer.append_data(img[::-1])

        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
        if render:
            env.render(**render_kwargs)

    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    if len(observations.shape) == 1:
        observations = np.expand_dims(observations, 1)
        next_o = np.array([next_o])
    next_observations = np.vstack(
        (
            observations[1:, :],
            np.expand_dims(next_o, 0)
        )
    )
    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
    )


def get_custom_generic_path_information(paths, path_length, stat_prefix=''):
    """
    Get an OrderedDict with a bunch of statistic names and values.

    Differs from normal rlkit utility function in the following ways:
    Grabs normalized reward / return values where reward is normalized to 1.0
    Grabs cumulative reward specified accumulated at @path_length timestep
    """
    statistics = OrderedDict()

    # Grab returns accumulated up to specified timestep
    for coef in [0.1, 0.2, 0.4, 1.0]:
        max_step = int(path_length * coef)

        returns = [sum(path["rewards"][:max_step]) for path in paths]
        rewards = np.vstack([path["rewards"][:max_step] for path in paths])

        statistics.update(eval_util.create_stats_ordered_dict('Rewards', rewards,
                                                              stat_prefix=stat_prefix+'step'+str(max_step)+'/'))
        statistics.update(eval_util.create_stats_ordered_dict('Returns', returns,
                                                              stat_prefix=stat_prefix+'step'+str(max_step)+'/'))

    actions = [path["actions"] for path in paths]
    if len(actions[0].shape) == 1:
        actions = np.hstack([path["actions"] for path in paths])
    else:
        actions = np.vstack([path["actions"] for path in paths])
    statistics.update(eval_util.create_stats_ordered_dict(
        'Actions', actions, stat_prefix=stat_prefix
    ))
    statistics['Num Paths'] = len(paths)
    statistics[stat_prefix + 'Average Returns'] = eval_util.get_average_returns(paths)

    return statistics