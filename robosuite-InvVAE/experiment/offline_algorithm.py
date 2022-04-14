
from models.utils.buffer import torchify_buffer, buffer_to, numpify_buffer
from models.dataset import CustomDataLoader
from torch.utils.data import DataLoader

import rlkit.torch.pytorch_util as ptu
from rlkit.core import logger, eval_util
from rlkit.core.rl_algorithm import _get_epoch_timings

from collections import OrderedDict
from torchvision.utils import save_image

import imageio as iio
import gtimer as gt
import numpy as np
import torch
import random
import os


class OfflineInvVAETransferAlgorithm(object):

    def __init__(
            self,
            source_agent,
            target_agent,
            transfer_model,

            source_trainer,
            target_trainer,
            transfer_trainer,
            dynamics_trainer,

            eval_source_env,
            eval_target_env,
            eval_source_collector,
            eval_target_collector,
            eval_max_path_length,

            source_general_dataset,
            target_general_dataset,
            source_vae_dataset,
            target_vae_dataset,
            paired_vae_dataset,
            source_eval_dataset,
            target_eval_dataset,

            source_train_dataset,
            target_train_dataset,
            source_train_buffer,
            target_train_buffer,

            transfer_batch_size,
            dynamics_batch_size,

            log_policy_per_epochs,
            log_model_per_epochs,
            log_transfer_per_epochs,
            num_loader_workers,

            num_agent_epochs,
            num_model_epochs,
            num_transfer_epochs,
            num_eval_steps_per_epoch,
            num_trains_per_policy_epoch,
            num_trains_per_transfer_epoch,

            save_snapshot,
            save_interval,
            save_model_path,
            **kwargs
    ):
        # models
        self.source_agent = source_agent
        self.target_agent = target_agent
        self.transfer_model = transfer_model

        # trainers
        self.source_trainer = source_trainer
        self.target_trainer = target_trainer
        self.transfer_trainer = transfer_trainer
        self.dynamics_trainer = dynamics_trainer

        # evaluation envs
        self.eval_source_env = eval_source_env
        self.eval_target_env = eval_target_env
        self.eval_source_collector = eval_source_collector
        self.eval_target_collector = eval_target_collector
        self.eval_max_path_length = eval_max_path_length

        # data loader
        self.source_general_loader = DataLoader(source_general_dataset, dynamics_batch_size,
                                                shuffle=True, drop_last=True, num_workers=num_loader_workers,
                                                collate_fn=lambda x: self._preprocess_episode_steps(x, 'source'))
        self.target_general_loader = DataLoader(target_general_dataset, dynamics_batch_size,
                                                shuffle=True, drop_last=True, num_workers=num_loader_workers,
                                                collate_fn=lambda x: self._preprocess_episode_steps(x, 'target'))
        self.source_vae_loader = CustomDataLoader(source_vae_dataset, transfer_batch_size,
                                                  shuffle=True, drop_last=True, num_workers=num_loader_workers,
                                                  collate_fn=lambda x: self._preprocess_episode_steps(x, 'source'))
        self.target_vae_loader = CustomDataLoader(target_vae_dataset, transfer_batch_size,
                                                  shuffle=True, drop_last=True, num_workers=num_loader_workers,
                                                  collate_fn=lambda x: self._preprocess_episode_steps(x, 'target'))
        self.paired_vae_loader = CustomDataLoader(paired_vae_dataset, transfer_batch_size,
                                                  shuffle=True, drop_last=True, num_workers=num_loader_workers,
                                                  collate_fn=self._preprocess_paired_steps)
        self.source_eval_loader = DataLoader(source_eval_dataset, transfer_batch_size,
                                             shuffle=True, drop_last=True, num_workers=num_loader_workers,
                                             collate_fn=lambda x: self._preprocess_episode_steps(x, 'source'))
        self.target_eval_loader = DataLoader(target_eval_dataset, transfer_batch_size,
                                             shuffle=True, drop_last=True, num_workers=num_loader_workers,
                                             collate_fn=lambda x: self._preprocess_episode_steps(x, 'target'))

        self.source_train_dataset = source_train_dataset
        self.target_train_dataset = target_train_dataset
        self.source_train_buffer = source_train_buffer
        self.target_train_buffer = target_train_buffer

        # training and logging parameters
        self.num_agent_epochs = num_agent_epochs
        self.num_model_epochs = num_model_epochs
        self.num_transfer_epochs = num_transfer_epochs

        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_trains_per_policy_epoch = num_trains_per_policy_epoch
        self.num_trains_per_transfer_epoch = num_trains_per_transfer_epoch

        self.log_policy_per_epochs = log_policy_per_epochs
        self.log_transfer_per_epochs = log_transfer_per_epochs
        self.log_model_per_epochs = log_model_per_epochs

        # start status
        self._model_start_epoch = 0
        self._agent_start_epoch = 0
        self._transfer_start_epoch = 0

        # curr status
        self._model_curr_epoch = 0
        self._agent_curr_epoch = 0
        self._transfer_curr_epoch = 0

        # save snapshot
        self.save_snapshot = save_snapshot
        self.save_interval = save_interval
        self.save_model_path = save_model_path

    def train(self):
        self.transfer_model.eval_mode('all')
        self.source_trainer.eval()
        self.target_trainer.eval()

        for epoch in gt.timed_for(
                range(self._model_start_epoch, self.num_model_epochs),
                save_itrs=True
        ):
            self.dynamics_trainer.train_model(self.source_general_loader,
                                              self.target_general_loader, 'inverse')
            self.dynamics_trainer.eval_model(self.source_eval_loader,
                                             self.target_eval_loader)

            self._model_curr_epoch += 1
            gt.stamp('training', unique=False)
            self._end_epoch(epoch, phase='model')

        self.transfer_model.eval_mode('dynamics_model')

        # Train the Transfer Component
        for epoch in gt.timed_for(
                range(self._transfer_start_epoch, self.num_transfer_epochs),
                save_itrs=True
        ):
            self.transfer_trainer.train_epoch(self.source_vae_loader,
                                              self.target_vae_loader,
                                              self.paired_vae_loader,
                                              self.num_trains_per_transfer_epoch)
            self.transfer_trainer.eval_epoch(self.source_eval_loader,
                                             self.target_eval_loader)

            self._transfer_curr_epoch += 1
            gt.stamp('training', unique=False)
            self._end_epoch(epoch, phase='transfer')

        # Fix the transfer module
        self.transfer_model.eval_mode('all')

        transfer_model = {
            "transfer_model": self.transfer_model
        }
        file_name = os.path.join(logger._snapshot_dir, 'transfer_model.pkl')
        torch.save(transfer_model, file_name)

        # Preload training buffer
        self._preload_dataset(self.source_train_dataset, self.source_train_buffer,
                              lambda x: self._preprocess_latent_steps(x, 'source'))
        self._preload_dataset(self.target_train_dataset, self.target_train_buffer,
                              lambda x: self._preprocess_latent_steps(x, 'target'))

        # Agent training phase
        for epoch in gt.timed_for(
                range(self._agent_start_epoch, self.num_agent_epochs),
                save_itrs=True,
        ):
            self.eval_source_collector.collect_new_paths(
                self.eval_max_path_length,
                self.num_eval_steps_per_epoch,
                discard_incomplete_paths=True,
            )
            self.eval_target_collector.collect_new_paths(
                self.eval_max_path_length,
                self.num_eval_steps_per_epoch,
                discard_incomplete_paths=True,
            )
            gt.stamp('evaluation sampling', unique=False)

            self.source_trainer.train()
            self.source_trainer.train_agent(self.num_trains_per_policy_epoch,
                                            self.source_train_buffer,
                                            lambda x: x)
            self.source_trainer.eval()
            self.target_trainer.train()
            self.target_trainer.train_agent(self.num_trains_per_policy_epoch,
                                            self.target_train_buffer,
                                            lambda x: x)
            self.target_trainer.eval()

            self._agent_curr_epoch += 1
            gt.stamp('training', unique=False)

            self._end_epoch(epoch, phase='agent')

        return file_name

    def _end_epoch(self, epoch, phase='agent'):
        if phase == 'agent':
            self._log_agent_stats(epoch)
        elif phase == 'model':
            self._log_model_stats(epoch)
        elif phase == 'transfer':
            self._log_transfer_stats(epoch)
        else:
            raise NotImplementedError

        self.eval_source_collector.end_epoch(epoch)
        self.eval_target_collector.end_epoch(epoch)
        self.source_trainer.end_epoch(epoch)
        self.target_trainer.end_epoch(epoch)
        self.transfer_trainer.end_epoch(epoch)
        self.dynamics_trainer.end_epoch(epoch)

        if self.save_snapshot and epoch % self.save_interval == 0:
            snapshot = self.get_snapshot()
            logger.save_itr_params(epoch, snapshot, '.tmp')
            logger.atomic_replace(['params.pkl.tmp'])

    def _log_agent_stats(self, epoch):
        logger.log("Agent epoch {} finished".format(epoch), with_timestamp=False)

        logger.record_dict(self.source_trainer.get_diagnostics('agent'), prefix='source/')
        logger.record_dict(self.target_trainer.get_diagnostics('agent'), prefix='target/')

        logger.record_dict(self.eval_source_collector.get_diagnostics(), prefix='source_eval/')
        logger.record_dict(self.eval_target_collector.get_diagnostics(), prefix='target_eval/')

        source_eval_paths = self.eval_source_collector.get_epoch_paths()
        target_eval_paths = self.eval_target_collector.get_epoch_paths()

        if hasattr(self.eval_source_env, 'get_diagnostics'):
            logger.record_dict(self.eval_source_env.get_diagnostics(source_eval_paths), prefix='source_eval/')
        if hasattr(self.eval_target_env, 'get_diagnostics'):
            logger.record_dict(self.eval_target_env.get_diagnostics(target_eval_paths), prefix='target_eval/')

        if len(source_eval_paths) > 0:
            logger.record_dict(
                get_custom_generic_path_information(source_eval_paths, self.eval_max_path_length),
                prefix='source_eval/',
            )
        if len(target_eval_paths) > 0:
            logger.record_dict(
                get_custom_generic_path_information(target_eval_paths, self.eval_max_path_length),
                prefix='target_eval/',
            )

        """
        Misc
        """
        prefixes, paths = [], []

        if len(source_eval_paths) > 0:
            prefixes.append('source')
            paths.append(source_eval_paths)
        if len(target_eval_paths) > 0:
            prefixes.append('target')
            paths.append(target_eval_paths)

        if (epoch % self.log_policy_per_epochs) == 0:
            for prefix, path in zip(prefixes, paths):
                writer = iio.get_writer(os.path.join(logger.get_snapshot_dir(), prefix+'_'+str(epoch)+'.mp4'),
                                        fps=10)
                for p in path:
                    for step in range(len(p['terminals'])):
                        writer.append_data(p['observations'][step]['frontview_image'][::-1])
                writer.close()

        gt.stamp('logging', unique=False)
        logger.record_dict(_get_epoch_timings())
        logger.record_tabular('Epoch', epoch)
        logger.dump_tabular(file='agent.csv')

    def _log_model_stats(self, epoch):
        if (epoch % self.log_model_per_epochs) == 0:
            batch = next(iter(self.target_eval_loader))
            batch = buffer_to(batch, ptu.device)
            recons = self.dynamics_trainer.sample_data(batch, 'target')
            save_path = os.path.join(logger.get_snapshot_dir(), 'Forward_'+str(epoch)+'.png')
            save_image(recons, save_path, nrow=3)

        logger.log("Model epoch {} finished".format(epoch), with_timestamp=False)

        logger.record_dict(self.dynamics_trainer.get_diagnostics('model'))

        gt.stamp('logging', unique=False)
        logger.record_dict(_get_epoch_timings())
        logger.record_tabular('Epoch', epoch)
        logger.dump_tabular(file='model.csv')

    def _log_transfer_stats(self, epoch):
        if epoch % self.log_transfer_per_epochs == 0:
            source_batch = next(iter(self.source_eval_loader))
            target_batch = next(iter(self.target_eval_loader))
            source_imgs = buffer_to(source_batch, ptu.device)['image_obses'].permute(0, 3, 1, 2) / 255. - 0.5
            target_imgs = buffer_to(target_batch, ptu.device)['image_obses'].permute(0, 3, 1, 2) / 255. - 0.5
            images = self.transfer_trainer.sample(source_imgs, target_imgs)

            source_images = torch.cat([source_imgs[:10], images['source_recon'][:10],
                                       images['target_fake'][:10]], dim=0) + 0.5
            target_images = torch.cat([target_imgs[:10], images['target_recon'][:10],
                                       images['source_fake'][:10]], dim=0) + 0.5

            source_path = os.path.join(logger.get_snapshot_dir(), 'VAE_source_' + str(epoch) + '.png')
            target_path = os.path.join(logger.get_snapshot_dir(), 'VAE_target_' + str(epoch) + '.png')
            save_image(source_images, source_path, nrow=10)
            save_image(target_images, target_path, nrow=10)

        logger.log("Transfer epoch {} finished".format(epoch), with_timestamp=False)

        logger.record_dict(self.transfer_trainer.get_diagnostics('transfer'))

        gt.stamp('logging', unique=False)
        logger.record_dict(_get_epoch_timings())
        logger.record_tabular('Epoch', epoch)
        logger.dump_tabular(file='transfer.csv')

    def _preload_dataset(self, dataset, replay_buffer, func):
        for idx in range(len(dataset)):
            epi = dataset.get_episode(idx)
            epi = func(epi)
            replay_buffer.add_path(epi)
        return replay_buffer

    def _preprocess_episode_steps(self, episodes, env):

        data = {}
        for key in ['actions', 'rewards', 'terminals']:
            values = []
            for epi in episodes:
                values.append(epi[key][1:])
            values = np.concatenate(values, axis=0)
            data[key] = torchify_buffer(values)

        if env == 'source':
            visual_key = self.eval_source_env.visual_keys
        elif env == 'target':
            visual_key = self.eval_target_env.visual_keys
        else:
            raise NotImplementedError

        for key in ['robot_states', visual_key]:
            values, next_values = [], []
            for epi in episodes:
                if key == visual_key:
                    if env == 'source':
                        epi['image_obses'] = self.eval_source_env.process_images(epi[visual_key])
                    elif env == 'target':
                        epi['image_obses'] = self.eval_target_env.process_images(epi[visual_key])
                    else:
                        raise NotImplementedError
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

    def _preprocess_paired_steps(self, episodes):
        data = {}
        source_visual_key = self.eval_source_env.visual_keys
        target_visual_key = self.eval_target_env.visual_keys
        source_env_name = self.eval_source_env.robot_names[0]
        target_env_name = self.eval_target_env.robot_names[0]

        source_imgs, target_imgs = [], []
        for epi in episodes:
            source_imgs.append(self.eval_source_env.process_images(epi[source_env_name + '_' + source_visual_key][:-1]))
            target_imgs.append(self.eval_target_env.process_images(epi[target_env_name + '_' + target_visual_key][:-1]))

        data['source_image_obses'] = torchify_buffer(np.concatenate(source_imgs, axis=0))
        data['target_image_obses'] = torchify_buffer(np.concatenate(target_imgs, axis=0))

        return data

    def _preprocess_latent_steps(self, epi, env):
        data = {}
        num_steps = len(epi['terminals'])
        for key in ['actions', 'rewards', 'terminals']:
            values = []
            for step in range(1, num_steps):
                values.append(epi[key][step])
            data[key] = values

        if env == 'source':
            visual_key = self.eval_source_env.visual_keys
            images = self.eval_source_env.process_images(epi[visual_key])
        elif env == 'target':
            visual_key = self.eval_target_env.visual_keys
            images = self.eval_target_env.process_images(epi[visual_key])
        else:
            raise NotImplementedError

        images = buffer_to(images, ptu.device)
        images = images.permute(0, 3, 1, 2) / 255. - 0.5
        with torch.no_grad():
            outputs = self.transfer_model(images, env)
            latents = outputs['latents']
        latents = numpify_buffer(latents)

        obses = []
        for step in range(num_steps):
            obs = {
                'robot_states': epi['robot_states'][step],
                'object_states': latents[step]
            }
            obses.append(obs)
        next_obses = obses[1:]
        obses = obses[:-1]

        data['observations'] = obses
        data['next_observations'] = next_obses

        return data

    def to(self, device):
        for net in self.source_trainer.networks:
            net.to(device)
        for net in self.target_trainer.networks:
            net.to(device)
        self.transfer_model.to(device)

    def get_snapshot(self):
        snapshot = dict(
            source_agent=self.source_agent,
            target_agent=self.target_agent,
            transfer_model=self.transfer_model,
            source_trainer=self.source_trainer,
            target_trainer=self.target_trainer,
            transfer_trainer=self.transfer_trainer,
            dynamics_trainer=self.dynamics_trainer,

            agent_curr_epoch=self._agent_curr_epoch,
            model_curr_epoch=self._model_curr_epoch,
            transfer_curr_epoch=self._transfer_curr_epoch,
            torch_cpu_rng_state=torch.random.get_rng_state(),
            torch_cuda_rng_state=torch.cuda.get_rng_state_all(),
            random_rng_state=random.getstate(),
            numpy_rng_state=np.random.get_state()
        )

        return snapshot

    def load_snapshot(self):
        snapshot = logger.load_itr_params(0)
        torch_cpu_rng_state = snapshot['torch_cpu_rng_state'].cpu()
        torch_gpu_rng_state = []
        for state in snapshot['torch_cuda_rng_state']:
            torch_gpu_rng_state.append(state.cpu())

        torch.random.set_rng_state(torch_cpu_rng_state)
        torch.cuda.set_rng_state_all(iter(torch_gpu_rng_state))
        random.setstate(snapshot['random_rng_state'])
        np.random.set_state(snapshot['numpy_rng_state'])

        self._transfer_start_epoch = snapshot['transfer_curr_epoch']
        self._model_start_epoch = snapshot['model_curr_epoch']
        self._agent_start_epoch = snapshot['agent_curr_epoch']
        self._transfer_curr_epoch = snapshot['transfer_curr_epoch']
        self._model_curr_epoch = snapshot['model_curr_epoch']
        self._agent_curr_epoch = snapshot['agent_curr_epoch']

        self.eval_source_collector._policy.distribution = snapshot['target_agent']
        self.eval_target_collector._policy.distribution = snapshot['source_agent']

        self.source_agent = snapshot['source_agent']
        self.target_agent = snapshot['target_agent']
        self.transfer_model = snapshot['transfer_model']
        self.source_trainer = snapshot['source_trainer']
        self.target_trainer = snapshot['target_trainer']
        self.transfer_trainer = snapshot['transfer_trainer']
        self.dynamics_trainer = snapshot['dynamics_trainer']
        self.eval_target_collector._policy.vae_model = snapshot['transfer_model']
        self.eval_target_collector._policy.vae_model = snapshot['transfer_model']

    def load_dynamics_model(self, path):
        snapshot = torch.load(path, map_location='cuda')
        self.transfer_model.inverse_model = torch.nn.ModuleDict({
            'source': snapshot['source_inverse_model'],
            'target': snapshot['target_inverse_model']
        })
        self.transfer_model.forward_model = torch.nn.ModuleDict({
            'source': snapshot['source_forward_model'],
            'target': snapshot['target_forward_model']
        })


class OfflineInvVAESharingAlgorithm(object):

    def __init__(
            self,
            source_agent,
            target_agent,
            transfer_model,

            source_trainer,
            target_trainer,

            eval_source_env,
            eval_target_env,
            eval_source_collector,
            eval_target_collector,
            eval_max_path_length,

            source_general_dataset,
            target_general_dataset,
            source_train_dataset,
            target_train_dataset,
            source_supervise_dataset,
            target_supervise_dataset,
            source_train_buffer,
            target_train_buffer,
            policy_batch_size,

            num_agent_epochs,
            num_eval_steps_per_epoch,
            num_trains_per_policy_epoch,
            log_policy_per_epochs,

            action_relabel,
            lambda_penalty,
            supervise_ratio,
            unsupervise_ratio,

            save_snapshot,
            save_interval,
            save_model_path,
            **kwargs
    ):

        assert supervise_ratio + unsupervise_ratio <= 1.0
        self.origin_unsup_batch_size = int(unsupervise_ratio * policy_batch_size)
        self.origin_sup_batch_size = int(supervise_ratio * policy_batch_size)
        self.transfer_batch_size = policy_batch_size - self.origin_unsup_batch_size - \
                                   self.origin_sup_batch_size

        # models
        self.source_agent = source_agent
        self.target_agent = target_agent
        self.transfer_model = transfer_model

        # trainers
        self.source_trainer = source_trainer
        self.target_trainer = target_trainer

        # evaluation envs
        self.eval_source_env = eval_source_env
        self.eval_target_env = eval_target_env
        self.eval_source_collector = eval_source_collector
        self.eval_target_collector = eval_target_collector
        self.eval_max_path_length = eval_max_path_length

        self.source_train_dataset = source_train_dataset
        self.target_train_dataset = target_train_dataset
        self.source_general_dataset = source_general_dataset
        self.target_general_dataset = target_general_dataset
        self.source_supervise_dataset = source_supervise_dataset
        self.target_supervise_dataset = target_supervise_dataset
        self.source_train_buffer = source_train_buffer
        self.target_train_buffer = target_train_buffer

        self.log_policy_per_epochs = log_policy_per_epochs
        self.num_agent_epochs = num_agent_epochs
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_trains_per_policy_epoch = num_trains_per_policy_epoch

        # curr stats
        self._agent_start_epoch = 0
        self._agent_curr_epoch = 0

        # training kwargs
        self.lambda_penalty = lambda_penalty
        self.action_relabel = action_relabel

        # save snapshot
        self.save_snapshot = save_snapshot
        self.save_interval = save_interval
        self.save_model_path = save_model_path

    def train(self):
        self.transfer_model.eval_mode('all')
        self.source_trainer.eval()
        self.target_trainer.eval()

        self._preload_dataset(self.target_train_dataset, self.source_train_buffer,
                              lambda x: self._preprocess_transfer_episode(x, 'target'), 'transfer')
        self._preload_dataset(self.source_supervise_dataset, self.source_train_buffer,
                              lambda x: self._preprocess_origin_episode(x, False, 'source'), 'origin_sup')
        self._preload_dataset(self.source_train_dataset, self.target_train_buffer,
                              lambda x: self._preprocess_transfer_episode(x, 'source'), 'transfer')
        self._preload_dataset(self.target_supervise_dataset, self.target_train_buffer,
                              lambda x: self._preprocess_origin_episode(x, False, 'target'), 'origin_sup')
        self._preload_dataset(self.source_general_dataset, self.source_train_buffer,
                              lambda x: self._preprocess_origin_episode(x, True, 'source'), 'origin_unsup')
        self._preload_dataset(self.target_general_dataset, self.target_train_buffer,
                              lambda x: self._preprocess_origin_episode(x, True, 'target'), 'origin_unsup')

        # Agent training phase
        for epoch in gt.timed_for(
                range(self._agent_start_epoch, self.num_agent_epochs),
                save_itrs=True,
        ):
            self.eval_source_collector.collect_new_paths(
                self.eval_max_path_length,
                self.num_eval_steps_per_epoch,
                discard_incomplete_paths=True,
            )
            self.eval_target_collector.collect_new_paths(
                self.eval_max_path_length,
                self.num_eval_steps_per_epoch,
                discard_incomplete_paths=True,
            )
            gt.stamp('evaluation sampling', unique=False)

            for step in range(self.num_trains_per_policy_epoch):
                self.source_trainer.train()
                batch = self.source_train_buffer.random_batch(self.transfer_batch_size,
                                                              self.origin_sup_batch_size,
                                                              self.origin_unsup_batch_size)
                batch = buffer_to(batch, ptu.device)
                self.source_trainer.train_step(batch)
                self.source_trainer.eval()
                self.target_trainer.train()
                batch = self.target_train_buffer.random_batch(self.transfer_batch_size,
                                                              self.origin_sup_batch_size,
                                                              self.origin_unsup_batch_size)
                batch = buffer_to(batch, ptu.device)
                self.target_trainer.train_step(batch)
                self.target_trainer.eval()

            self._agent_curr_epoch += 1
            gt.stamp('training', unique=False)

            self._end_epoch(epoch, phase='agent')

    def _end_epoch(self, epoch, phase='agent'):
        if phase == 'agent':
            self._log_agent_stats(epoch)
        else:
            raise NotImplementedError

        self.eval_source_collector.end_epoch(epoch)
        self.eval_target_collector.end_epoch(epoch)
        self.source_trainer.end_epoch(epoch)
        self.target_trainer.end_epoch(epoch)

        if self.save_snapshot and epoch % self.save_interval == 0:
            snapshot = self.get_snapshot()
            logger.save_itr_params(epoch, snapshot, '.tmp')
            logger.atomic_replace(['params.pkl.tmp'])

    def _log_agent_stats(self, epoch):
        logger.log("Agent epoch {} finished".format(epoch), with_timestamp=False)

        logger.record_dict(self.source_trainer.get_diagnostics('agent'), prefix='source/')
        logger.record_dict(self.target_trainer.get_diagnostics('agent'), prefix='target/')

        logger.record_dict(self.eval_source_collector.get_diagnostics(), prefix='source_eval/')
        logger.record_dict(self.eval_target_collector.get_diagnostics(), prefix='target_eval/')

        source_eval_paths = self.eval_source_collector.get_epoch_paths()
        target_eval_paths = self.eval_target_collector.get_epoch_paths()

        if hasattr(self.eval_source_env, 'get_diagnostics'):
            logger.record_dict(self.eval_source_env.get_diagnostics(source_eval_paths), prefix='source_eval/')
        if hasattr(self.eval_target_env, 'get_diagnostics'):
            logger.record_dict(self.eval_target_env.get_diagnostics(target_eval_paths), prefix='target_eval/')

        if len(source_eval_paths) > 0:
            logger.record_dict(
                get_custom_generic_path_information(source_eval_paths, self.eval_max_path_length),
                prefix='source_eval/',
            )
        if len(target_eval_paths) > 0:
            logger.record_dict(
                get_custom_generic_path_information(target_eval_paths, self.eval_max_path_length),
                prefix='target_eval/',
            )

        """
        Misc
        """

        prefixes, paths = [], []

        if len(source_eval_paths) > 0:
            prefixes.append('source')
            paths.append(source_eval_paths)
        if len(target_eval_paths) > 0:
            prefixes.append('target')
            paths.append(target_eval_paths)

        if (epoch % self.log_policy_per_epochs) == 0:
            for prefix, path in zip(prefixes, paths):
                writer = iio.get_writer(os.path.join(logger.get_snapshot_dir(), prefix+'_'+str(epoch)+'.mp4'),
                                        fps=10)
                for p in path:
                    for step in range(len(p['terminals'])):
                        writer.append_data(p['observations'][step]['frontview_image'][::-1])
                writer.close()

        gt.stamp('logging', unique=False)
        logger.record_dict(_get_epoch_timings())
        logger.record_tabular('Epoch', epoch)
        logger.dump_tabular(file='agent.csv')

    def _preload_dataset(self, dataset, buffer, func, type):
        for idx in range(len(dataset)):
            epi = dataset.get_episode(idx)
            epi = func(epi)
            buffer.add_path(epi, type)
        return buffer

    def _preprocess_transfer_episode(self, epi, env='source'):
        data = {}
        num_steps = len(epi['terminals'])

        values = []
        for step in range(1, num_steps):
            values.append(epi['terminals'][step])
        data['terminals'] = values

        actions = epi['actions'][1:]
        rewards = epi['rewards'][1:]

        if env == 'source':
            visual_key = self.eval_source_env.visual_keys
            images = self.eval_source_env.process_images(epi[visual_key])
        elif env == 'target':
            visual_key = self.eval_target_env.visual_keys
            images = self.eval_target_env.process_images(epi[visual_key])
        else:
            raise NotImplementedError

        images = buffer_to(images, ptu.device)
        images = images.permute(0, 3, 1, 2) / 255. - 0.5
        with torch.no_grad():
            outputs = self.transfer_model(images, env)
            latents = outputs['latents']

            if env == 'source':
                images = self.transfer_model.decode(latents, 'target')[0]
                image_obs = images[:-1]
                next_image_obs = images[1:]
                pred_actions = self.transfer_model.inverse_model['target'](image_obs, next_image_obs)

                uncertainty = torch.mean((pred_actions - buffer_to(actions, ptu.device)) ** 2, dim=-1, keepdim=True)
                uncertainty = numpify_buffer(uncertainty)
                rewards = rewards - self.lambda_penalty * uncertainty

                if self.action_relabel:
                    actions = numpify_buffer(pred_actions)
            elif env == 'target':
                images = self.transfer_model.decode(latents, 'source')[0]
                image_obs = images[:-1]
                next_image_obs = images[1:]
                pred_actions = self.transfer_model.inverse_model['source'](image_obs, next_image_obs)

                uncertainty = torch.mean((pred_actions - buffer_to(actions, ptu.device)) ** 2, dim=-1, keepdim=True)
                uncertainty = numpify_buffer(uncertainty)
                rewards = rewards - self.lambda_penalty * uncertainty

                if self.action_relabel:
                    actions = numpify_buffer(pred_actions)
            else:
                raise NotImplementedError

        for token, values in zip(['actions', 'rewards'], [actions, rewards]):
            value = []
            for step in range(num_steps-1):
                value.append(values[step])
            data[token] = value

        latents = numpify_buffer(latents)
        obses = []
        for step in range(num_steps):
            obs = {
                'robot_states': epi['robot_states'][step],
                'object_states': latents[step]
            }
            obses.append(obs)
        next_obses = obses[1:]
        obses = obses[:-1]

        data['observations'] = obses
        data['next_observations'] = next_obses

        return data

    def _preprocess_origin_episode(self, epi, unsup, env='source'):
        data = {}
        num_steps = len(epi['terminals'])

        for key in ['actions', 'rewards', 'terminals']:
            values = []
            for step in range(1, num_steps):
                values.append(epi[key][step])
            data[key] = values

        if env == 'source':
            visual_key = self.eval_source_env.visual_keys
            images = self.eval_source_env.process_images(epi[visual_key])
        elif env == 'target':
            visual_key = self.eval_target_env.visual_keys
            images = self.eval_target_env.process_images(epi[visual_key])
        else:
            raise NotImplementedError

        images = buffer_to(images, ptu.device)
        images = images.permute(0, 3, 1, 2) / 255. - 0.5
        with torch.no_grad():
            outputs = self.transfer_model(images, env)
            latents = outputs['latents']
        latents = numpify_buffer(latents)
        obses = []
        for step in range(num_steps):
            obs = {
                'robot_states': epi['robot_states'][step],
                'object_states': latents[step]
            }
            obses.append(obs)
        next_obses = obses[1:]

        data['observations'] = obses
        data['next_observations'] = next_obses

        if unsup:
            for i in range(num_steps-1):
                data['rewards'][i] = data['rewards'][i] * 0.
        return data

    def to(self, device):
        for net in self.source_trainer.networks:
            net.to(device)
        for net in self.target_trainer.networks:
            net.to(device)
        self.transfer_model.to(device)

    def get_snapshot(self):
        snapshot = dict(
            source_agent=self.source_agent,
            target_agent=self.target_agent,
            source_trainer=self.source_trainer,
            target_trainer=self.target_trainer,
            agent_curr_epoch=self._agent_curr_epoch,
            torch_cpu_rng_state=torch.random.get_rng_state(),
            torch_cuda_rng_state=torch.cuda.get_rng_state_all(),
            random_rng_state=random.getstate(),
            numpy_rng_state=np.random.get_state()
        )

        return snapshot

    def load_snapshot(self):
        snapshot = logger.load_itr_params(0)
        torch_cpu_rng_state = snapshot['torch_cpu_rng_state'].cpu()
        torch_gpu_rng_state = []
        for state in snapshot['torch_cuda_rng_state']:
            torch_gpu_rng_state.append(state.cpu())

        torch.random.set_rng_state(torch_cpu_rng_state)
        torch.cuda.set_rng_state_all(iter(torch_gpu_rng_state))
        random.setstate(snapshot['random_rng_state'])
        np.random.set_state(snapshot['numpy_rng_state'])

        self._agent_start_epoch = snapshot['agent_curr_epoch']
        self._agent_curr_epoch = snapshot['agent_curr_epoch']

        self.eval_source_collector._policy.distribution = snapshot['source_agent']
        self.eval_target_collector._policy.distribution = snapshot['target_agent']

        self.source_agent = snapshot['source_agent']
        self.target_agent = snapshot['target_agent']
        self.source_trainer = snapshot['source_trainer']
        self.target_trainer = snapshot['target_trainer']

    def load_transfer_model(self, path):
        snapshot = torch.load(path, map_location='cuda')
        self.transfer_model = snapshot['transfer_model']
        self.eval_target_collector._policy.vae_model = snapshot['transfer_model']
        self.eval_source_collector._policy.vae_model = snapshot['transfer_model']


def get_custom_generic_path_information(paths, path_length, stat_prefix=''):
    statistics = OrderedDict()

    # Grab returns accumulated up to specified timestep
    for coef in [1.0]:
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