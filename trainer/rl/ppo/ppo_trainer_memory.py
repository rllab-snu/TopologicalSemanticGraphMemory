#!/usr/bin/env python3

# Copyright (without_goal+curr_emb) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from NuriUtils.myLRScheduler import GradualWarmupScheduler
from types import SimpleNamespace

from habitat import Config, logger
from habitat_baselines.common.base_trainer import BaseRLTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from env_utils.make_env_utils import construct_envs
from habitat_baselines.common.rollout_storage import RolloutStorage
from habitat_baselines.common.utils import (
    batch_obs,
    linear_decay,
)
from trainer.rl.ppo.ppo import PPO
import pickle, time
from env_utils import *
from model.policy import *


@baseline_registry.register_trainer(name="custom_ppo_memory")
class PPOTrainer_Memory(BaseRLTrainer):
    r"""Trainer class for PPO algorithm
    Paper: https://arxiv.org/abs/1707.06347.
    """
    supported_tasks = ["Nav-v0"]

    def __init__(self, config=None):
        self.is_wandb = config.ARGS['wandb']
        self.version = config.ARGS['version']
        super().__init__(config)
        self.actor_critic = None
        self.agent = None
        self.envs = None
        if config is not None:
            logger.info(f"config: {config}")

        self._static_encoder = False
        self._encoder = None

        self.last_observations = None
        self.last_recurrent_hidden_states = None
        self.last_prev_actions = None
        self.last_masks = None
        self.mode = self.config.RUN_TYPE
        self.args = SimpleNamespace(**self.config['ARGS'])

    def _setup_actor_critic_agent(self, ppo_cfg: Config) -> None:
        r"""Sets up actor critic and agent for PPO.

        Args:
            ppo_cfg: config node with relevant params

        Returns:
            None
        """
        logger.add_filehandler(self.config.LOG_FILE)

        self.actor_critic = eval(self.config.POLICY)(
            observation_space=self.envs.observation_spaces[0],
            action_space=self.envs.action_spaces[0],
            hidden_size=ppo_cfg.hidden_size,
            rnn_type=ppo_cfg.rnn_type,
            num_recurrent_layers=ppo_cfg.num_recurrent_layers,
            backbone=ppo_cfg.backbone,
            goal_sensor_uuid=self.config.TASK_CONFIG.TASK.GOAL_SENSOR_UUID,
            normalize_visual_inputs="panoramic_rgb" in self.envs.observation_spaces[0].spaces,
            cfg = self.config
        )

        self.actor_critic.to(self.device)
        if self.args.resume == 'none':
            if ppo_cfg.pretrained_encoder or ppo_cfg.rl_pretrained or ppo_cfg.il_pretrained:
                if torch.__version__ < '1.7.0' and getattr(ppo_cfg, 'pretrained_step', False):
                    pretrained_state = {'state_dict': pickle.load(open(ppo_cfg.pretrained_weights, 'rb')),
                                        'extra_state': {'step': ppo_cfg.pretrained_step}}
                else:
                    pretrained_state = torch.load(ppo_cfg.pretrained_weights, map_location="cpu")
                print('loaded ', ppo_cfg.pretrained_weights)

            if ppo_cfg.rl_pretrained:
                try:
                    self.actor_critic.load_state_dict(
                        {
                            k[len("actor_critic.") :]: v
                            for k, v in pretrained_state["state_dict"].items()
                        }
                    )
                except:
                    initial_state_dict = self.actor_critic.state_dict()
                    initial_state_dict.update({
                        k[len("actor_critic."):]: v
                        for k, v in pretrained_state['state_dict'].items()
                        if k[len("actor_critic."):] in initial_state_dict and
                           v.shape == initial_state_dict[k[len("actor_critic."):]].shape
                    })
                    print({
                        k[len("actor_critic."):]: v.shape
                        for k, v in pretrained_state['state_dict'].items()
                        if k[len("actor_critic."):] in initial_state_dict and
                           v.shape != initial_state_dict[k[len("actor_critic."):]].shape
                    })
                    self.actor_critic.load_state_dict(initial_state_dict)

                    print('############### loaded state dict selectively')
                try:
                    self.resume_steps = pretrained_state['extra_state']['step']
                except:
                    pass
            elif ppo_cfg.pretrained_encoder:
                try:
                    prefix = "actor_critic.net.img_encoder."
                    self.actor_critic.net.img_encoder.load_state_dict(
                        {
                            k[len(prefix) :]: v
                            for k, v in pretrained_state["state_dict"].items()
                            if k.startswith(prefix)
                        }
                    )
                    print('############### loaded pretrained visual encoder only')
                except:
                    prefix = "img_encoder."
                    initial_state_dict = self.actor_critic.net.img_encoder.state_dict()
                    initial_state_dict.update({
                            k[len(prefix) :]: v
                            for k, v in pretrained_state.items()
                            if k.startswith(prefix)
                        })
                    self.actor_critic.net.img_encoder.load_state_dict(initial_state_dict)
                    print('###############loaded pretrained visual encoder ',ppo_cfg.pretrained_weights)
            elif ppo_cfg.il_pretrained:
                pretrained_state = pretrained_state['state_dict']
                try:
                    mk, uk = self.actor_critic.load_state_dict(pretrained_state, strict=False)
                    if len(mk) > 0:
                        print("Missing", mk)
                    if len(uk) > 0:
                        print("Unexpected", uk)
                except:
                    # raise
                    initial_state_dict = self.actor_critic.state_dict()
                    initial_state_dict.update({
                        k: v
                        for k, v in pretrained_state.items()
                        if k in initial_state_dict and
                           v.shape == initial_state_dict[k].shape
                    })
                    self.actor_critic.load_state_dict(initial_state_dict)
                    print('selectively loaded il pretrained checkpoint')
                self.resume_steps = 0
                print('il pretrained checkpoint loaded')

        if self.args.resume != 'none':
            resume_state = torch.load(self.args.resume, map_location="cpu")
            try:
                self.actor_critic.load_state_dict(
                    {
                        k[len("actor_critic."):]: v
                        for k, v in resume_state["state_dict"].items()
                    }
                )
                self.resume_steps = resume_state['extra_state']['step']
                if "pth_time" in resume_state['extra_state']:
                    self.pth_time = resume_state['extra_state']['pth_time']
                if "env_time" in resume_state['extra_state']:
                    self.env_time = resume_state['extra_state']['env_time']
                if "update" in resume_state['extra_state']:
                    self.num_updates_done = resume_state['extra_state']['update']
                self.count_checkpoints = int(self.args.resume.split("/")[-1].split(".")[1])
            except:
                self.actor_critic.load_state_dict(
                    {
                        k[len("actor_critic."):]: v
                        for k, v in resume_state.items()
                    }
                )
                self.resume_steps = 0 #resume_state['extra_state']['step']
                self.pth_time = 0 #resume_state['extra_state']['pth_time']
                self.env_time = 0 #resume_state['extra_state']['env_time']
                self.num_updates_done = 0 #resume_state['extra_state']['update']
                self.count_checkpoints = 0 #int(self.args.resume.split("/")[-1].split(".")[1])

        if not ppo_cfg.train_encoder:
            self._static_encoder = True
            for param in self.actor_critic.net.img_encoder.parameters():
                param.requires_grad_(False)

        if ppo_cfg.reset_critic:
            nn.init.orthogonal_(self.actor_critic.critic.fc.weight)
            nn.init.constant_(self.actor_critic.critic.fc.bias, 0)

        self.agent = PPO(
            args=self.args,
            actor_critic=self.actor_critic,
            clip_param=ppo_cfg.clip_param,
            ppo_epoch=ppo_cfg.ppo_epoch,
            num_mini_batch=ppo_cfg.num_mini_batch,
            value_loss_coef=ppo_cfg.value_loss_coef,
            entropy_coef=ppo_cfg.entropy_coef,
            lr=ppo_cfg.lr,
            eps=ppo_cfg.eps,
            max_grad_norm=ppo_cfg.max_grad_norm,
            use_normalized_advantage=ppo_cfg.use_normalized_advantage,
        )

    def save_checkpoint(
        self, file_name: str, extra_state: Optional[Dict] = None
    ) -> None:
        checkpoint = {
            "state_dict": self.agent.state_dict(),
            "config": self.config,
        }
        if extra_state is not None:
            checkpoint["extra_state"] = extra_state
        torch.save(
            checkpoint, os.path.join(self.config.CHECKPOINT_FOLDER, file_name)
        )
        curr_checkpoint_list = [os.path.join(self.config.CHECKPOINT_FOLDER,x)
                                for x in os.listdir(self.config.CHECKPOINT_FOLDER)
                                if 'ckpt' in x]
        if len(curr_checkpoint_list) >= 25 :
            oldest_file = min(curr_checkpoint_list, key=os.path.getctime)
            os.remove(oldest_file)

    def load_checkpoint(self, checkpoint_path: str, *args, **kwargs) -> Dict:
        r"""Load checkpoint of specified path as a dict.

        Args:
            checkpoint_path: path of target checkpoint
            *args: additional positional args
            **kwargs: additional keyword args

        Returns:
            dict containing checkpoint info
        """
        return torch.load(checkpoint_path, *args, **kwargs)

    METRICS_BLACKLIST = {"top_down_map", "collisions.is_collision", 'episode', 'step', 'goal_index.num_goals', 'goal_index.curr_goal_index', 'gt_pose', 'ortho_map', 'ortho_map.agent_rot', 'ortho_map.agent_loc'}

    @classmethod
    def _extract_scalars_from_info(
        cls, info: Dict[str, Any]
    ) -> Dict[str, float]:
        result = {}
        for k, v in info.items():
            if k in cls.METRICS_BLACKLIST:
                continue

            if isinstance(v, dict):
                result.update(
                    {
                        k + "." + subk: subv
                        for subk, subv in cls._extract_scalars_from_info(
                            v
                        ).items()
                        if (k + "." + subk) not in cls.METRICS_BLACKLIST
                    }
                )
            # Things that are scalar-like will have an np.size of 1.
            # Strings also have an np.size of 1, so explicitly ban those
            elif np.size(v) == 1 and not isinstance(v, str) and not isinstance(v, list):
                result[k] = float(v)
            elif len(v) == 1:
                result[k] = float(v[0])
            elif isinstance(v, list):
                result[k] = float(np.array(v).mean())
            elif isinstance(v, str) or isinstance(v, tuple):
                result[k] = v
            else:
                result[k] = float(v.mean())


        return result

    @classmethod
    def _extract_scalars_from_infos(
        cls, infos: List[Dict[str, Any]]
    ) -> Dict[str, List[float]]:

        results = defaultdict(list)
        for i in range(len(infos)):
            for k, v in cls._extract_scalars_from_info(infos[i]).items():
                results[k].append(v)

        return results

    def _collect_rollout_step(
        self, rollouts, current_episode_reward, running_episode_stats
    ):
        pth_time = 0.0
        env_time = 0.0

        t_sample_action = time.time()
        # sample actions
        with torch.no_grad():

            (
                values,
                actions,
                actions_log_probs,
                recurrent_hidden_states,
                _,
                preds,
                _
            ) = self.actor_critic.act(
                self.last_observations,
                self.last_recurrent_hidden_states,
                self.last_prev_actions,
                self.last_masks
            )

        pth_time += time.time() - t_sample_action

        t_step_env = time.time()
        if preds is not None:
            pred1, pred2 = preds
            have_been = torch.sigmoid(pred1[:,:]).detach().cpu().numpy().tolist() if pred1 is not None else None
            pred_target_distance = torch.sigmoid(pred2[:,:]).detach().cpu().numpy().tolist() if pred2 is not None else None

            log_strs = []
            for i in range(len(actions)):
                hb = have_been[i] if have_been is not None else None
                ptd = pred_target_distance[i] if pred_target_distance is not None else None
                log_str = ''
                if hb is not None:
                    log_str += 'have_been: ' + ' '.join(['%.3f'%hb_ag for hb_ag in hb]) + ' '
                if ptd is not None:
                    try:
                        log_str += 'pred_progress: ' + ' '.join([('%.3f'*self.num_goals)%tuple(pd_ag) for pd_ag in ptd]) + ' '
                    except:
                        log_str += 'pred_progress: ' + ' '.join(['%.3f'%hb_ag for hb_ag in ptd]) + ' '
                log_strs.append(log_str)
            self.envs.call(['log_info']*self.num_processes,[{'log_type':'str', 'info':log_strs[i]} for i in range(self.num_processes)])

        acts = actions.detach().cpu().numpy().tolist()
        batch, rewards, dones, infos = self.envs.step(acts)
        # if self.config.ARGS['render']:
        #     self.envs.render('human')
        env_time += time.time() - t_step_env

        t_update_stats = time.time()

        rewards = torch.tensor(
            rewards, dtype=torch.float, device=current_episode_reward.device
        )

        masks = torch.tensor(
            [[0.0] if done else [1.0] for done in dones],
            dtype=torch.float,
            device=current_episode_reward.device,
        )
        if current_episode_reward.shape != rewards:
            rewards = rewards.unsqueeze(-1)
        current_episode_reward += rewards
        running_episode_stats["reward"] += (1 - masks) * current_episode_reward
        running_episode_stats["count"] += 1 - masks
        for k, v in self._extract_scalars_from_infos(infos).items():
            if k == 'scene':
                running_episode_stats[k] = v
                continue
            v = torch.tensor(
                v, dtype=torch.float, device=current_episode_reward.device
            ).unsqueeze(1)
            if k not in running_episode_stats:
                running_episode_stats[k] = torch.zeros_like(
                    running_episode_stats["count"]
                )
            running_episode_stats[k] += (1 - masks) * v

        current_episode_reward *= masks
        rollouts.insert(
            {k: v[:self.num_processes] for k,v in batch.items()},
            recurrent_hidden_states[:,:self.num_processes],
            actions[:self.num_processes],
            actions_log_probs[:self.num_processes],
            values[:self.num_processes],
            rewards[:self.num_processes],
            masks[:self.num_processes],
        )
        self.last_observations = batch
        self.last_recurrent_hidden_states = recurrent_hidden_states
        self.last_prev_actions = actions.unsqueeze(-1)
        self.last_masks = masks.to(self.device)
        pth_time += time.time() - t_update_stats
        return pth_time, env_time, self.num_processes

    def _update_agent(self, ppo_cfg, rollouts):
        t_update_model = time.time()
        with torch.no_grad():
            last_observation = {
                k: v[rollouts.step].to(self.device) for k, v in rollouts.observations.items()
            }
            next_value = self.actor_critic.get_value(
                last_observation,
                rollouts.recurrent_hidden_states[rollouts.step],
                rollouts.prev_actions[rollouts.step],
                rollouts.masks[rollouts.step],
            ).detach()

        rollouts.compute_returns(
            next_value, ppo_cfg.use_gae, ppo_cfg.gamma, ppo_cfg.tau
        )

        value_loss, action_loss, dist_entropy, aux_losses = self.agent.update(rollouts)

        rollouts.after_update()

        return (
            time.time() - t_update_model,
            value_loss,
            action_loss,
            dist_entropy,
            aux_losses
        )

    def train(self) -> None:
        if torch.cuda.device_count() <= 1:
            self.config.defrost()
            self.config.TORCH_GPU_ID = 0
            self.config.SIMULATOR_GPU_ID = 0
            self.config.freeze()
        self.envs = construct_envs(
            self.config, eval(self.config.ENV_NAME), fix_on_cpu=getattr(self.config,'FIX_ON_CPU',False)
        )
        self.num_agents = self.config.NUM_AGENTS
        self.num_goals = self.config.NUM_GOALS

        ppo_cfg = self.config.RL.PPO
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.sim_device = (
            torch.device("cuda", self.config.SIMULATOR_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        if not os.path.isdir(self.config.CHECKPOINT_FOLDER):
            os.makedirs(self.config.CHECKPOINT_FOLDER)

        if self.is_wandb:
            import wandb
            self.wandb_run = wandb.init(project="ObjectNavigation", config=self.config, name=self.mode + '_' + self.config.VERSION, tags=self.config.TAGS)

        self._setup_actor_critic_agent(ppo_cfg)

        logger.info(
            "agent number of parameters: {}".format(
                sum(param.numel() for param in self.agent.parameters())
            )
        )

        num_train_processes, num_val_processes = self.config.NUM_PROCESSES, self.config.NUM_VAL_PROCESSES
        total_processes = num_train_processes + num_val_processes
        memory_list = ['img_memory_feat', 'img_memory_relpose', 'img_memory_mask', 'img_memory_A', 'img_memory_time'] +\
                      ['obj_memory_feat', 'obj_memory_category', 'obj_memory_mask', 'obj_memory_A_OV', 'obj_memory_time']
        targ_object_list = ['target_loc_object', 'target_loc_object_category', 'target_loc_object_mask', 'target_loc_object_score']
        curr_object_list = ['object', 'object_category', 'object_mask', 'object_score']
        aux_list = ['have_been', 'have_seen', 'is_img_target', 'is_obj_target']
        OBS_LIST = self.config.OBS_TO_SAVE + memory_list + targ_object_list + curr_object_list + aux_list + ['img_memory_idx', 'target_goal']
        self.num_processes = num_train_processes
        rollouts = RolloutStorage(
            ppo_cfg.num_steps,
            num_train_processes,
            self.envs.observation_spaces[0],
            self.envs.action_spaces[0],
            ppo_cfg.hidden_size,
            self.actor_critic.net.num_recurrent_layers,
            OBS_LIST=OBS_LIST,
            MEMORY_LIST=memory_list,
            device=self.device
        )
        rollouts.to(self.device)
        batch = self.envs.reset()

        if isinstance(batch, list):
            batch = batch_obs(batch, device=self.device)
        for sensor in rollouts.observations:
            try:
                num_vis_node = int(batch['img_memory_mask'].sum(1).max().item())
                num_obj_node = int(batch['obj_memory_mask'].sum(1).max().item())
                if sensor in memory_list:
                    aa = batch[sensor][:num_train_processes]
                    if "img_memory" in sensor:
                        rollouts.observations[sensor][0][:, :num_vis_node].copy_(aa[:, :num_vis_node])
                        if "A" in sensor:
                            rollouts.observations[sensor][0][:, :num_vis_node, :num_vis_node].copy_(aa[:, :num_vis_node, :num_vis_node])
                    elif "obj_memory" in sensor:
                        rollouts.observations[sensor][0][:, :num_obj_node].copy_(aa[:, :num_obj_node])
                        if "A" in sensor:
                            rollouts.observations[sensor][0][:, :num_obj_node, :num_vis_node].copy_(aa[:, :num_obj_node, :num_vis_node])
                else:
                    rollouts.observations[sensor][0].copy_(batch[sensor][:num_train_processes])
            except:
                print('error on copying observation : ', sensor, 'expected_size:', rollouts.observations[sensor][0].shape, 'actual_size:',batch[sensor][:num_train_processes].shape)
                raise

        self.last_observations = batch
        self.last_recurrent_hidden_states = torch.zeros(self.actor_critic.net.num_recurrent_layers, total_processes, ppo_cfg.hidden_size).to(self.device)
        self.last_prev_actions = torch.zeros(total_processes, rollouts.prev_actions.shape[-1]).to(self.device)
        self.last_masks = torch.zeros(total_processes, 1).to(self.device)

        # batch and observations may contain shared PyTorch CUDA
        # tensors.  We must explicitly clear them here otherwise
        # they will be kept in memory for the entire duration of training!
        batch = None
        observations = None

        current_episode_reward = torch.zeros(self.envs.num_envs, 1)
        running_episode_stats = dict(
            count=torch.zeros(self.envs.num_envs, 1),
            reward=torch.zeros(self.envs.num_envs, 1),
        )
        window_episode_stats = defaultdict(
            lambda: deque(maxlen=ppo_cfg.reward_window_size)
        )

        t_start = time.time()
        env_time = 0 if not hasattr(self, 'env_time') else self.env_time
        pth_time = 0 if not hasattr(self, 'pth_time') else self.pth_time
        start_steps = 0 if not hasattr(self, 'resume_steps') else self.resume_steps
        self.num_steps_done = 0 if not hasattr(self, 'resume_steps') else self.resume_steps
        count_checkpoints = 0 if not hasattr(self, 'count_checkpoints') else self.count_checkpoints
        self.num_updates_done = 0 if not hasattr(self, 'num_updates_done') else self.num_updates_done

        lr_scheduler = LambdaLR(
            optimizer=self.agent.optimizer,
            lr_lambda=lambda x: 1 - self.percent_done(),
        )
        lr_scheduler = GradualWarmupScheduler(self.agent.optimizer, multiplier=1, total_epoch=200, after_scheduler=lr_scheduler)

        while not self.is_done():
            if ppo_cfg.use_linear_clip_decay:
                self.agent.clip_param = ppo_cfg.clip_param * (
                    1 - self.percent_done()
                )

            for step in range(ppo_cfg.num_steps):
                (
                    delta_pth_time,
                    delta_env_time,
                    delta_steps,
                ) = self._collect_rollout_step(
                    rollouts, current_episode_reward, running_episode_stats
                )
                pth_time += delta_pth_time
                env_time += delta_env_time
                self.num_steps_done += delta_steps

            (
                delta_pth_time,
                value_loss,
                action_loss,
                dist_entropy,
                aux_losses
            ) = self._update_agent(ppo_cfg, rollouts)
            self.num_updates_done += 1
            # self.envs.call(['set_update']*self.num_processes,[{'num_update':self.num_updates_done} for i in range(self.num_processes)])

            pth_time += delta_pth_time
            rollouts.after_update()

            for k, v in running_episode_stats.items():
                if k == 'scene': continue
                window_episode_stats[k].append(v.clone())

            if ppo_cfg.use_linear_lr_decay:
                lr_scheduler.step()

            deltas = {
                k: (
                    (v[-1][:self.num_processes] - v[0][:self.num_processes]).sum().item()
                    if len(v) > 1
                    else v[0][:self.num_processes].sum().item()
                )
                for k, v in window_episode_stats.items()
            }

            deltas["count"] = max(deltas["count"], 1.0)
            if self.is_wandb:
                self.wandb_run.log(
                    {
                        'value_loss': value_loss,
                        "action_loss": action_loss,
                        "havebeen_loss": aux_losses[0],
                        "progress_loss": aux_losses[1],
                        "have_seen_loss": aux_losses[2],
                        "is_target_loss": aux_losses[3],
                        # "gt_action_loss": gt_act_loss,
                        "dist_entropy": dist_entropy,
                        "success rate": deltas["success"] / deltas["count"],
                        "SPL": deltas["spl"] / deltas["count"],
                        "distance_to_goal": deltas["distance_to_goal"] / deltas["count"],
                        "start_to_goal": deltas["start_to_goal"] / deltas["count"],
                        "reward": deltas["reward"] / deltas["count"],
                        "update_step": self.num_updates_done,
                        "fps": (self.num_steps_done - start_steps) / (time.time() - t_start),
                        "collisions": deltas["collisions.count"] / deltas["count"],
                        # "episode_id": self.envs.episode_id()[0],
                    },
                    step=self.num_updates_done
                )

            eval_deltas = {
                k: (
                    (v[-1][self.num_processes:] - v[0][self.num_processes:]).sum().item()
                    if len(v) > 1
                    else v[0][self.num_processes:].sum().item()
                )
                for k, v in window_episode_stats.items()
            }
            eval_deltas["count"] = max(eval_deltas["count"], 1.0)

            # log stats
            if self.num_updates_done > 0 and self.num_updates_done % self.config.RL.LOG_INTERVAL == 0:
                logger.info(
                    "update: {}\tfps: {:.3f}\t".format(
                        self.num_updates_done, (self.num_steps_done - start_steps) / (time.time() - t_start)
                    )
                )

                logger.info(
                    "update: {}\tenv-time: {:.3f}s\tpth-time: {:.3f}s\tlr: {:.7f}\t"
                    "steps: {}/{}({:.2}%)".format(
                        self.num_updates_done, env_time, pth_time,  lr_scheduler.get_lr()[0], self.num_steps_done, int(self.config.TOTAL_NUM_STEPS), self.percent_done()*100.
                    )
                )

                logger.info(
                    "Average window size: {}  {}".format(
                        len(window_episode_stats["count"]),
                        "  ".join(
                            "{}: {:.3f}".format(k, v / deltas["count"])
                            for k, v in deltas.items()
                            if k != "count"
                        ),
                    )
                )
                if num_val_processes > 0:
                    logger.info(
                        "validation metrics: {}".format(
                            "  ".join(
                                "{}: {:.3f}".format(k, v / eval_deltas["count"])
                                for k, v in eval_deltas.items()
                                if k != "count"
                            ),
                        )
                    )

            # checkpoint model
            if self.num_updates_done % self.config.RL.CHECKPOINT_INTERVAL == 0:
                self.save_checkpoint(
                    "%s/ckpt.%05d.pth"%(self.version, count_checkpoints), dict(step=self.num_steps_done, env_time=env_time, pth_time=pth_time, num_updates_done=self.num_updates_done)
                )
                count_checkpoints += 1

        self.envs.close()
        if self.is_wandb:
            self.wandb_run.finish()
