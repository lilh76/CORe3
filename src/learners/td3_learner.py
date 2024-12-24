import copy
import torch as th
import torch.nn.functional as F
from torch.optim import RMSprop, Adam
from copy import deepcopy
import numpy as np
import os

from utils.debug import *
from modules.critics import REGISTRY as critic_registry

class TD3Learner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_actions = args.n_actions
        self.actor_freq = args.actor_freq
        self.logger = logger

        self.mac = mac
        self.target_mac = copy.deepcopy(self.mac)
        self.agent_params = list(mac.parameters())

        self.critic1 = critic_registry[args.critic_type](args)
        self.target_critic1 = copy.deepcopy(self.critic1)
        self.critic2 = critic_registry[args.critic_type](args)
        self.target_critic2 = copy.deepcopy(self.critic2)

        self.critic_params = list(self.critic1.parameters()) + list(self.critic2.parameters())

         
        self.policy_noise = args.policy_noise
        self.noise_clip = args.noise_clip
        self.max_action = th.tensor(args.max_action, device=args.device)

        if self.args.optim_type.lower() == "rmsprop":
            self.agent_optimiser = RMSprop(params=self.agent_params, lr=self.args.lr, alpha=self.args.optim_alpha, eps=self.args.optim_eps, weight_decay=self.args.weight_decay)
        elif self.args.optim_type.lower() == "adam":
            self.agent_optimiser = Adam(params=self.agent_params, lr=self.args.lr)
        else:
            raise ValueError("Invalid optimiser type", self.args.optim_type)

        self.log_stats_t = -self.args.learner_log_interval - 1
        self.training_steps = 0
        self.last_target_update_step = 0
        self.last_target_update_episode = 0

        self.log_actor = {"actor_loss": [], "actor_grad_norm": []}

    def create_head(self, learning_obj):
        self.critic1.create_head(learning_obj)
        self.critic2.create_head(learning_obj)
        self._update_targets_hard()

    def set_optimiser(self):
        self.critic_params = list(self.critic1.parameters()) + list(self.critic2.parameters())
        if self.args.optim_type.lower() == "rmsprop":
            self.critic_optimiser = RMSprop(params=self.critic_params, lr=self.args.lr, alpha=self.args.optim_alpha, eps=self.args.optim_eps, weight_decay=self.args.weight_decay)
        elif self.args.optim_type.lower() == "adam":
            self.critic_optimiser = Adam(params=self.critic_params, lr=self.args.lr)

    def train(self, batch: dict, t_env: int, episode_num: int, model_learner):
        critic_log = self.train_critic(batch, model_learner)

        if (self.training_steps + 1) % self.actor_freq == 0:
            states = batch["states"]  
            preferences = batch["preferences"]  
            batch_size = states.shape[0]

            actions = self.mac.forward(states, preferences).view(batch_size, -1)
            Q = self.critic1(states, actions, preferences, self.args.seen_obj)
            wQ = th.bmm(preferences.unsqueeze(1), Q.unsqueeze(-1)).squeeze()
            actor_loss = -wQ.mean()

            self.agent_optimiser.zero_grad()
            actor_loss.backward()
            actor_grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, max_norm=100)
            self.agent_optimiser.step()

            if (self.args.target_update_interval_or_tau > 1 and (episode_num - self.last_target_update_episode) / self.args.target_update_interval_or_tau >= 1.0):
                self._update_targets_hard()
                self.last_target_update_episode = episode_num
            elif self.args.target_update_interval_or_tau <= 1.0:
                self._update_targets_soft(self.args.target_update_interval_or_tau)

            self.log_actor["actor_loss"].append(actor_loss.item())
            self.log_actor["actor_grad_norm"].append(actor_grad_norm.item())

        self.training_steps += 1
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            for k, v in critic_log.items():
                self.logger.log_stat(k, v, t_env)
            if len(self.log_actor["actor_loss"]) > 0:
                ts = len(self.log_actor["actor_loss"])
                for k, v in self.log_actor.items():
                    self.logger.log_stat(k, sum(v) / ts, t_env)
                    self.log_actor[k].clear()

            self.log_stats_t = t_env

    def train_critic(self, batch: dict, model_learner):
        critic_log = {}
        states       = batch['states']  
        actions      = batch['actions']  
        rewards      = batch['rewards']  
        rewards_real = batch['rewards_real']  
        preferences  = batch['preferences']  
        next_states  = batch['next_states']  
        dones        = batch['dones']  

        forward_obj, n_learning_obj_fake, preferences_update, rewards = \
            model_learner.preprocess(states, actions, next_states, dones, rewards, rewards_real, preferences)

        bs = actions.shape[0]
        current_Q1 = self.critic1(states, actions, preferences, forward_obj)   
        current_Q2 = self.critic2(states, actions, preferences, forward_obj)   
        with th.no_grad():
             
            noise = ((th.randn_like(th.tensor(actions)) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip).to(self.args.device))
            next_actions = self.target_mac.forward(next_states, preferences).view(bs, -1)   
            next_actions = (next_actions + noise).clip(-self.max_action, self.max_action)
            target_Q1 = self.target_critic1(next_states, next_actions, preferences, forward_obj)   
            target_Q2 = self.target_critic2(next_states, next_actions, preferences, forward_obj)   
            pb_target_vals1 = th.bmm(preferences_update.unsqueeze(1), target_Q1.unsqueeze(-1)).squeeze()
            pb_target_vals2 = th.bmm(preferences_update.unsqueeze(1), target_Q2.unsqueeze(-1)).squeeze()
            min_idx = th.argmin(th.cat([pb_target_vals1.unsqueeze(-1), pb_target_vals2.unsqueeze(-1)],dim=-1,),dim=-1)
            target_vals = th.zeros(bs, n_learning_obj_fake, device=self.args.device)
            assert bs == min_idx.shape[0]
            for i in range(bs):
                if min_idx[i] == 0:
                    target_vals[i, :] = target_Q1[i, :]
                else:
                    target_vals[i, :] = target_Q2[i, :]

            target_vals = target_vals.detach()

            mask = 1 - dones.repeat(1, n_learning_obj_fake)
            target_Q = rewards + mask * self.args.gamma * target_vals

        td_loss = F.smooth_l1_loss(current_Q1, target_Q) + F.smooth_l1_loss(current_Q2, target_Q)

        critic_loss = td_loss
        self.critic_optimiser.zero_grad()
        critic_loss.backward()
        critic_grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params, max_norm=100)
        self.critic_optimiser.step()

        critic_log["critic_loss"] = critic_loss.item()
        critic_log["critic_grad_norm"] = critic_grad_norm.item()
        td_error1 = current_Q1 - target_Q
        td_error2 = current_Q2 - target_Q
        critic_log["td_error1_abs"] = td_error1.abs().mean().item()
        critic_log["td_error2_abs"] = td_error2.abs().mean().item()
        critic_log["q_taken1_mean"] = current_Q1.mean().item()
        critic_log["q_taken2_mean"] = current_Q2.mean().item()
        critic_log["target_mean"] = target_Q.mean().item()
        return critic_log
    
    def _update_targets_hard(self):
        self.target_mac.load_state(self.mac)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        for obj, head in self.critic1.obj2head.items():
            self.target_critic1.obj2head[obj] = deepcopy(head)
        for obj, head in self.critic2.obj2head.items():
            self.target_critic2.obj2head[obj] = deepcopy(head)

    def _update_targets_soft(self, tau):
        for target_param, param in zip(self.target_mac.parameters(), self.mac.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        self.critic1.cuda()
        self.target_critic1.cuda()
        self.critic2.cuda()
        self.target_critic2.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.critic1.state_dict(), "{}/critic1.th".format(path))
        th.save(self.critic2.state_dict(), "{}/critic2.th".format(path))
        for obj, head in self.critic1.obj2head.items():
            th.save(head.state_dict(), f"{path}/critic1head_{obj}.th")
        for obj, head in self.critic2.obj2head.items():
            th.save(head.state_dict(), f"{path}/critic2head_{obj}.th")
        th.save(self.agent_optimiser.state_dict(), "{}/agent_opt.th".format(path))
        th.save(self.critic_optimiser.state_dict(), "{}/critic_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.critic1.load_state_dict(th.load("{}/critic1.th".format(path), map_location=lambda storage, loc: storage))
        self.critic2.load_state_dict(th.load("{}/critic2.th".format(path), map_location=lambda storage, loc: storage))

        for obj, head in self.critic1.obj2head.items():
            head.load_state_dict(th.load(f"{path}/critic1head_{obj}.th", map_location=lambda storage, loc: storage))
        for obj, head in self.critic2.obj2head.items():
            head.load_state_dict(th.load(f"{path}/critic2head_{obj}.th", map_location=lambda storage, loc: storage))
        self._update_targets_hard()

        self.agent_optimiser.load_state_dict(th.load("{}/agent_opt.th".format(path),map_location=lambda storage, loc: storage))
        self.critic_optimiser.load_state_dict(th.load("{}/critic_opt.th".format(path),map_location=lambda storage, loc: storage))
