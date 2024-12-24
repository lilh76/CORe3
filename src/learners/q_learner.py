import torch as th
from torch.optim import RMSprop, Adam
from copy import deepcopy
import numpy as np

from utils.debug import *

class QLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.target_mac = deepcopy(self.mac)
        self.log_stats_t = -self.args.learner_log_interval - 1
        self.training_steps = 0
        self.last_target_update_episode = 0

    def create_head(self, learning_obj):
        self.mac.create_head(learning_obj)
        self._update_targets_hard()

    def set_optimiser(self):
        self.params = self.mac.parameters()
        if self.args.optim_type.lower() == "rmsprop":
            self.optimiser = RMSprop(params=self.params, lr=self.args.lr, alpha=self.args.optim_alpha, eps=self.args.optim_eps, weight_decay=self.args.weight_decay)
        elif self.args.optim_type.lower() == "adam":
            self.optimiser = Adam(params=self.params, lr=self.args.lr, weight_decay=self.args.weight_decay)

    def train(self, batch, t_env: int, episode_num: int, model_learner):

        states       = batch['states']  
        actions      = batch['actions'].type(th.int64)  
        rewards      = batch['rewards']  
        rewards_real = batch['rewards_real']  
        preferences  = batch['preferences']  
        next_states  = batch['next_states']  
        dones        = batch['dones']  

        forward_obj, n_learning_obj_fake, preferences_update, rewards = \
            model_learner.preprocess(states, actions, next_states, dones, rewards, rewards_real, preferences)

        bs = states.shape[0]
        mac_out = self.mac.forward(states, preferences, forward_obj)  
        n_actions = mac_out.shape[-1]
        chosen_action_qvals = th.gather(mac_out, dim=2, index=actions.unsqueeze(1).repeat(1, n_learning_obj_fake, 1)).squeeze(-1)  
        target_mac_out = self.target_mac.forward(next_states, preferences, forward_obj)  
        preferences_update = preferences_update.unsqueeze(1).repeat(1, n_actions, 1).reshape(-1, 1, n_learning_obj_fake)
         
        target_mac_out_with_preference = target_mac_out.permute(0, 2, 1).reshape(-1, n_learning_obj_fake, 1)
         
        target_mac_out_with_preference = th.matmul(preferences_update, target_mac_out_with_preference).reshape(bs, n_actions)  
        best_actions = target_mac_out_with_preference.max(dim=1)[1]  
        best_actions = best_actions.unsqueeze(-1).unsqueeze(-1).repeat(1, n_learning_obj_fake, 1)  
        target_max_qvals = th.gather(target_mac_out, dim=2, index=best_actions).squeeze(-1)  
        mask = 1 - dones.repeat(1, n_learning_obj_fake)  
        targets = rewards + mask * self.args.gamma * target_max_qvals  
        loss = ((chosen_action_qvals - targets.detach()) ** 2).mean()
            
         
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()
         
         

        self.training_steps += 1
        if self.args.target_update_interval_or_tau > 1 and (episode_num - self.last_target_update_episode) / self.args.target_update_interval_or_tau >= 1.0:
            self._update_targets_hard()
            self.last_target_update_episode = episode_num
        elif self.args.target_update_interval_or_tau <= 1.0:
            self._update_targets_soft(self.args.target_update_interval_or_tau)

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)     
            self.logger.log_stat("q_taken_mean", chosen_action_qvals.mean().item(), t_env)
            self.logger.log_stat("target_mean", targets.mean().item(), t_env)
            self.log_stats_t = t_env
            
    def _update_targets_hard(self):
        self.target_mac.load_state(self.mac)

    def _update_targets_soft(self, tau):
        for target_param, param in zip(self.target_mac.parameters(), self.mac.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.target_mac.load_models(path)
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
