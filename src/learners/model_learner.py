import torch as th
from torch.optim import RMSprop, Adam
from copy import deepcopy
import numpy as np

from utils.debug import *
from modules.world_models import REGISTRY

class ModelLearner:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.world_model = REGISTRY[args.world_model_type](args)

    def create_head(self, learning_obj, phase_idx):
        if self.args.world_model_type == "mlp":
            self.world_model.create_head(learning_obj)
    
    def set_optimiser(self, phase_idx):
        if self.args.world_model_type == "mlp":
            self.params_model = self.world_model.parameters(phase_idx)
            if self.args.optim_type.lower() == "rmsprop":
                self.optimiser_model = RMSprop(params=self.params_model, lr=self.args.lr_model, alpha=self.args.optim_alpha, eps=self.args.optim_eps, weight_decay=self.args.weight_decay)
            elif self.args.optim_type.lower() == "adam":
                self.optimiser_model = Adam(params=self.params_model, lr=self.args.lr_model, weight_decay=self.args.weight_decay) 

    def train_world_model(self, batch, t_env: int, epoch):
        
        if self.args.world_model_type == "table":
            self.world_model.update(batch)
            self.logger.log_stat("dict_size_1", len(self.world_model.d), t_env)
            self.logger.log_stat("dict_size_2", sum([len(v) for k, v in self.world_model.d.items()]), t_env)

        elif self.args.world_model_type == "mlp":

            states      = batch['states']  

            if self.args.action_selector == 'continuous' : 
                actions     = batch['actions']
            else:
                actions     = batch['actions'].type(th.int64)  
                actions     = th.eye(self.args.n_actions, device=self.args.device)[actions.flatten()]

            rewards     = batch['rewards']  
            next_states = batch['next_states']  
             
            model_inputs = th.cat([states, actions], dim=-1)  
            mus_state, log_vars_state, mus_rewards, log_vars_rewards = self.world_model(model_inputs, forward_obj=self.args.learning_obj)  

            n_models, bs, _ = mus_rewards.shape

            mus = th.cat([mus_state, mus_rewards], dim=-1)
            log_vars = th.cat([log_vars_state, log_vars_rewards], dim=-1)
            inv_vars = th.exp(-log_vars).reshape(n_models * bs, -1)  
            delta = (mus - th.cat([next_states, rewards], dim=-1)).reshape(n_models * bs, -1)  

            if self.args.use_stochastic_world_model:
                l1 = (delta * inv_vars * delta).mean()
                l2 = log_vars.mean()
            else:
                l1 = (delta ** 2).mean()
                l2 = 0.
            loss = l1 + l2
            self.logger.log_stat("model_loss", loss.item(), t_env + epoch)
            self.optimiser_model.zero_grad()
            loss.backward()
            self.optimiser_model.step()    
        else:
            assert 0


    def preprocess(self, states, actions, next_states, dones, rewards, rewards_real, preferences):
         
        predict_obj = set(self.args.seen_obj) - set(self.args.learning_obj)  
        if self.args.use_world_model and len(predict_obj) > 0:
            if self.args.world_model_type == "table":
                model_inputs = th.cat([states, actions], dim=1)
            elif self.args.world_model_type == "mlp":
                if self.args.action_selector != 'continuous':
                    actions_onehot = th.eye(self.args.n_actions, device=self.args.device)[actions.flatten()]
                    model_inputs = th.cat([states, actions_onehot], dim=1)
                else:
                    model_inputs = th.cat([states, actions], dim=1)
            forward_obj_model = sorted(list(predict_obj), key=lambda x: self.args.seen_obj.index(x))  
            rewards_pred = self.world_model.predict_rewards(model_inputs, forward_obj=self.args.seen_obj)  
            pred_reward_idx = np.where(np.isin(np.array(self.args.seen_obj), np.array(forward_obj_model)))[0]  
            real_reward_idx = np.where(np.isin(np.array(self.args.seen_obj), np.array(self.args.learning_obj)))[0]  
            rewards_train = th.zeros(self.args.batch_size, self.args.n_seen_obj).to(self.args.device)
            rewards_train[:, pred_reward_idx] = rewards_pred[:, pred_reward_idx]  
            rewards_train[:, real_reward_idx] = rewards                           
            rewards = rewards_train                                               
            if self.args.oracle:
                rewards = rewards_real[:, self.args.seen_obj]  
            forward_obj = self.args.seen_obj  
            n_learning_obj_fake = self.args.n_seen_obj
            preferences_update = preferences  
        else:
            forward_obj = self.args.learning_obj  
            n_learning_obj_fake = self.args.n_learning_obj
            idx = np.where(np.isin(np.array(self.args.seen_obj), np.array(self.args.learning_obj)))[0]  
            preferences_update = preferences[:, idx]  
        return forward_obj, n_learning_obj_fake, preferences_update, rewards


    def cuda(self):
        if self.args.world_model_type == "mlp":
            self.world_model.cuda()

    def save_models(self, path):
        self.world_model.save_models(path)

    def load_models(self, path):
        self.world_model.load_models(path)
