import torch.nn as nn
import torch.nn.functional as F
import torch as th
import random

from utils.debug import *

class RewardHead(nn.Module):
    def __init__(self, args, hidden_dim, use_stochastic=True):
        super().__init__()
        self.args = args
        self.use_stochastic = use_stochastic
        self.MAX_LOG_VAR = th.tensor(-2, dtype=th.float32)
        self.MIN_LOG_VAR = th.tensor(-5., dtype=th.float32)
        self.mu_output = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, 1)
            )
        self.var_output = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, 1)
            )

    def forward(self, feature):
        mu_output = self.mu_output(feature)  
        log_var_output = self.var_output(feature)
        log_var_output = self.MAX_LOG_VAR - F.softplus(self.MAX_LOG_VAR - log_var_output)
        log_var_output = self.MIN_LOG_VAR + F.softplus(log_var_output - self.MIN_LOG_VAR)
        if not self.use_stochastic:
            log_var_output *= 0.
        return mu_output, log_var_output
    
    def parameters(self):
        params = list(self.mu_output.parameters()) + list(self.mu_output.parameters())
        return params

class MLPWorldModel(nn.Module):
    def __init__(self, args, hidden_dims, use_stochastic=True, index=0):
        super().__init__()
        self.args = args
        self.use_stochastic = use_stochastic
        self.index = index
        self.MAX_LOG_VAR = th.tensor(-2, dtype=th.float32)
        self.MIN_LOG_VAR = th.tensor(-5., dtype=th.float32)
        feature_extractor = []
        feature_extractor += [nn.Linear(args.state_shape + args.n_actions, hidden_dims[0]), nn.LeakyReLU(), ]  
        for i in range(len(hidden_dims) - 1):                                                                  
            feature_extractor += [nn.Linear(hidden_dims[i], hidden_dims[i+1]), nn.LeakyReLU()]
        self.feature_extractor = nn.Sequential(*feature_extractor)
        self.state_mu_output = nn.Sequential(
                nn.Linear(hidden_dims[-1], hidden_dims[-1]),
                nn.LeakyReLU(),
                nn.Linear(hidden_dims[-1], args.state_shape)
            )
        self.state_var_output = nn.Sequential(
                nn.Linear(hidden_dims[-1], hidden_dims[-1]),
                nn.LeakyReLU(),
                nn.Linear(hidden_dims[-1], args.state_shape)
            )
        self.hidden_dims = hidden_dims

        self.obj2reward_head = {}

    def create_head(self, obj):
        new_head = RewardHead(self.args, self.hidden_dims[-1], self.use_stochastic)
        self.obj2reward_head[obj] = new_head

    def forward(self, input, forward_obj):
        feature = self.feature_extractor(input)
        state_mu = self.state_mu_output(feature)  
        state_log_var = self.state_var_output(feature)
        state_log_var = self.MAX_LOG_VAR - F.softplus(self.MAX_LOG_VAR - state_log_var)
        state_log_var = self.MIN_LOG_VAR + F.softplus(state_log_var - self.MIN_LOG_VAR)
        if not self.use_stochastic:
            state_log_var *= 0.
        rewards_outputs = [self.obj2reward_head[obj](feature) for obj in forward_obj]  
        rewards_mu = th.cat([mu_log_var[0] for mu_log_var in rewards_outputs], dim=1)  
        rewards_log_var = th.cat([mu_log_var[1] for mu_log_var in rewards_outputs], dim=1)
        return state_mu, state_log_var, rewards_mu, rewards_log_var
    
    def _cuda(self):
        self.cuda()
        for obj, reward_head in self.obj2reward_head.items():
            reward_head.cuda()

    def parameters(self, phase_idx):
        if self.args.world_model_update_newhead and phase_idx > 0:  
            learning_obj = self.args.learning_obj_lst[phase_idx]
            params = []
            for obj in learning_obj:
                head = self.obj2reward_head.get(obj, None)
                if head is not None:
                     
                    params += list(head.parameters())
                else:
                    assert0('create head before getting model parameters')
        else:
            params = list(self.feature_extractor.parameters()) + \
                    list(self.state_mu_output.parameters()) + \
                    list(self.state_var_output.parameters())
            for obj, reward_head in self.obj2reward_head.items():
                params += list(reward_head.parameters())

        return params
    
    def save_models(self, path):
        th.save(self.feature_extractor.state_dict(), f"{path}/feature_extractor_{self.index}.th")
        for obj, head in self.obj2reward_head.items():
            th.save(head.mu_output.state_dict(), f"{path}/reward_head_mu_{self.index}_{obj}.th")
            th.save(head.var_output.state_dict(), f"{path}/reward_head_log_var_{self.index}_{obj}.th")

    def load_models(self, path):
        self.feature_extractor.load_state_dict(th.load(f"{path}/feature_extractor_{self.index}.th", map_location=lambda storage, loc: storage))
        for obj, head in self.obj2reward_head.items():
            head.mu_output.load_state_dict(th.load(f"{path}/reward_head_mu_{self.index}_{obj}.th", map_location=lambda storage, loc: storage))
            head.var_output.load_state_dict(th.load(f"{path}/reward_head_log_var_{self.index}_{obj}.th", map_location=lambda storage, loc: storage))
            
class MLPModelEnsemble(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = args.device
        self.use_stochastic = args.use_stochastic_world_model
        self.n_models = args.n_world_models
        hidden_dims=(1024, 1024, 1024)
        self.models = [MLPWorldModel(args, hidden_dims, use_stochastic=self.use_stochastic, index=i) for i in range(self.n_models)]

    def create_head(self, obj):
        for model in self.models:
            model.create_head(obj)
            if self.args.use_cuda:
                model._cuda()

    def forward(self, x, forward_obj, model_index=None):
         
        bs = x.shape[0]
        if model_index is None:
            model_outs = [model(x, forward_obj) for model in self.models]
             
            mus_state        = th.cat([outs[0] for outs in model_outs]).reshape(self.n_models, bs, -1)  
            log_vars_state   = th.cat([outs[1] for outs in model_outs]).reshape(self.n_models, bs, -1)  
            mus_rewards      = th.cat([outs[2] for outs in model_outs]).reshape(self.n_models, bs, -1)  
            log_vars_rewards = th.cat([outs[3] for outs in model_outs]).reshape(self.n_models, bs, -1)  
        else:
            model_outs = self.models[model_index](x, forward_obj)
            mus_state        = model_outs[0].reshape(1, bs, -1)
            log_vars_state   = model_outs[1].reshape(1, bs, -1)
            mus_rewards      = model_outs[2].reshape(1, bs, -1)
            log_vars_rewards = model_outs[3].reshape(1, bs, -1)

        return mus_state, log_vars_state, mus_rewards, log_vars_rewards
        
    def predict_rewards(self, inputs, forward_obj):
         
        bs = inputs.shape[0]
        _, _, mus, log_vars = self.forward(inputs, forward_obj)  
        idx = [i for i in range(bs)]
        if self.use_stochastic:
            out = th.normal(mus, th.exp(0.5 * log_vars))
        else:
            out = th.normal(mus, 0.00001)
        randomlist = random.choices(range(out.shape[0]), k=bs)
        out = out[randomlist, idx, :]  
        return out
    
    def cuda(self):
        for model in self.models:
            model._cuda()

    def parameters(self, phase_idx):
        params = []
        for model in self.models:
            params += model.parameters(phase_idx)
        return params

    def save_models(self, path):
        for model in self.models:
            model.save_models(path)

    def load_models(self, path):
        for model in self.models:
            model.load_models(path)