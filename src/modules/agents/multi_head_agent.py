import copy
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy
from utils.debug import *

class MultiHeadAgent(nn.Module):
    def __init__(self, args):
        super(MultiHeadAgent, self).__init__()
        self.args = args
        self.gru = None
        input_shape = args.state_shape
        if args.handle_preference_input_method == 'mean':
            input_shape += 1
        elif args.handle_preference_input_method == 'max':
            input_shape += args.max_n_obj
        elif args.handle_preference_input_method == 'rnn' or args.handle_preference_input_method == 'transformer':
            args.preference_hidden_dim = 8
            self.gru = nn.GRU(2, args.preference_hidden_dim, batch_first=True)
            input_shape += args.preference_hidden_dim

        self.feature_extractor = FeatureExtractor(args, input_shape)

        self.obj2head = {}

    def create_head(self, learning_obj):
        new_head = Head(self.args, self.args.hidden_dim, self.args.n_actions)
        if self.args.use_cuda:
            new_head.cuda()
        self.obj2head[learning_obj] = new_head       

    def forward(self, state, preference, forward_obj):
        h = self.feature_extractor(state, preference, self.gru)  
        q = [self.obj2head[obj](h).unsqueeze(1) for obj in forward_obj]
        q = th.cat(q, dim=1)  
        return q
    
class FeatureExtractor(nn.Module):
    def __init__(self, args, input_dim) -> None:
        super(FeatureExtractor, self).__init__()
        self.args = args
        hidden_dim = args.hidden_dim
        
        if args.n_feature_extractor_layer == 1:
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
            )
        elif args.n_feature_extractor_layer == 2:
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
        elif args.n_feature_extractor_layer == 3:
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
        elif args.n_feature_extractor_layer == 4:
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
        else:
            assert 0

    def forward(self, state, preference, gru):
         
        if self.args.handle_preference_input_method == 'mean':
            preference_feature = preference.mean(dim=-1, keepdim=True)
            inputs = th.cat([state, preference_feature], dim=-1)
        elif self.args.handle_preference_input_method == 'max':
            bs, n_learning_obj = preference.shape
            padding_num = self.args.max_n_obj - n_learning_obj
            paddings = th.zeros(bs, padding_num).to(preference.device)
            padded_preference = th.cat([preference, paddings], dim=1)
            inputs = th.cat([state, padded_preference], dim=-1)
        elif self.args.handle_preference_input_method == 'rnn' or self.args.handle_preference_input_method == 'transformer':
            bs, n_learning_obj = preference.shape
            preference = preference.unsqueeze(1)  
            idx_tensor = th.arange(0, preference.shape[-1]).to(preference.device).unsqueeze(0).unsqueeze(0).repeat(bs, 1, 1)  
            preference = th.cat((idx_tensor, preference), dim=1)  
            preference = th.transpose(preference, 1, 2)  
             
            preference, _ = gru(preference)  
            preference = preference[:, -1, :]  
            inputs = th.cat([state, preference], dim=-1)
        return self.mlp(inputs)

class Head(nn.Module):
    def __init__(self, args, feature_input_dim, output_dim) -> None:
        super(Head, self).__init__()
        self.args = args
        input_dim = feature_input_dim
        hidden_dim = args.hidden_dim

        if args.n_head_layer == 1:
            self.mlp = nn.Linear(input_dim, output_dim)
        elif args.n_head_layer == 2:
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )
        elif args.n_head_layer == 3:
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )
        elif args.n_head_layer == 4:
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )
        else:
            assert 0

    def forward(self, inputs):
        return self.mlp(inputs)
