from typing import Iterator
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from utils.debug import *
from modules.agents.multi_head_agent import FeatureExtractor, Head

def init_weights(m):
    if type(m) == nn.Linear:
        th.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0)

class TD3Critic(nn.Module):
    def __init__(self, args):
        super(TD3Critic, self).__init__()
        self.args = args
        self.gru = None
        input_shape = args.state_shape + args.n_actions
        if args.handle_preference_input_method == 'mean':
            input_shape += 1
        elif args.handle_preference_input_method == 'max':
            input_shape += args.max_n_obj
        elif args.handle_preference_input_method == 'rnn' or args.handle_preference_input_method == 'transformer':
            args.preference_hidden_dim = 8
            self.gru = nn.GRU(2, args.preference_hidden_dim, batch_first=True)
            input_shape += args.preference_hidden_dim

        self.feature_extractor = FeatureExtractorTD3(args, input_shape)

        self.obj2head = {}

    def create_head(self, learning_obj):
        new_head = Head(self.args, self.args.hidden_dim, 1)
        if self.args.use_cuda:
            new_head.cuda()
        self.obj2head[learning_obj] = new_head  

    def forward(self, states, actions, preference, forward_obj):
        h = self.feature_extractor(states, actions, preference, self.gru)  
        q = [self.obj2head[obj](h) for obj in forward_obj]
        q = th.cat(q, dim=-1)  
        return q
    
    def parameters(self):
        params = list(self.feature_extractor.parameters())
        for obj, head in self.obj2head.items():
            params += list(head.parameters())
        return params

class FeatureExtractorTD3(FeatureExtractor):
    def __init__(self, args, input_dim) -> None:
        super(FeatureExtractorTD3, self).__init__(args, input_dim)

    def forward(self, state, action, preference, gru):
         
        if self.args.handle_preference_input_method == 'mean':
            preference_feature = preference.mean(dim=-1, keepdim=True)
            inputs = th.cat([state, action, preference_feature], dim=-1)
        elif self.args.handle_preference_input_method == 'max':
            bs, n_learning_obj = preference.shape
            padding_num = self.args.max_n_obj - n_learning_obj
            paddings = th.zeros(bs, padding_num).to(preference.device)
            padded_preference = th.cat([preference, paddings], dim=1)
            inputs = th.cat([state, action, padded_preference], dim=-1)
        elif self.args.handle_preference_input_method == 'rnn' or self.args.handle_preference_input_method == 'transformer':
            bs, n_learning_obj = preference.shape
            preference = preference.unsqueeze(1)  
            idx_tensor = th.arange(0, preference.shape[-1]).to(preference.device).unsqueeze(0).unsqueeze(0).repeat(bs, 1, 1)  
            preference = th.cat((idx_tensor, preference), dim=1)  
            preference = th.transpose(preference, 1, 2)  
             
            preference, _ = gru(preference)  
            preference = preference[:, -1, :]  
            inputs = th.cat([state, action, preference], dim=-1)
        return self.mlp(inputs)
