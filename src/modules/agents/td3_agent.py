import copy
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy
from utils.debug import *


def init_weights(m):
    if type(m) == nn.Linear:
        th.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0)


class TD3Agent(nn.Module):
    def __init__(self, args):
        super(TD3Agent, self).__init__()
        self.args = args
        self.gru = None
        self.hidden_dim = args.hidden_dim

        self.input_dim = args.state_shape
        if args.handle_preference_input_method == 'mean':
            self.input_dim += 1
        elif args.handle_preference_input_method == 'max':
            self.input_dim += args.max_n_obj
        elif args.handle_preference_input_method == 'rnn' or args.handle_preference_input_method == 'transformer':
            args.preference_hidden_dim = 8
            self.gru = nn.GRU(2, args.preference_hidden_dim, batch_first=True)
            self.input_dim += args.preference_hidden_dim

        self.affine_in = nn.Linear(
            self.input_dim, self.hidden_dim
        )
        self.affine_hid = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.affine_out = nn.Linear(self.hidden_dim, self.args.n_actions)

        self.affine_in.apply(init_weights)
        self.affine_hid.apply(init_weights)
        self.affine_out.apply(init_weights)
        self.max_action = th.tensor(args.max_action, device=args.device)

    def forward(self, state, preference):
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
             
            preference, _ = self.gru(preference)  
            preference = preference[:, -1, :]  
            inputs = th.cat([state, preference], dim=-1)

        x = F.relu(self.affine_in(inputs))
        x = F.relu(self.affine_hid(x))
        x = th.tanh(self.affine_out(x))
        return (self.max_action * x).reshape(-1, 1, self.args.n_actions)
