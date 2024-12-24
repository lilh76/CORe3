from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th
from torch.autograd import Variable
from torch.distributions import Categorical
import torch.nn.functional as F
from copy import deepcopy
import os

from utils.debug import *


class ConMAC:
    def __init__(self, args):
        self.args = args
        self._build_agent()
        self.action_selector = action_REGISTRY[args.action_selector](args)

    def create_head(self, learning_obj):
        self.agent.create_head(learning_obj)
        if self.args.use_cuda:
            self.cuda()

    def select_actions(self, state, t_ep, t_env, bs=slice(None), test_mode=False, preference=None):
        state = th.tensor(state, device=self.args.device, dtype=th.float32) 
        agent_outputs = self.forward(state, preference.unsqueeze(0))
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], t_env, preference=preference, test_mode=test_mode)
        return chosen_actions

    def forward(self, state, preference):
        agent_outs = self.agent(state, preference)
        return agent_outs  
    
    def parameters(self):
        params = list(self.agent.parameters())
        return params

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agent(self):
        self.agent = agent_REGISTRY[self.args.agent](self.args)
