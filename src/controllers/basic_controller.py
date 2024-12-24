from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th
import os
from copy import deepcopy

from utils.debug import *

class BasicMAC:
    def __init__(self, args):
        self.args = args
        self._build_agent()
        self.agent_output_type = args.agent_output_type
        self.action_selector = action_REGISTRY[args.action_selector](args)

    def create_head(self, learning_obj):
        self.agent.create_head(learning_obj)
        if self.args.use_cuda:
            self.cuda()

    def select_actions(self, state, t_ep, t_env, bs=slice(None), test_mode=False, preference=None):
        state = th.tensor(state, device=self.args.device, dtype=th.float32) 
        agent_outputs = self.forward(state, preference.unsqueeze(0), forward_obj=self.args.seen_obj)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], t_env, preference=preference, test_mode=test_mode)
        return chosen_actions

    def forward(self, state, preference, forward_obj):
        agent_outs = self.agent(state, preference, forward_obj)

        return agent_outs 

    def parameters(self):
        params = list(self.agent.parameters())
        for obj, head in self.agent.obj2head.items():
            params += list(head.parameters())
        return params

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())
        for obj, head in other_mac.agent.obj2head.items():
            self.agent.obj2head[obj] = deepcopy(head)

    def cuda(self):
        self.agent.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))
        for obj, head in self.agent.obj2head.items():
            th.save(head.state_dict(), f"{path}/{obj}_head.th")

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))
        head_files = []
        for file in os.listdir(path):
            if "head" in file:
                head_files.append(file)
        head_files = sorted(head_files)
        for head_file in head_files:
            obj = int(head_file.split('_')[0])
            self.agent.create_head(obj)
            self.agent.obj2head[obj].load_state_dict(th.load(f"{path}/{head_file}", map_location=lambda storage, loc: storage))

    def _build_agent(self):
        self.agent = agent_REGISTRY[self.args.agent](self.args)

