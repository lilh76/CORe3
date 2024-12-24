import torch as th
from torch.distributions import Categorical
from .epsilon_schedules import DecayThenFlatSchedule

from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

from utils.debug import *

REGISTRY = {}

class EpsilonGreedyActionSelector:
    def __init__(self, args):
        self.args = args
        self.schedule = DecayThenFlatSchedule(
            args.epsilon_start,
            args.epsilon_finish,
            args.epsilon_anneal_time,
            decay="linear",
        )
        self.epsilon = self.schedule.eval(0)

        self.FloatTensor = th.cuda.FloatTensor if args.use_cuda else th.FloatTensor
        self.LongTensor = th.cuda.LongTensor if args.use_cuda else th.LongTensor

    def select_action(self, agent_inputs, t_env, preference=None, test_mode=False):
        assert agent_inputs.shape[0] == 1
        q_values = agent_inputs[0] 
        q_values = q_values.transpose(1, 0)  
        q_values = th.mv(q_values, preference)  
        agent_inputs = q_values.unsqueeze(0).unsqueeze(
            0
        )  

        self.epsilon = self.schedule.eval(t_env)

        if test_mode:
            self.epsilon = 0.0

        masked_q_values = agent_inputs.clone()

        random_numbers = th.rand_like(agent_inputs[:, :, 0])
        pick_random = (random_numbers < self.epsilon).long()
        avail_actions = agent_inputs * 0 + 1
        random_actions = Categorical(avail_actions.float()).sample().long()

        picked_actions = (
            pick_random * random_actions
            + (1 - pick_random) * masked_q_values.max(dim=2)[1]
        )
        return picked_actions


REGISTRY["epsilon_greedy"] = EpsilonGreedyActionSelector


class ContinuousActionSelector:
    def __init__(self, args):
        self.args = args
        self.max_action = th.tensor(args.max_action, device=args.device)

    def select_action(self, agent_inputs, t_env, preference=None, test_mode=False):
        action = agent_inputs[0, 0]

        if not test_mode:
            chosen = (
                action + th.normal(0.0, self.max_action * self.args.expl_noise)
            ).clip(-self.max_action, self.max_action)
            
            if t_env < self.args.start_timesteps:
                chosen = th.tensor(self.args.action_space.sample(), device=self.args.device)
        else:
            chosen = action


        
        chosen = chosen.unsqueeze(0).detach()
        return chosen


REGISTRY["continuous"] = ContinuousActionSelector
