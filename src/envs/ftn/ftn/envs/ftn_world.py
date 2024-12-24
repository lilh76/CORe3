import sys
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__)))[:-12])
from utils.debug import *
import numpy as np
import gym
from gym import Env
from gym.utils import seeding

from .fruit_data import fruit_data

class ftnEnv(Env):
    def __init__(self, depth, max_episode_steps) -> None:
        self.reward_dim = 6
        self.tree_depth = depth  
        branches = np.zeros((int(2 ** self.tree_depth - 1), self.reward_dim))
        fruits = np.array(fruit_data.FRUITS[str(depth)])
        self.tree = np.concatenate(
            [
                branches,
                fruits
            ])

        self.action_space = gym.spaces.Tuple(tuple([gym.spaces.Discrete(2)]))
        self.observation_space = gym.spaces.Tuple(tuple([gym.spaces.Box(np.array([0, 0]),np.array([self.tree_depth, 2 ** self.tree_depth-1]))]))

        self.current_state = np.array([0, 0])
        self.terminal = False

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def get_avail_actions(self):
        if self.current_state[0] == self.tree_depth:
            return [0, 0]
        else:
            return [1, 1]

    def get_ind(self, pos):
        return int(2 ** pos[0] - 1) + pos[1]

    def get_tree_value(self, pos):
        return self.tree[self.get_ind(pos)]

    def reset(self):
        self.current_state = np.array([0, 0])
        self.terminal = False
        return self.current_state
    
    def close(self):
        pass

    def step(self, action):
        '''
            step one move and feed back reward
        '''
        direction_dict = {
            0: np.array([1, self.current_state[1]]),   
            1: np.array([1, self.current_state[1] + 1]),   
        }
        
        direction = direction_dict[action]
        self.current_state = self.current_state + direction

        reward = self.get_tree_value(self.current_state)
        if self.current_state[0] == self.tree_depth:
            self.terminal = True
        
        return self.current_state, reward, self.terminal, {}


if __name__ == '__main__':
    
    a = ftnEnv(6, 1)