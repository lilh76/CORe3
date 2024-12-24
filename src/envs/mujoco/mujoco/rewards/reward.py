import math
import numpy as np
from argparse import Namespace

from utils.debug import *

class Reward:
    def __init__(self, env_name, args:Namespace=None) -> None:
        self.base_reward = args.base_reward if hasattr(args, 'base_reward') else 1.0
        self.speed2keep = args.speed2keep if hasattr(args, 'speed2keep') else 0.5
        self.direction = args.direction if hasattr(args, 'direction') else np.array([1,1], dtype=np.float32)
        self.init_pos = np.array([0, 0, 0], dtype=np.float32)
        self.env_name = env_name
        pass

    def get_reward(self, old_pos, new_pos, action):
         
        backward_reward = self.backward_reward(old_pos=old_pos, new_pos=new_pos)
        downward_reward = self.downward_reward(old_pos=old_pos, new_pos=new_pos)
        energy_reward = self.energy_reward(action=action)

        if self.env_name == 'ant':
            ybackward_reward = self.ybackward_reward(old_pos=old_pos, new_pos=new_pos)
            reward = np.array([backward_reward, energy_reward, ybackward_reward], dtype=np.float32)
        elif self.env_name == 'hopper':
            reward = np.array([backward_reward, energy_reward, downward_reward], dtype=np.float32)
        else:
            assert0('no such an environment')
        
        return reward

     
    def forward_reward(self, old_pos, new_pos):
        dis = new_pos[0] - old_pos[0]
        return dis * self.base_reward * 40

    def backward_reward(self, old_pos, new_pos):
        dis = new_pos[0] - old_pos[0]
        return -(dis * self.base_reward * 40)
    
    def ybackward_reward(self, old_pos, new_pos):
        dis = new_pos[1] - old_pos[1]
        return -(dis * self.base_reward * 40)
    
    def upward_reward(self, old_pos, new_pos):
         
        high = new_pos[2] - self.init_pos[2]
        return high * self.base_reward * 12
    
    def downward_reward(self, old_pos, new_pos):
         
        high = new_pos[2] - self.init_pos[2]
        return -(high * self.base_reward * 12)
    
    def energy_reward(self, action):
        reward_energy = 5.0 - 1.0 * np.square(action).sum() 
        return reward_energy
    
    def speed_keeping_reward(self, old_pos, new_pos):
        new_speed = np.linalg.norm((old_pos - new_pos)[:2], ord=2)
        return self.base_reward / (math.pow(new_speed - self.speed2keep, 2) + 1e-1)
    
    def direction_keeping_reward(self, old_pos, new_pos):
         
        base_reward = self.base_reward * 5
        new_direction = (new_pos - old_pos)[:2]
        if np.linalg.norm(new_direction, ord=2) == 0:
            return -1 * base_reward 
        cos_sim = new_direction.dot(self.direction) / (np.linalg.norm(new_direction, ord=2)) * (np.linalg.norm(self.direction, ord=2))
        return cos_sim * base_reward 
    
    def stable_reward(self, old_pos, new_pos):
        dis = np.linalg.norm((new_pos - old_pos)[:2], ord=2)
        if dis == 0:
            return 0
        delta_height = abs(new_pos[2] - old_pos[2])
        tan = delta_height / dis
        return self.base_reward / (tan + 1e-1)

    def reset(self, init_pos):
        self.init_pos = init_pos


