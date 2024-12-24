from functools import partial
import gym
from gym.spaces import flatdim
import numpy as np
from gym.wrappers import TimeLimit as GymTimeLimit
from utils.debug import *

def env_fn(env, **kwargs):
    return env(**kwargs)

REGISTRY = {}

class TimeLimit(GymTimeLimit):
    def __init__(self, env, max_episode_steps=None):
        super(TimeLimit, self).__init__(env)
        if max_episode_steps is None and self.env.spec is not None:
            max_episode_steps = env.spec.max_episode_steps
        elif max_episode_steps is None and getattr(self.env, "token", None) == "gym_cooking":
             
            max_episode_steps = self.env.max_steps
         
         
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None
        

    def step(self, action):
        assert (
            self._elapsed_steps is not None
        ), "Cannot call env.step() before calling reset()"
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
             
            done = True

        return observation, reward, done, info
    
    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)

class DiscreteWrapper:
    def __init__(self, key, time_limit, **kwargs):
        self.episode_limit = time_limit
        env_name = kwargs["env_name"]
        self._env = TimeLimit(gym.make(f"{env_name}:{key}"), max_episode_steps=time_limit)

        self._obs = None
        self.longest_action_space = max(self._env.action_space, key=lambda x: x.n)
        self.longest_observation_space = max(
                self._env.observation_space, key=lambda x: x.shape
            )
         

        self._seed = kwargs["seed"]
        self._env.seed(self._seed)

    def step(self, actions):
        """ Returns reward, terminated, info """
        actions = [int(a) for a in actions]
        self._obs, reward, done, info = self._env.step(actions[0])
        self._obs = [self._obs]
         
        return [reward], done, {}

    def get_obs(self):
        return self._obs

    def get_obs_size(self):
        """ Returns the shape of the observation """
        return flatdim(self.longest_observation_space)

    def get_state(self):
        return np.concatenate(self._obs, axis=0).astype(np.float32)
         

    def get_state_size(self):
        return flatdim(self.longest_observation_space)
        
         

    def get_avail_actions(self):
        if hasattr(self._env, "get_avail_actions"):
            avail_actions = [self._env.get_avail_actions()]
        else:
            avail_actions = [flatdim(self._env.action_space) * [1]]
        
        return avail_actions


    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
         
        return flatdim(self.longest_action_space)

    def reset(self):
        """ Returns initial observations and states"""
         
        self._obs = [self._env.reset()]
        return self.get_obs(), self.get_state()

    def render(self):
        pass
         

    def close(self):
        self._env.close()

    def seed(self):
        return self._env.seed

    def save_replay(self):
        pass

    def get_stats(self):
        return {}

    def get_env_info(self):
        env_info = {
            "state_shape": self.get_state_size(),
            "obs_shape": self.get_obs_size(),
            "n_actions": self.get_total_actions(),
            "episode_limit": self.episode_limit,
        }
        return env_info

class ContinuousWrapper:
    def __init__(self, key, time_limit, **kwargs):
        self.episode_limit = time_limit
        env_name = kwargs["env_name"]
        self._env = TimeLimit(gym.make(f"{env_name}:{key}"), max_episode_steps=time_limit)


        self._obs = None
         
        self.longest_action_space = self._env.action_space
         
         
         
        self.longest_observation_space = self._env.observation_space
         

        self._seed = kwargs["seed"]
        self._env.seed(self._seed)

    def step(self, actions):
        """ Returns reward, terminated, info """
         
         
         
        actions = actions.cpu().numpy()
        self._obs, reward, done, info = self._env.step(actions)
        self._obs = [self._obs]
         
        return [reward], done, {}

    def get_obs(self):
        return self._obs

    def get_obs_size(self):
        """ Returns the shape of the observation """
        return flatdim(self.longest_observation_space)

    def get_state(self):
         
        return np.concatenate(self._obs, axis=0).astype(np.float32)
         

    def get_state_size(self):
        return flatdim(self.longest_observation_space)
        
         

    def get_avail_actions(self):
        if hasattr(self._env, "get_avail_actions"):
            avail_actions = [self._env.get_avail_actions()]
        else:
            avail_actions = [flatdim(self._env.action_space) * [1]]
        
        return avail_actions


    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
         
        return flatdim(self.longest_action_space)

    def reset(self):
        """ Returns initial observations and states"""
         
        self._obs = [self._env.reset()]
        return self.get_obs(), self.get_state()

    def render(self):
        pass
         

    def close(self):
        self._env.close()

    def seed(self):
        return self._env.seed

    def save_replay(self):
        pass

    def get_stats(self):
        return {}

    def get_env_info(self):
        env_info = {
            "state_shape": self.get_state_size(),
            "obs_shape": self.get_obs_size(),
            "n_actions": self.get_total_actions(),
            "episode_limit": self.episode_limit,
            "max_action": self.longest_action_space.high,
            'action_dim': self.longest_action_space.shape[0]
        }
        return env_info



REGISTRY["ftn"] = partial(env_fn, env=DiscreteWrapper)
REGISTRY["grid"] = partial(env_fn, env=DiscreteWrapper)
REGISTRY["cmc"] = partial(env_fn, env=ContinuousWrapper)
REGISTRY["mujoco"] = partial(env_fn, env=ContinuousWrapper)