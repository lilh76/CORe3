import numpy as np
import torch
from utils.debug import *

class TransitionReplayBuffer:
    def __init__(self, args, buffer_size, state_shape, action_dim, reward_dim, preference_dim):
        self.args = args
        self.device = args.device
        self.buffer_size = buffer_size
        self.state_shape = state_shape
        self.action_dim = action_dim
        if "int" in str(type(state_shape)):
            self.buffer = {
                "states": np.zeros((buffer_size, state_shape), dtype=np.float32),
                "actions": np.zeros((buffer_size, action_dim), dtype=np.float32),
                "rewards": np.zeros((buffer_size, reward_dim), dtype=np.float32),
                "rewards_real": np.zeros((buffer_size, args.max_n_obj), dtype=np.float32),
                "preferences": np.zeros((buffer_size, preference_dim), dtype=np.float32),
                "next_states": np.zeros((buffer_size, state_shape), dtype=np.float32),
                "dones": np.zeros((buffer_size, 1), dtype=np.float32)
            }
        elif "tuple" in str(type(state_shape)):
            self.buffer = {
                "states": np.zeros((buffer_size, *state_shape), dtype=np.float32),
                "actions": np.zeros((buffer_size, action_dim), dtype=np.float32),
                "rewards": np.zeros((buffer_size, reward_dim), dtype=np.float32),
                "rewards_real": np.zeros((buffer_size, args.max_n_obj), dtype=np.float32),
                "preferences": np.zeros((buffer_size, preference_dim), dtype=np.float32),
                "next_states": np.zeros((buffer_size, *state_shape), dtype=np.float32),
                "dones": np.zeros((buffer_size, 1), dtype=np.float32)
            }
        else:
            assert 0
        self.position = 0
        self.current_size = 0

    def push(self, trajectory):
        num_steps = len(trajectory["states"])
        if self.args.use_her and self.args.weight_num > 0:
            num_steps *= (self.args.weight_num + 1)

        if self.position + num_steps > self.buffer_size:
            self.position = 0

        states = np.array(trajectory["states"])
        actions = np.array(trajectory["actions"])
        rewards = np.array(trajectory["rewards"])
        rewards_real = np.array(trajectory["rewards_real"])
        preferences = np.array(trajectory["preferences"])
        next_states = np.array(trajectory["next_states"])
        dones = np.array(trajectory["dones"])

        if self.args.use_her and self.args.weight_num > 0:
            num = states.shape[0]
            states = np.tile(states, (self.args.weight_num+1,1))
            actions = np.tile(actions, (self.args.weight_num+1,1))
            rewards = np.tile(rewards, (self.args.weight_num+1,1))
            rewards_real = np.tile(rewards_real, (self.args.weight_num+1,1))
            preferences = np.tile(preferences, (self.args.weight_num+1,1))
            next_states = np.tile(next_states, (self.args.weight_num+1,1))
            dones = np.tile(dones, self.args.weight_num+1)

            new_preferences = self.__generate_preference(num)
            preferences[num:] = new_preferences
        

        end_position = self.position + num_steps
        if end_position <= self.buffer_size:
            self.buffer["states"][self.position:end_position] = states
            self.buffer["actions"][self.position:end_position] = actions
            self.buffer["rewards"][self.position:end_position] = rewards

            self.buffer["rewards_real"][self.position:end_position] = rewards_real
            self.buffer["preferences"][self.position:end_position] = preferences
            self.buffer["next_states"][self.position:end_position] = next_states
            self.buffer["dones"][self.position:end_position] = dones.reshape(-1, 1)
        else:
            overflow = end_position - self.buffer_size
            self.buffer["states"][self.position:] = states[:-overflow]
            self.buffer["actions"][self.position:] = actions[:-overflow]
            self.buffer["rewards"][self.position:] = rewards[:-overflow]
            self.buffer["rewards_real"][self.position:] = rewards_real[:-overflow]
            self.buffer["preferences"][self.position:] = preferences[:-overflow]
            self.buffer["next_states"][self.position:] = next_states[:-overflow]
            self.buffer["dones"][self.position:] = dones[:-overflow].reshape(-1, 1)

            self.buffer["states"][:overflow] = states[-overflow:]
            self.buffer["actions"][:overflow] = actions[-overflow:]
            self.buffer["rewards"][:overflow] = rewards[-overflow:]
            self.buffer["rewards_real"][:overflow] = rewards_real[-overflow:]
            self.buffer["preferences"][:overflow] = preferences[-overflow:]
            self.buffer["next_states"][:overflow] = next_states[-overflow:]
            self.buffer["dones"][:overflow] = dones[-overflow:].reshape(-1, 1)

        self.position = (self.position + num_steps) % self.buffer_size
        self.current_size = min(self.current_size + num_steps, self.buffer_size)
        
    def sample(self, batch_size):
        indices = np.random.choice(self.current_size, batch_size, replace=False)
        batch = {
            "states": torch.tensor(self.buffer["states"][indices], device=self.device, dtype=torch.float32),
            "actions": torch.tensor(self.buffer["actions"][indices], device=self.device),
            "rewards": torch.tensor(self.buffer["rewards"][indices], device=self.device),
            "rewards_real": torch.tensor(self.buffer["rewards_real"][indices], device=self.device),
            "preferences": torch.tensor(self.buffer["preferences"][indices], device=self.device, dtype=torch.float32),
            "next_states": torch.tensor(self.buffer["next_states"][indices], device=self.device, dtype=torch.float32),
            "dones": torch.tensor(self.buffer["dones"][indices], device=self.device)
        }
        return batch
    
    def __generate_preference(self, num_steps):
        if self.args.use_world_model: 
            preference = np.random.randn(num_steps * self.args.weight_num, self.args.n_seen_obj)
            preference = (np.abs(preference) / np.linalg.norm(preference, ord=1, axis=1, keepdims=True))
        else:
            preference = np.zeros((num_steps * self.args.weight_num, self.args.n_seen_obj))
            preference_learning_obj = np.random.randn(num_steps * self.args.weight_num, self.args.n_learning_obj)
            preference_learning_obj = (np.abs(preference_learning_obj) / np.linalg.norm(preference_learning_obj, ord=1, axis=1, keepdims=True)) 
            index = np.where(np.isin(np.array(self.args.seen_obj), np.array(self.args.learning_obj)))[0] 
            preference[:, index] = preference_learning_obj 
        return preference

    def can_sample(self, batch_size):
        return self.current_size >= batch_size
