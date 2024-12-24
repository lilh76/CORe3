from envs import REGISTRY as env_REGISTRY
import numpy as np
from utils.debug import *
from pymoo.indicators.hv import Hypervolume

from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import logging

logging.getLogger('matplotlib').setLevel(logging.WARNING)


class EpisodeRunner:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1
        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

         
        self.log_train_stats_t = -1000000

    def setup(self, mac):
        self.mac = mac
        self.train_stats = {}
        self.test_stats = {}
        
        self.learning_obj_lst = self.args.learning_obj_lst
        self.preference_batch_dict = {}
        self.preference_batch_test_lst = []  
        self.forward_obj_lst = []  
         
        for obj in self.learning_obj_lst:
            obj = sorted(obj,  key=lambda x: self.args.seen_obj.index(x))
            l = len(obj)
             
            self.preference_batch_dict[l] = self.preference_batch_dict.get(l, \
                                                th.tensor(self.recursive_generate_preference_batch_test(1, l, preference_step_size=self.args.test_preference_step_size[l]), device=self.args.device))
            preference_batch_padded = th.zeros(self.preference_batch_dict[l].shape[0], self.args.n_seen_obj)\
                                      .to(self.args.device).to(self.preference_batch_dict[l].dtype)
            index = np.where(np.isin(np.array(self.args.seen_obj), np.array(obj)))[0]
            try:
                preference_batch_padded[:, index] = self.preference_batch_dict[l]
            except:
                assert0(obj, self.args.seen_obj, index, preference_batch_padded.shape, self.preference_batch_dict[l].shape)
            self.preference_batch_test_lst.append(preference_batch_padded)
            self.forward_obj_lst.append(obj)
            if obj == self.args.learning_obj:
                break
        
        preference_batch_all = self.recursive_generate_preference_batch_test(1, self.args.n_seen_obj, \
                                                                                            preference_step_size=self.args.test_preference_step_size[self.args.n_seen_obj])
        self.preference_batch_test_lst.append(th.tensor(preference_batch_all, device=self.args.device))
        self.forward_obj_lst.append(self.args.seen_obj)


    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
         
        self.env.reset()
        self.t = 0

    def run(self, preference=None, eps_greedy_t=0):
        self.reset()
        traj = {
            "states": [],
            "actions": [],
            "rewards": [],
            "rewards_real": [],
            "preferences": [],
            "next_states": [],
            "dones": [],
        }

        terminated = False

        while not terminated:
            state = self.env.get_obs()

            actions = self.mac.select_actions(state, t_ep=self.t, t_env=eps_greedy_t, test_mode=False, \
                                              preference=preference)

            reward, terminated, env_info = self.env.step(actions[0])
            reward = reward[0]   
            reward = reward[:self.args.max_n_obj]
            reward_train = reward[self.args.learning_obj]  

            traj["states"].append(state[0])
            traj["actions"].append(actions[0].cpu().numpy())
            traj["rewards"].append(reward_train)
            traj["rewards_real"].append(reward)  
            traj["preferences"].append(preference.cpu().numpy())
            traj["next_states"].append(self.env.get_obs()[0])
            traj["dones"].append(terminated)

            self.t += 1

        cur_stats = self.train_stats
        log_prefix = ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        self.t_env += self.t

        if self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return traj

    def _log(self, stats, prefix):

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean", v / stats["n_episodes"], self.t_env)
        stats.clear()

    
    def eval_agent_discrete(self, final_test=False):
        gamma = 1
        FloatTensor = th.cuda.FloatTensor if self.args.use_cuda else th.FloatTensor

        hypervolume_lst = []
        sparsity_lst = []
        for i in range(len(self.preference_batch_test_lst)):
            preference_batch_test = self.preference_batch_test_lst[i]
            forward_obj = self.forward_obj_lst[i]

            if preference_batch_test.shape[0] > 300:
                preference_batch = preference_batch_test.type(FloatTensor)[
                        np.random.randint(0, preference_batch_test.shape[0], size=300)]
            else:
                preference_batch = preference_batch_test.type(FloatTensor)

            recovered_objs = []
            for i in range(preference_batch.shape[0]):
                test_preference = preference_batch[i, :]
                self.reset()
                terminated = False
                cnt = 0
                tot_rewards = 0
                while not terminated:
                    state = self.env.get_obs()
                    
                    actions = self.mac.select_actions(state, t_ep=self.t, t_env=0, test_mode=True, \
                                                      preference=test_preference)

                    reward, terminated, env_info = self.env.step(actions[0])
                    reward = reward[0]   
                    reward = reward[:self.args.max_n_obj]
                    reward = reward[forward_obj]

                    tot_rewards += reward * np.power(gamma, cnt)
                    cnt += 1
                recovered_objs.append(tot_rewards)

            n_obj = len(forward_obj)
            obj = np.array(recovered_objs)

            if self.args.env_args["key"] == "dst-v1":
                obj_tmp = np.zeros(obj.shape)
                obj_tmp[:, 1] = 19
                obj_tmp = obj_tmp + obj
                hv = Hypervolume(ref_point=np.zeros(n_obj))
                hypervolume = hv.do(-obj_tmp)
            elif self.args.env_args["key"] == "mo-mountaincarcontinuous-v0":
                hv = Hypervolume(ref_point=(np.zeros(n_obj)+500))
                hypervolume = hv.do(-obj)
            elif self.args.env_args["key"] == "mo-lunar-lander-continuous-v2":
                hv = Hypervolume(ref_point=(np.zeros(n_obj)+100))
                hypervolume = hv.do(-obj)
            elif "four_rooms" in self.args.env_args["key"]:
                hv = Hypervolume(ref_point=(np.zeros(n_obj)+1))
                hypervolume = hv.do(-obj)
            else:
                hv = Hypervolume(ref_point=np.zeros(n_obj))
                hypervolume = hv.do(-obj)
            
            hypervolume_lst.append(hypervolume)
            sparsity_lst.append(self.compute_sparsity(-obj))

        return hypervolume_lst, sparsity_lst
    
    def compute_sparsity(self, obj_batch):
        non_dom = NonDominatedSorting().do(obj_batch, only_non_dominated_front=True)        
        objs = obj_batch[non_dom]
        sparsity_sum = 0     
        for objective in range(objs.shape[-1]):
            objs_sort = np.sort(objs[:,objective])
            sp = 0
            for i in range(len(objs_sort)-1):
                sp +=  np.power(objs_sort[i] - objs_sort[i+1],2)
            sparsity_sum += sp
        if len(objs) > 1:
            sparsity = sparsity_sum/(len(objs)-1)
        else:   
            sparsity = 0

        return sparsity
    
    def recursive_generate_preference_batch_test(self, sum_left, dim, preference_step_size)-> np.array:
        assert sum_left >= 0
        assert dim > 0

        if dim == 1:
            return np.array(sum_left, dtype=np.float32).reshape(1,1)
        
        step_size = preference_step_size
        iter_num_float = sum_left / step_size
        iter_num = round(iter_num_float)

        rnd = lambda x: round(x, round(np.log10(1.0 / step_size) + 2))

        preference_lst = []
        for i in range(0, iter_num+1, 1):
            sub_sum_left = rnd(sum_left) - rnd(i * step_size)
            preference = np.array(rnd(i * step_size), dtype=np.float32).reshape(1, 1)
            sub_preference = self.recursive_generate_preference_batch_test(sub_sum_left, dim-1, preference_step_size)
            preference = np.repeat(preference, sub_preference.shape[0], axis=0)
            preference = np.concatenate([preference, sub_preference], axis=1)
            preference_lst.append(preference)
        preference_batch_test = np.concatenate(preference_lst, axis=0)
        return preference_batch_test

