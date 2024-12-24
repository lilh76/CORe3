 
 
 

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from os import path
from ..rewards.reward import Reward

from utils.debug import *

class AntEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.obj_dim = 2
        self.cost_weights = np.ones(self.obj_dim) / self.obj_dim
        self.reward_generator = Reward(env_name='ant')
        mujoco_env.MujocoEnv.__init__(self, model_path = path.join(path.abspath(path.dirname(__file__)), "assets/ant.xml"), frame_skip = 5)
        utils.EzPickle.__init__(self)
        self.action_space_type = "Continuous"
         
        self.reward_space = np.zeros((2,))


         
         



    def step(self, a):
        xposbefore = self.get_body_com("torso")[0]
        yposbefore = self.get_body_com("torso")[1]
        old_pos = self.get_body_com("torso").copy()
         
         
        a = np.clip(a, -1.0, 1.0)

        self.do_simulation(a, self.frame_skip)

        xposafter = self.get_body_com("torso")[0]
        yposafter = self.get_body_com("torso")[1]
        new_pos = self.get_body_com("torso").copy()

        ctrl_cost = .5 * np.square(a).sum()
        survive_reward = 1.0
        other_reward = - ctrl_cost + survive_reward

        vx_reward = (xposafter - xposbefore) / self.dt + other_reward
        vy_reward = (yposafter - yposbefore) / self.dt + other_reward

        reward = self.cost_weights[0] * vx_reward + self.cost_weights[1] * vy_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all()
        done = not notdone
        ob = self._get_obs()

        rewards = self.reward_generator.get_reward(old_pos, new_pos, a)
        rewards = np.concatenate([np.array([vx_reward, vy_reward] ,dtype=np.float32), rewards], dtype=np.float32)
        return ob, rewards, done, {}
         

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.1, high=0.1
        )
        qvel = self.init_qvel + self.np_random.standard_normal(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        self.reward_generator.reset(self.get_body_com("torso").copy())
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def set_params(self, params):
        if params['cost_weights'] is not None:
            self.cost_weights = np.copy(params["cost_weights"])