import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from os import path
from ..rewards.reward import Reward

class HopperEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.obj_dim = 2
        self.reward_generator = Reward(env_name='hopper')
        mujoco_env.MujocoEnv.__init__(self, model_path = path.join(path.abspath(path.dirname(__file__)), "assets/hopper.xml"), frame_skip = 5)
        utils.EzPickle.__init__(self)
        self.action_space_type = "Continuous"
        self.reward_space = np.zeros((2,))

    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        old_pos = self.get_body_com("torso").copy()
         
         
        a = np.clip(a, [-2.0, -2.0, -4.0], [2.0, 2.0, 4.0])
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        new_pos = self.get_body_com("torso").copy()
        alive_bonus = 1.0
        reward_others = alive_bonus - 2e-4 * np.square(a).sum()
        reward_run = 1.5 * (posafter - posbefore) / self.dt + reward_others
        reward_jump = 12. * (height - self.init_qpos[1]) + reward_others
        s = self.state_vector()
        done = not((s[1] > 0.4) and abs(s[2]) < np.deg2rad(90) and abs(s[3]) < np.deg2rad(90) and abs(s[4]) < np.deg2rad(90) and abs(s[5]) < np.deg2rad(90))

        ob = self._get_obs()

        rewards = self.reward_generator.get_reward(old_pos, new_pos, a)
        rewards = np.concatenate([np.array([reward_run, reward_jump],dtype=np.float32), rewards])
        return ob, rewards, done, {}
         

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            np.clip(self.sim.data.qvel.flat, -10, 10)
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv
        )
        self.set_state(qpos, qvel)
        
        self.reward_generator.reset(self.get_body_com("torso").copy())
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20
