from gym.envs.registration import register
register(
    id = 'MO-Ant-v2',
    entry_point = 'mujoco.envs:AntEnv',
    max_episode_steps=500,
)

register(
    id = 'MO-Hopper-v2',
    entry_point = 'mujoco.envs:HopperEnv',
    max_episode_steps=500,
)

