from gym.envs.registration import registry, register, make, spec
from itertools import product
 
 
sizes = [11]
foods = [5]
for s, f in product(sizes, foods):
    register(
        id=f"grid-{s}x{s}-{f}f-v1",
        entry_point="grid.envs:gridEnv",
        kwargs={
            "field_size": (s, s),
            "max_food": f,
            "max_episode_steps": 50,  
        },
    )