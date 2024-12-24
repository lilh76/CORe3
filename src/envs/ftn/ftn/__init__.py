from gym.envs.registration import registry, register, make, spec
from itertools import product
 

depth = [5, 6, 7]
for d in depth:
    register(
        id=f'ft{d}-v1',
        entry_point='ftn.envs:ftnEnv',
        kwargs={
            'depth': d,
            'max_episode_steps': 100,  
        },
    )


'''
sizes = [13]
players = [1]
coop = [False]
foods = [4]
for s, p, f, c in product(sizes, players, foods, coop):
    register(
        id="Foraging-{0}x{0}-{1}p-{2}f{3}-v1".format(s, p, f, "-coop" if c else ""),
        entry_point="lbforaging.foraging:ForagingEnv",
        kwargs={
            "players": p,
            "max_player_level": 3,
            "field_size": (s, s),
            "max_food": f,
            "sight": s,
            "max_episode_steps": 500,
            "force_coop": c,
        },
    )
'''
