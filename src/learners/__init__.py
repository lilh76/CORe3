from .q_learner import QLearner
from .td3_learner import TD3Learner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY['td3_learner'] = TD3Learner