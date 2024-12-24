REGISTRY = {}

from .multi_head_agent import MultiHeadAgent
REGISTRY["multi_head"] = MultiHeadAgent

from .td3_agent import TD3Agent
REGISTRY['td3'] = TD3Agent
