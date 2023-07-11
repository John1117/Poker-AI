import numpy as np
import torch
from env import DEFAULT_DTYPE
from network import PolicyNetwork, ValueNetwork
from tensordict.nn import TensorDictModule

class RandomPolicy():
    def __init__(self, num_of_act=22, dtype=DEFAULT_DTYPE):
        self.num_of_act = num_of_act
        self.dtype = dtype

    def __call__(self, obs=None):
        return torch.ones(self.num_of_act, dtype=self.dtype) / self.num_of_act

class Player():
    def __init__(self, policy_fn, eval_fn=None, use_nn=False):
        self.policy = policy_fn
        self.eval = eval_fn
        self.use_nn = use_nn


if __name__ == '__main__':
    p1 = Player(RandomPolicy())
    pn = PolicyNetwork()
    p2 = Player(TensorDictModule(PolicyNetwork(), in_keys=['observation'], out_keys=['prob']))

    obs = torch.rand(26, dtype=DEFAULT_DTYPE)
    a1 = p1.policy(obs)
    a2 = p2.policy(obs)
    print(a1, a2)