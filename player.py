import numpy as np
import torch
from env import DEFAULT_DTYPE
from network import PolicyNetwork, ValueNetwork


class RandomPolicy():
    def __init__(self, num_of_act=22, dtype=DEFAULT_DTYPE):
        self.num_of_act = num_of_act
        self.dtype = dtype

    def __call__(self, obs=None):
        return torch.ones(self.num_of_act, dtype=self.dtype) / self.num_of_act

class Player():
    def __init__(self, policy_fn, eval_fn=None):
        self.use_nn = isinstance(policy_fn, PolicyNetwork)
        self.policy = policy_fn
        self.eval = eval_fn


if __name__ == '__main__':
    p1 = Player(RandomPolicy())
    p2 = Player(PolicyNetwork())

    obs = torch.rand(26, dtype=DEFAULT_DTYPE)
    a1 = p1.policy(obs)
    a2 = p2.policy(obs)
    print(a1, a2)