import numpy as np
import torch as tor
from torch import nn
from network import PolicyNetwork, ValueNetwork

class RandomPolicy():
    def __init__(self, num_of_act=22):
        self.num_of_act = num_of_act

    def __call__(self, obs=None):
        return np.ones(self.num_of_act) / self.num_of_act
    
class NNPolicy():
    def __init__(self, num_of_obs=26, num_of_act=22, hid_arch=None):
        self.to_numpy = True
        self.net = PolicyNetwork(num_of_obs, num_of_act, hid_arch)

    def __call__(self, obs):
        if self.to_numpy:
            return self.net(obs).numpy(force=True)
        else:
            return self.net(obs)

class Player():
    def __init__(self, policy_fn, eval_fn=None):
        self.use_nn = isinstance(policy_fn, NNPolicy)
        self.policy = policy_fn
        self.eval = eval_fn


if __name__ == '__main__':
    p1 = Player(RandomPolicy())
    p2 = Player(NNPolicy())

    obs = np.random.rand(26)
    a1 = p1.policy(obs)
    a2 = p2.policy(obs)
    print(a1, a2)