import torch
from env import DEFAULT_DTYPE
from network import PolicyNetwork, ValueNetwork
from tensordict.nn import TensorDictModule


class UniformPolicy():
    def __init__(self, num_of_act=22, dtype=DEFAULT_DTYPE, device='cpu'):
        self.num_of_act = num_of_act
        self.dtype = dtype
        self.device = device

    def __call__(self, obs=None):
        return torch.ones(self.num_of_act, dtype=self.dtype, device=self.device) / self.num_of_act

class AlwaysCallPolicy():
    def __init__(self, num_of_act=22, dtype=DEFAULT_DTYPE, device='cpu'):
        self.num_of_act = num_of_act
        self.dtype = dtype
        self.device = device

    def __call__(self, obs=None):
        act_probs = torch.zeros(self.num_of_act, dtype=self.dtype, device=self.device)
        act_probs[1] = 1
        return act_probs
    
class AlwaysFoldPolicy():
    def __init__(self, num_of_act=22, dtype=DEFAULT_DTYPE, device='cpu'):
        self.num_of_act = num_of_act
        self.dtype = dtype
        self.device = device

    def __call__(self, obs=None):
        act_probs = torch.zeros(self.num_of_act, dtype=self.dtype, device=self.device)
        act_probs[0] = 1
        return act_probs
    
class AlwaysShovePolicy():
    def __init__(self, num_of_act=22, dtype=DEFAULT_DTYPE, device='cpu'):
        self.num_of_act = num_of_act
        self.dtype = dtype
        self.device = device

    def __call__(self, obs=None):
        act_probs = torch.zeros(self.num_of_act, dtype=self.dtype, device=self.device)
        act_probs[-1] = 1
        return act_probs

class Player():
    def __init__(self, policy_fn, eval_fn=None, use_nn=False):
        self.policy_fn = policy_fn
        self.eval_fn = eval_fn
        self.use_nn = use_nn

    def policy(self, obs):
        return self.policy_fn(obs)

    def eval(self, obs):
        return self.eval_fn(obs)


if __name__ == '__main__':
    unif_policy = UniformPolicy()
    policy_net = PolicyNetwork()
    value_net = ValueNetwork()
    p1 = Player(policy_net, value_net, use_nn=True)
    tsm = TensorDictModule(unif_policy, in_keys='obs', out_keys='prob')
    obs = torch.rand(26)
    out = tsm(obs)
    print(out)