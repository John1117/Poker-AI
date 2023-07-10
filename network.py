import numpy as np
import torch
from torch import nn
import warnings
from env import DEFAULT_DTYPE
from tensordict.nn import TensorDictModule


def get_nn_arch(num_of_input=26, num_of_output=22, hid_arch=None):
    nn_arch = (num_of_input,)
    if hid_arch is None:
        nn_arch += (num_of_output,)
    elif isinstance(hid_arch, int) and hid_arch > 0:
        nn_arch += (hid_arch, num_of_output)
    elif isinstance(hid_arch, (list, tuple)) and all(isinstance(n_neu, int) and n_neu > 0 for n_neu in hid_arch):
        nn_arch += tuple(hid_arch) + (num_of_output,)
    else:
        raise TypeError('hid_arch should be None, positive int, or 1d-list/tuple of positive int')
    return nn_arch

class PolicyNetwork(nn.Module):
    def __init__(self, num_of_obs=26, num_of_act=22, hid_arch=None, dtype=DEFAULT_DTYPE, device=None):
        super().__init__()

        nn_arch = get_nn_arch(num_of_obs, num_of_act, hid_arch)
        
        n_lyr = len(nn_arch)
        seq = nn.Sequential()
        for i in range(n_lyr-1):
            if i == n_lyr-2:
                seq.append(nn.Linear(nn_arch[i], nn_arch[i+1], dtype=dtype, device=device))
                seq.append(nn.Softmax(dim=-1))
            else:
                seq.append(nn.Linear(nn_arch[i], nn_arch[i+1], dtype=dtype, device=device))
                seq.append(nn.ReLU())

        self.model = seq

    def forward(self, obs):
        if isinstance(obs, (int, float, list, tuple, range, np.ndarray)):
            #warnings.warn('input of policy net should be Tensor')
            obs = torch.Tensor(obs)
        return self.model(obs)
    
class ValueNetwork(nn.Module):
    def __init__(self, num_of_obs=26, hid_arch=None, dtype=DEFAULT_DTYPE, device=None):
        super().__init__()
        
        nn_arch = get_nn_arch(num_of_obs, 1, hid_arch)

        n_lyr = len(nn_arch)
        seq = nn.Sequential()
        for i in range(n_lyr-1):
            if i == n_lyr-2:
                seq.append(nn.Linear(nn_arch[i], nn_arch[i+1], dtype=dtype, device=device))
            else:
                seq.append(nn.Linear(nn_arch[i], nn_arch[i+1], dtype=dtype, device=device))
                seq.append(nn.ReLU())

        self.model = seq

    def forward(self, obs):
        if isinstance(obs, (int, float, list, tuple, range, np.ndarray)):
            #warnings.warn('input of policy net should be Tensor')
            obs = torch.Tensor(obs)
        return self.model(obs)


if __name__=='__main__':
    obs = torch.rand(26)
    pn = PolicyNetwork(hid_arch=(40,40))
    pnm = TensorDictModule(pn, ['obs'], ['act'])
    print('\n', pnm)

    vn = ValueNetwork(hid_arch=(40,40))
    print('\n', vn)
    