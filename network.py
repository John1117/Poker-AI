import numpy as np
import torch
from torch import nn


def get_nn_arch(n_input=72, n_output=22, hidden_arch=None):
    nn_arch = (n_input,)
    if hidden_arch is None:
        nn_arch += (n_output,)
    elif isinstance(hidden_arch, int) and hidden_arch > 0:
        nn_arch += (hidden_arch, n_output)
    elif isinstance(hidden_arch, (list, tuple)) and all(isinstance(n_neuron, int) and n_neuron > 0 for n_neuron in hidden_arch):
        nn_arch += tuple(hidden_arch) + (n_output,)
    else:
        raise TypeError('hidden_arch should be None, positive int, or 1d-list/tuple of positive int')
    return nn_arch

class PolicyNetwork(nn.Module):
    def __init__(self, n_observation=27, n_action=22, hidden_arch=None, dtype=torch.float64, device='cpu'):
        super().__init__()

        nn_arch = get_nn_arch(n_observation, n_action, hidden_arch)
        
        n_layer = len(nn_arch)
        self.model = nn.Sequential()
        for i in range(n_layer-1):
            if i == n_layer-2:
                final_layer = nn.Linear(nn_arch[i], nn_arch[i+1], dtype=dtype, device=device)
                # with torch.no_grad():
                #     nn.init.zeros_(final_layer.weight)
                #     final_layer.weight[0] = 1
                #     final_layer.weight[2] = 1
                #     nn.init.zeros_(final_layer.bias)
                #     final_layer.bias[0] = 1
                #     final_layer.bias[2] = 1
                self.model.append(final_layer)
                self.model.append(nn.Softmax(dim=-1))
            else:
                self.model.append(nn.Linear(nn_arch[i], nn_arch[i+1], dtype=dtype, device=device))
                self.model.append(nn.ReLU())

    def forward(self, obs):
        # if isinstance(obs, (int, float, list, tuple, range, np.ndarray)):
        #     #warnings.warn('input of policy net should be Tensor')
        #     obs = torch.Tensor(obs)
        return self.model(obs)
    
class ValueNetwork(nn.Module):
    def __init__(self, n_observation=27, hidden_arch=None, dtype=torch.float64, device='cpu'):
        super().__init__()
        
        nn_arch = get_nn_arch(n_observation, 1, hidden_arch)

        n_layer = len(nn_arch)
        self.model = nn.Sequential()
        for i in range(n_layer-1):
            if i == n_layer-2:
                self.model.append(nn.Linear(nn_arch[i], nn_arch[i+1], dtype=dtype, device=device))
            else:
                self.model.append(nn.Linear(nn_arch[i], nn_arch[i+1], dtype=dtype, device=device))
                self.model.append(nn.ReLU())


    def forward(self, obs):
        # if isinstance(obs, (int, float, list, tuple, range, np.ndarray)):
        #     #warnings.warn('input of policy net should be Tensor')
        #     obs = torch.Tensor(obs)
        return self.model(obs)

