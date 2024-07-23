import numpy as np
import torch


def check_dim(arr):
    return arr.reshape(1, -1) if arr.ndim < 2 else arr


class BasePolicy():
    def __init__(self, n_act=22, dtype=np.float_):
        self.__N_ACT = n_act
        self.__DTYPE = dtype

    @property
    def n_act(self):
        return self.__N_ACT
    
    @property
    def dtype(self):
        return self.__DTYPE

    def __call__(self, obs=None, valid_act_filter=None):
        NotImplemented


class UniformPolicy(BasePolicy):

    def __init__(self, n_act=22, dtype=np.float_):
        super().__init__(n_act, dtype)
        
    def __call__(self, obs=None, valid_act_filter=None):
        obs = check_dim(obs)
        valid_act_filter = check_dim(valid_act_filter)
        return (valid_act_filter / valid_act_filter.sum(axis=1, keepdims=True)).astype(self.dtype)
    

class AlwaysFoldPolicy(BasePolicy):

    def __init__(self, n_act=22, dtype=np.float_):
        super().__init__(n_act, dtype)
        
    def __call__(self, obs=None, valid_act_filter=None):
        obs = check_dim(obs)
        valid_act_filter = check_dim(valid_act_filter)
        act_probs = np.zeros((len(obs), self.n_act), dtype=self.dtype)
        can_fold = valid_act_filter[:, 0:1]
        act_probs[:, 0:2] = np.concatenate((can_fold, ~can_fold), axis=1)
        return act_probs
    

class AlwaysCallPolicy(BasePolicy):

    def __init__(self, n_act=22, dtype=np.float_):
        super().__init__(n_act, dtype)

    def __call__(self, obs=None, valid_act_filter=None):
        obs = check_dim(obs)
        valid_act_filter = check_dim(valid_act_filter)
        act_probs = np.zeros((len(obs), self.n_act), dtype=super().dtype)
        can_call = valid_act_filter[:, 1:2]
        act_probs[:, 0:2] = np.concatenate((~can_call, can_call), axis=1)
        return act_probs
    

class AlwaysAllInPolicy(BasePolicy):

    def __init__(self, n_act=22, dtype=np.float_):
        super().__init__(n_act, dtype)
        
    def __call__(self, obs=None, valid_act_filter=None):
        obs = check_dim(obs)
        valid_act_filter = check_dim(valid_act_filter)
        act_probs = np.zeros((len(obs), self.n_act), dtype=self.dtype)
        can_all_in = valid_act_filter[:, 2:3]
        act_probs[:, 0:4:2] = np.concatenate((~can_all_in, can_all_in), axis=1)
        return act_probs
    

class NeuralNetworkPolicy(BasePolicy):

    def __init__(self, n_act=22, dtype=np.float_, model=None):
        super().__init__(n_act, dtype)
        self.model = model

    def __call__(self, obs=None, valid_act_filter=None, return_torch_tensor=False):
        obs = check_dim(obs)
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs)
        act_probs = self.model(obs)
        if return_torch_tensor:
            return act_probs
        else:
            return act_probs.detach().numpy()
        
class NeuralNetworkEvaluator():

    def __init__(self, dtype=np.float_, model=None):
        self.model = model

    def __call__(self, obs=None, return_torch_tensor=False):
        obs = check_dim(obs)
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs)
        act_probs = self.model(obs)
        if return_torch_tensor:
            return act_probs
        else:
            return act_probs.detach().numpy()
