import numpy as np
import torch


def check_dim(arr):
    return arr.reshape(1, -1) if arr.ndim < 2 else arr


class BasePolicy():
    def __init__(self, n_action=22, dtype=np.float_):
        self.n_action = n_action
        self.dtype = dtype

    def __call__(self, observation=None, valid_action_filter=None):
        NotImplemented


class UniformPolicy(BasePolicy):

    def __init__(self, n_action=22, dtype=np.float_):
        super().__init__(n_action, dtype)
        
    def __call__(self, observation=None, valid_action_filter=None):
        observation = check_dim(observation)
        valid_action_filter = check_dim(valid_action_filter)
        return (valid_action_filter / valid_action_filter.sum(axis=1, keepdims=True)).astype(self.dtype)
    

class AlwaysFoldPolicy(BasePolicy):

    def __init__(self, n_action=22, dtype=np.float_):
        super().__init__(n_action, dtype)
        
    def __call__(self, observation=None, valid_action_filter=None):
        observation = check_dim(observation)
        valid_action_filter = check_dim(valid_action_filter)
        action_probs = np.zeros((len(observation), self.n_action), dtype=self.dtype)
        can_fold = valid_action_filter[:, 0:1]
        action_probs[:, 0:2] = np.concatenate((can_fold, ~can_fold), axis=1)
        return action_probs
    

class AlwaysCallPolicy(BasePolicy):

    def __init__(self, n_action=22, dtype=np.float_):
        super().__init__(n_action, dtype)

    def __call__(self, observation=None, valid_action_filter=None):
        observation = check_dim(observation)
        valid_action_filter = check_dim(valid_action_filter)
        action_probs = np.zeros((len(observation), self.n_action), dtype=self.dtype)
        can_call = valid_action_filter[:, 1:2]
        action_probs[:, 0:2] = np.concatenate((~can_call, can_call), axis=1)
        return action_probs
    

class AlwaysAllInPolicy(BasePolicy):

    def __init__(self, n_action=22, dtype=np.float_):
        super().__init__(n_action, dtype)
        
    def __call__(self, observation=None, valid_action_filter=None):
        observation = check_dim(observation)
        valid_action_filter = check_dim(valid_action_filter)
        action_probs = np.zeros((len(observation), self.n_action), dtype=self.dtype)
        can_all_in = valid_action_filter[:, 2:3]
        action_probs[:, 0:4:2] = np.concatenate((~can_all_in, can_all_in), axis=1)
        return action_probs
    

class NeuralNetworkPolicy(BasePolicy):

    def __init__(self, n_action=22, dtype=np.float_, model=None):
        super().__init__(n_action, dtype)
        self.model = model

    def __call__(self, observation=None, valid_action_filter=None, return_torch_tensor=False):
        observation = check_dim(observation)
        if isinstance(observation, np.ndarray):
            observation = torch.tensor(observation)
        action_probs = self.model(observation)
        if return_torch_tensor:
            return action_probs
        else:
            return action_probs.detach().numpy()
        
class NeuralNetworkEvaluator():

    def __init__(self, dtype=np.float_, model=None):
        self.model = model

    def __call__(self, observation=None, return_torch_tensor=False):
        observation = check_dim(observation)
        if isinstance(observation, np.ndarray):
            observation = torch.tensor(observation)
        action_probs = self.model(observation)
        if return_torch_tensor:
            return action_probs
        else:
            return action_probs.detach().numpy()
