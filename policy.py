import numpy as np
import warnings
import torch
from torch import nn
from network import PolicyNetwork


DEFAULT_DTYPE = np.float32


def check_obs(obs, dtype=DEFAULT_DTYPE):
    if isinstance(obs, (list, tuple)):
        obs = np.array(obs)
    elif isinstance(obs, np.ndarray):
        pass
    else:
        raise TypeError('obs should be list, tuple, np.ndarray')

    if obs.ndim == 1:
        obs = np.expand_dims(obs, axis=0)
    elif obs.ndim == 2:
        pass
    else:
        raise ValueError('dim of obs should be 1 or 2')
    
    if obs.shape[-1] != 20:
        raise ValueError('shape of obs should be 20')

     
    # obs = [hand(0:2), flop(2:5), turn(5:6), river(6:7), pot(7:8), bet(8:14), chip(14,20), line(20:26)]
    card = obs[:, 0:7]
    if np.any((card < 0) & (card > 52)) or np.any(np.mod(card, 1) != 0):
        raise ValueError('invalid card code')
    
    chip = obs[:, 7:14]
    if np.any(chip < 0):
        raise ValueError('negative chip')
    
    line = obs[:, 14:20]
    if np.any(np.mod(line, 1) != 0):
        raise ValueError('invalid action line code')
    
    hand = obs[:, 0:2]
    if np.any(hand[:, :-1] > hand[:, 1:]):
        warnings.warn('unsorted hand cards')
        obs[:, 0:2] = np.sort(hand, axis=-1)
    
    flop = obs[:, 2:5]
    if np.any(flop[:, :-1] > flop[:, 1:]):
        warnings.warn('unsorted flop')
        obs[:, 2:5] = np.sort(flop, axis=-1)

    return obs



"""
class NNPolicy():
    def __init__(self, n_obs=20, n_act=22, hid_arch=None):
        self.net = PolicyNetwork(n_obs, n_act, hid_arch)

    def __call__(self, obs):
        return self.net(obs)
    
class RandomPolicy():
    def __init__(self, n_act=22):
        self.n_act = n_act

    def __call__(self, obs):
        l = np.random.rand(self.n_act)
        return l/l.sum(axis=-1, keepdims=True)
"""