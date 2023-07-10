import numpy as np
from env import FakeGameTorchEnv, ACT_DICT
from network import PolicyNetwork, ValueNetwork
from player import RandomPolicy, Player
from utils import coll_data

import torch
from tensordict.tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.objectives.value import GAE
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator


#game env
num_of_player = 6
game = FakeGameTorchEnv(num_of_player)

#policy
rand_policy = RandomPolicy()

#policy network
pol_net = PolicyNetwork()

#value network
val_net = ValueNetwork()
val_mod = TensorDictModule(val_net, in_keys=['observation'], out_keys=['state_value'])

#player list
nn_player_sit_idx_lst = [0, 2, 3]
player_lst = []
for sit_idx in range(num_of_player):
    if sit_idx in nn_player_sit_idx_lst:
        player_lst.append(Player(pol_net, val_net))
    else:
        player_lst.append(Player(rand_policy))

gamma = 1.0
lmbda = 0.95
adv_mod = GAE(gamma=gamma, lmbda=lmbda, value_network=val_mod)

num_of_iter = 1
num_of_epoch = 1
for iter_idx in range(num_of_iter):
    data_tsd = coll_data(game, player_lst)
    for _ in range(num_of_epoch):
        adv_mod(data_tsd)
        print(data_tsd)
