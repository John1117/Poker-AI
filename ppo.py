import numpy as np
from env import FakeGameTorchEnv
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
from torchrl.modules import ProbabilisticActor, ValueOperator
from torch.distributions.categorical import Categorical
from torchrl.objectives import ClipPPOLoss


#game env
dtype = torch.float32
device = 'cpu'
num_of_player = 6
odd_incr = 0.05
game = FakeGameTorchEnv(num_of_player, odd_incr)

#random policy
rand_pol = RandomPolicy()

#policy network
pol_net = TensorDictModule(PolicyNetwork(), in_keys=['observation'], out_keys=['prob'])
#value network
val_net = TensorDictModule(ValueNetwork(), in_keys=['observation'], out_keys=['state_value'])

#player list
nn_player_sit_idx_lst = [0, 2, 3]
player_lst = []
for sit_idx in range(num_of_player):
    if sit_idx in nn_player_sit_idx_lst:
        player_lst.append(Player(pol_net, val_net, use_nn=True))
    else:
        player_lst.append(Player(rand_pol))


#gae module
gamma = 1.0
lmbda = 0.95
compute_adv = GAE(gamma=gamma, lmbda=lmbda, value_network=val_net)


#buffer
num_of_coll = 1
max_buffer_size = num_of_coll * num_of_player * 100
batch_per_epoch = 64
buffer = ReplayBuffer(storage=LazyTensorStorage(max_buffer_size), sampler=SamplerWithoutReplacement())

num_of_iter = 1
num_of_epoch = 1
for iter_idx in range(num_of_iter):
    data_tsd = coll_data(game, player_lst, num_of_coll)
    for _ in range(num_of_epoch):
        #update log prob of act
        print(data_tsd['observation'].shape)
        print(data_tsd['action'].shape)
        act_prob = pol_net(data_tsd['observation'])
        print(act_prob.shape)
        act_distr = Categorical(act_prob)
        
        log_prob = act_distr.log_prob(data_tsd['action'].reshape(-1)).reshape(-1, 1)
        print(log_prob)
        data_tsd.update({'sample_log_prob': log_prob})

        #update adv
        compute_adv(data_tsd)