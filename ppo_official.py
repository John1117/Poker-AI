from env import FakeGameTorchEnv, GameTorchEnv, ACT_DICT
from network import PolicyNetwork, ValueNetwork
from player import UniformPolicy, Player, AlwaysCallPolicy, AlwaysFoldPolicy, AlwaysShovePolicy
from utils import coll_data
from collections import defaultdict 
import matplotlib.pyplot as plt

import torch
from torch.optim import Adam
from torch.distributions.categorical import Categorical
from torchrl.objectives.value.functional import vec_generalized_advantage_estimate

# replay buffer stuff
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage



# create the game env
device = 'cpu'
game = GameTorchEnv(num_of_player=2, act_dict=ACT_DICT, take_turn=False, dtype=torch.float32, device=device)

# training device
training_device = 'mps'

# shove policy
shove_pol = AlwaysShovePolicy(num_of_act=game.num_of_act)

# network
hid_arch = (16, 16)
pol_net = PolicyNetwork(num_of_obs=game.num_of_obs, num_of_act=game.num_of_act, hid_arch=hid_arch)
val_net = ValueNetwork(num_of_obs=game.num_of_obs, num_of_act=game.num_of_act, hid_arch=hid_arch)

# player list
player_lst = [Player(shove_pol), Player(pol_net, val_net, use_nn=True)]

eps = 0.2
gamma = 1.0
lmbda = 1.0

num_of_iter = 5000
num_of_coll = 100
max_buffer_size = num_of_coll * 10
num_of_epoch = 10
batch_per_epoch = max_buffer_size

buffer = ReplayBuffer(storage=LazyTensorStorage(max_buffer_size), sampler=SamplerWithoutReplacement(), batch_size=batch_per_epoch)

lr = 3e-4
optim = Adam(params=[{'params': pol_net.parameters()}, {'params': val_net.parameters()}], lr=lr)

# training iteration start here
log = defaultdict(list)
plt.ion()
fig = plt.figure()
ax = fig.add_subplot()
for iter_idx in range(num_of_iter):
    data = coll_data(game, player_lst, num_of_coll)
    final_rwd = data['next']['rwd'][data['next']['done']]

    log['avg_rwd'].append(final_rwd.mean().item())
    log['rwd_std'].append(final_rwd.std().item())
    #print(f'iter = {iter_idx}: avg rwd before training = {log["avg_rwd"][-1]}')

    # (?) update log prob of act, since the pol_net has changed after every epoch
    #act_probs = pol_net(data['obs'])
    #act_distr = Categorical(act_probs)
    #act_log_prob = act_distr.log_prob(data['act'].reshape(-1)).reshape(-1, 1)
    #data.set('act_log_prob', act_log_prob.detach())
    
    for i in range(num_of_epoch):
        # evaluate val again, since the val_net has changed after every epoch
        val = val_net(data['obs'])
        next_val = val_net(data['next']['obs'])
        data.set('val', val)
        data.set(('next', 'val'), next_val)

        # compute adv, detach the adv and targ_val to avoid grad tracking
        adv, targ_val = vec_generalized_advantage_estimate(gamma, lmbda, data['val'], data['next']['val'], data['next']['rwd'], data['next']['done'])
        data.set('adv', adv.detach())
        data.set(('targ_val'), targ_val.detach())

        # store data into buffer for later random sampling
        buffer.extend(data)

        # start update
        smp = buffer.sample()

        # pol loss
        act_probs = pol_net(smp['obs'])
        act_distr = Categorical(act_probs)
        new_act_log_prob = act_distr.log_prob(smp['act'].reshape(-1)).reshape(-1, 1)
        r = torch.exp(new_act_log_prob - smp['act_log_prob'])
        clip_r = torch.clip(r, min=1-eps, max=1+eps)
        pol_loss = -torch.mean(torch.min(r * smp['adv'], clip_r * smp['adv']))
        
        # val loss
        new_val = val_net(smp['obs'])
        val_loss = torch.mean(torch.square(new_val - smp['targ_val']))

        loss = pol_loss + val_loss
        loss.backward(retain_graph=True)
        optim.step()
        optim.zero_grad()

        # clear the buffer
        buffer.empty()

    #end of training epoch
    iters = range(len(log['avg_rwd']))
    ax.clear()
    ax.errorbar(x=iters, y=log['avg_rwd'], yerr=log['rwd_std'], fmt='b.-', ecolor=(0, 0, 1, 0.1))
    #ax.plot(iters, log['avg_rwd'], 'b.-')
    ax.set_xlabel('iter')
    ax.set_ylabel('avg rwd')
    ax.grid()
    fig.canvas.flush_events()
plt.ioff()
plt.show()