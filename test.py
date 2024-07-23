# %%
import numpy as np
import pandas as pd
from env import Game
from network import PolicyNetwork, ValueNetwork
from policy import AlwaysAllInPolicy, NeuralNetworkPolicy, UniformPolicy, AlwaysCallPolicy, NeuralNetworkEvaluator
from collect import collect_data, Buffer
import matplotlib.pyplot as plt
from collections import defaultdict
from loss import get_discounted_return, get_advantage

import torch
from torch.optim import Adam
from torch.distributions.categorical import Categorical


# %%
game = Game(n_player=2)

always_all_in_policy = AlwaysAllInPolicy(n_act=game.n_act)

hidden_arch = (64, 64)
policy_network = PolicyNetwork(n_obs=game.n_obs, n_act=game.n_act, hidden_arch=hidden_arch)
value_network = ValueNetwork(n_obs=game.n_obs, hidden_arch=hidden_arch)

nn_policy = NeuralNetworkPolicy(n_act=game.n_act, model=policy_network)
nn_evaluator = NeuralNetworkEvaluator(model=value_network)

policies = [always_all_in_policy, nn_policy]

epsilon = 0.2
gamma = 1.0
lmbda = 1.0
discount = 0.99
temporal_diff_weight= 0.95
use_generalized_advantage = True

n_iter = 100
collect_index = 1
n_collect = 100
n_epoch = 10
batch_size = n_collect

buffer = Buffer(max_size=10000)

lr = 3e-4
optim = Adam(params=[{'params': policy_network.parameters()}, {'params': value_network.parameters()}], lr=lr)

log = defaultdict(list)
plt.ion()
fig = plt.figure()
ax = fig.add_subplot()

for i in range(n_iter):

    data = collect_data(game=game, policies=policies, collect_index=collect_index, n_collect=n_collect)

    rewards = data['reward']
    dones = data['done']
    next_rewards = data['next_reward']
    next_dones = data['next_done']
    discounted_returns = get_discounted_return(next_rewards, next_dones, discount=0.99)
    data['discounted_return'] = discounted_returns
    
    final_rewards = next_rewards[next_dones]
    
    vpip = (data['act'] == 2).sum() / n_collect * 100

    log['avg_reward'].append(final_rewards.mean())
    log['reward_std'].append(final_rewards.std())
    log['vpip'].append(vpip)
    
    for e in range(n_epoch):

        values = nn_evaluator(data['obs'])
        next_values = nn_evaluator(data['next_obs'])
        sampled_advantages = get_advantage(rewards, next_rewards, dones, next_dones, values, next_values, discount=discount, temporal_diff_weight=temporal_diff_weight, use_generalized_advantage=use_generalized_advantage)

        data['value'] = values
        data['next_value'] = next_values
        data['advantage'] = sampled_advantages

        buffer.add(data)

        sampled_data = buffer.sample(batch_size)

        sampled_obss = sampled_data['obs']
        sampled_acts = torch.tensor(sampled_data['act'])
        sampled_act_probs = torch.tensor(sampled_data['act_prob'])
        
        new_act_probss = nn_policy(sampled_obss, return_torch_tensor=True)
        new_act_probs = torch.index_select(input=new_act_probss, dim=1, index=sampled_acts.reshape(-1))
        
        sampled_advantages = torch.tensor(sampled_data['advantage'])
        r = torch.exp(torch.log(new_act_probs) - torch.log(sampled_act_probs))
        clip_r = torch.clip(r, 1-epsilon, 1+epsilon)
        policy_loss = -torch.mean(torch.min(r * sampled_advantages, clip_r * sampled_advantages))

        sampled_values = torch.tensor(sampled_data['value'])
        new_values = nn_evaluator(sampled_obss, return_torch_tensor=True)
        value_loss = torch.mean(torch.square(new_values - sampled_values))

        loss = policy_loss + value_loss
        loss.backward(retain_graph=True)
        optim.step()
        optim.zero_grad()
        
        buffer.clear()

    # print(nn_policy.model.model[0].weight[0:3:2])
    # print(nn_policy.model.model[0].bias[0:3:2])

    #end of training epoch
    iters = range(len(log['avg_reward']))
    ax.clear()
    ax.errorbar(x=iters, y=log['avg_reward'], yerr=log['reward_std'], fmt='b-', ecolor=(0, 0, 1, 0.1), label='Avg. reward')

    ax.plot(iters, log['vpip'], 'r-', label='VPIP')
    ax.set_xlabel('iter')
    # ax.set_ylabel('avg rwd')
    # ax.set_ylim(-5, 5)
    ax.grid()
    ax.legend()
    fig.canvas.flush_events()
plt.ioff()
plt.show()

# %%
