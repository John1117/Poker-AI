# %%
import numpy as np
import torch
from policy import NeuralNetworkPolicy
from network import PolicyNetwork
from torch import nn
from torch.optim import Adam
from torch.distributions.categorical import Categorical
from collections import defaultdict
import matplotlib.pyplot as plt



# %%
class RangeAllInPolicy():
    def __init__(self, n_act=22, dtype=np.float32):
        self.n_act = n_act
        self.dtype = dtype

    def __call__(self, obs=None, valid_act_filter=None):
        act_probs = np.zeros((len(obs), self.n_act), dtype=self.dtype)
        hand = obs[:, 0:2]
        value = (hand - 1) % 13 + 2
        suit = (hand - 1) // 13

        for i in range(len(obs)):
            if suit[i, 0] == suit[i, 1] or value[i, 0] == value[i, 1]:
                act_probs[i, 2] = 1
            else:
                act_probs[i, 0] = 1
        return act_probs
    
base_obs = np.array([[
    0, 0, 
    0, 0, 0, 0, 0, 
    1, 
    101, 
    100, 1, 0, 0, 0, 0, 
    0, 99, 0, 0, 0, 0, 
    3, 1, 0, 0, 0, 0
]], dtype=np.float32)

n = 100000
obs = np.repeat(base_obs, n, axis=0)

hands = np.zeros((n, 2), dtype=np.float32)
for i in range(n):
    deck = np.random.permutation(52) + 1
    hands[i] = np.sort(deck[0:2])
obs[:, 0:2] = hands

range_all_in_policy = RangeAllInPolicy()
act_probs = range_all_in_policy(obs)
print(act_probs[:, 2].sum())
print(15/51)


# %%
n_data = 10000
obs = np.repeat(base_obs, n_data, axis=0)

hands = np.zeros((n_data, 2), dtype=np.float32)
for i in range(n_data):
    deck = np.random.permutation(52) + 1
    hands[i] = np.sort(deck[0:2])
obs[:, 0:2] = hands

act_probs = range_all_in_policy(obs)
act_probs = torch.tensor(act_probs)
act_distr = Categorical(probs=act_probs)
act = act_distr.sample()


# %%
policy_network = PolicyNetwork(hidden_arch=(32,))


# %%
n_epoch = 50
batch_size = 64
# loss_fn = nn.CrossEntropyLoss()
loss_fn = nn.MSELoss()

lr = 1e-4
optimizer = Adam(policy_network.parameters(), lr=lr)

log = defaultdict(list)
policy_network.train()
for epoch in range(n_epoch):
    n_batch = 0
    total_loss = 0
    for batch_obs, batch_act_probs in zip(np.array_split(obs, len(obs) // batch_size), np.array_split(act_probs, len(obs) // batch_size)):
        
        n_batch += 1
        batch_obs = torch.tensor(batch_obs, requires_grad=True)
        pred_act_probs = policy_network(batch_obs)

        loss = loss_fn(pred_act_probs, batch_act_probs)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()

    print(epoch, total_loss/n_batch)
    log['loss'].append(total_loss/n_batch)

policy_network.eval()
iters = np.arange(len(log['loss']))
plt.plot(iters, log['loss'], 'b-')
plt.show()




# %%


pred_act_probs = policy_network(torch.tensor(obs)).detach().numpy()
x = 2005
m = 10
print(act_probs[x:x+m,0:3])
print(pred_act_probs[x:x+m,0:3])



# %%
