import numpy as np


def get_discounted_return(rewards, dones, discount=0.99):

    discounted_returns = rewards.copy()
    for i in range(len(rewards)-2, 0, -1):
        discounted_returns[i] = rewards[i] + discounted_returns[i+1] * discount * (1 - dones[i])

    return discounted_returns


def get_advantage(rewards, next_rewards, dones, next_dones, values, next_values, discount=0.99, temporal_diff_weight=0.95, use_generalized_advantage=True):
    
    values = values * (1 - dones) + rewards * dones
    
    if use_generalized_advantage:
        
        next_values = next_values * (1 - next_dones) + next_rewards * next_dones
        temporal_diffs = rewards + discount * next_values * (1 - dones) - values
        weighted_discount = discount * temporal_diff_weight

        generalized_advantages = temporal_diffs.copy()
        for i in range(len(rewards)-2, 0, -1):
            generalized_advantages[i] = temporal_diffs[i] + generalized_advantages[i+1] * weighted_discount * (1 - dones[i])
        return generalized_advantages
    
    else:
        discounted_returns = get_discounted_return(rewards, dones, discount)
        return discounted_returns - values