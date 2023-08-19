import numpy as np
import torch
from tensordict.tensordict import TensorDict

card_code = list(range(53))
#print(f'\n card_code\n{card_code}\n')
#print([(i - 1) % 13 for i in card_code])
#print([(i - 1) // 13 for i in card_code])

val_map = {}
for i in range(13):
    if i==8:
        val_map[i] = 'T'
    elif i==9:
        val_map[i] = 'J'
    elif i==10:
        val_map[i] = 'Q'
    elif i==11:
        val_map[i] = 'K'
    elif i==12:
        val_map[i] = 'A'
    else:
        val_map[i] = str(i+2)
#print(f'\n val_map\n{val_map}\n')

suit_map = {0: 's', 1: 'h', 2: 'd', 3: 'c'}
#print(f'\n suit_map\n{suit_map}\n')

card_map = {}
for i in card_code:
    if i==0:
        card_map[i] = 'N'
    else:
        card_map[i] = val_map[(i - 1) % 13] + suit_map[(i - 1) // 13]
#print(f'\n card_map\n{card_map}\n')

def check_rank(card, return_highest=True):
    if card.ndim == 1:
        card = card.reshape(1, -1)
    
    val = (card - 1) % 13 + 2
    val = val.sort(dim=1, descending=True).values # sort the card val descendingly first for later check straight and rank of pairs
    suit = (card - 1) // 13
    rank = torch.zeros((card.size(dim=0), 6), dtype=card.dtype, device=card.device)
    RANK_ORDER = torch.tensor([15 ** 5, 15 ** 4, 15 ** 3, 15 ** 2, 15 ** 1, 1], dtype=torch.int32, device=card.device)
    SMALLEST_STRAIGHT = torch.tensor([14, 5, 4, 3, 2], dtype=card.dtype, device=card.device)

    for i, (v, s) in enumerate(zip(val, suit)):
        v, count = v.unique_consecutive(return_counts=True)
        if count.size(dim=0) == 5: # no any pair, maybe has straight and flush
            is_normal_straight = (v.diff() == -1).all(dim=0)
            is_smallest_straight = (v == SMALLEST_STRAIGHT).all(dim=0)
            is_straight = is_normal_straight | is_smallest_straight
            is_flush = (s == s[0]).all(dim=0)
            if is_straight:
                # determine the rank of straight
                if is_normal_straight:
                    rank[i, 1] = v[0]
                else: # is the smallest straight
                    rank[i, 1] = 5

                if is_flush: # straight flush
                    rank[i, 0] = 8
                else: # straight without flush (normal straight)
                    rank[i, 0] = 4
            elif is_flush:
                rank[i, 0] = 5
                rank[i, 1:6] = v
            else: # is only high card
                rank[i, 1:6] = v

        elif count.size(dim=0) == 4: # one pair
            rank[i, 0] = 1
            rank[i, 1] = v[count == 2]
        elif count.size(dim=0) == 3:  # two pair or set
            if 2 in count:  # if two pair
                rank[i, 0] = 2
                rank[i, 1:3] = v[count == 2]
                rank[i, 3] = v[count == 1]
            else: # is set
                rank[i, 0] = 3
                rank[i, 1] = v[count == 3]
        else: # full house or four of kind
            if 2 in count: # full house
                rank[i, 0] = 6
                rank[i, 1] = v[count == 3]
                rank[i, 2] = v[count == 2]
            else: # is four of kind
                rank[i, 0] = 7
                rank[i, 1] = v[count == 4]

    rank = (rank * RANK_ORDER).sum(dim=1)
    if return_highest:
        return rank.max(dim=0).values
    else:
        return rank