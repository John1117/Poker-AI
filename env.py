# %%
import torch as tc
from poker import check_rank


# %%
def make_act_dict(odd_incr=0.05, dtype=tc.float32, device='cpu'):
    odds = tc.arange(start=odd_incr, end=1, step=odd_incr, dtype=dtype, device=device)
    bet_sizes = odds / (1 - odds)

    n_default_act = 4
    n_bet = bet_sizes.size(dim=0)
    n_act = n_default_act + n_bet
    act_dict = {'n_act': n_act}
    act_dict['bet_sizes'] = bet_sizes

    act_dict[0] = {'code': 'F', 'name': 'fold'}
    act_dict[1] = {'code': 'X', 'name': 'check'}
    act_dict[2] = {'code': 'C', 'name': 'call'}
    act_dict[3] = {'code': 'A', 'name': 'all in'}
    for i in range(n_default_act, n_act):
        act_dict[i] = {'code': 'B', 'name': f'bet {(bet_sizes[i-n_default_act]*100).round(decimals=0).to(tc.int64)}% of pot', 'bet_size': bet_sizes[i-n_default_act]}
    return act_dict
act_dict = make_act_dict(0.05)
act_dict


# %%
class GameTorchEnv():
    def __init__(self, n_player=2, odd_incr=0.05, rot=False, dtype=tc.float32, device='cpu'):

        self.dtype = dtype
        self.device = device

        # obs
        self.max_n_player = 6
        self.n_obs = 9 + 3 * self.max_n_player # 2 for hand, 5 for board, 1 for position, 1 for eff_pot and 3x6 for player's bet, chip and encoded act line
        
        # act
        self.act_dict = make_act_dict(odd_incr, dtype, device)
        self.n_act = self.act_dict['n_act']
        self.bet_sizes = self.act_dict['bet_sizes']
        
        # player idxs
        assert n_player in range(2, self.max_n_player + 1)
        self.n_player = n_player
        self.player_idxs = tc.arange(self.max_n_player, dtype=tc.int64, device=device)
        self.occupied = self.player_idxs < self.n_player
        
        self.rot = rot
        self.n_game = 0
        self.SB_idx = -1

    def reset(self, player_init_chips=None):

        self.n_game += 1

        # position
        self.SB_idx = (self.SB_idx + 1 * self.rot) % self.n_player
        self.BB_idx = (self.SB_idx + 1) % self.n_player
        self.whose_turn = (self.SB_idx + 2) % self.n_player # BTN_idx for heads-up, UTG_idx otherwise
        
        # hand and board cards
        deck = tc.randperm(52, dtype=tc.int64, device=self.device) + 1 # reserve 0 for no card
        self.player_hands = deck[:self.max_n_player * 2].reshape(self.max_n_player, 2).sort(dim=-1).values # sort the hand to make it unique
        self.flop = deck[12:15].sort().values # sort the flop to make it unique
        self.turn = deck[15:16]
        self.river = deck[16:17]
        
        # player chip, bet, total chip put in pot
        self.player_init_chips = player_init_chips
        if player_init_chips is not None:
            self.player_init_chips = 100 * self.occupied.to(dtype=self.dtype) # init to 100BB
        self.player_bets = tc.zeros(self.max_n_player, dtype=self.dtype, device=self.device)
        self.player_bets[self.SB_idx] = 0.5
        self.player_bets[self.BB_idx] = 1
        self.player_puts = self.player_bets.clone()
        self.player_chips = self.player_init_chips - self.player_puts


        # encoded act line
        self.player_lines = self.occupied.to(dtype=tc.int64)

        # game state
        self.street = 0
        self.showdown = False
        self.is_over = False
        
        # call/raise limitation
        self.max_bet = 1 # limp to 1BB
        self.min_raise = 2 # raise to 2BB

        # player state, the first 3 are temporary status in a betting round, the last 2 are permanent status in a game
        self.who_check, self.who_call, self.who_raise, self.who_fold, self.who_shove = tc.zeros((5, self.max_n_player), dtype=tc.bool, device=self.device)
        self.who_raise[self.BB_idx] = True

        # player rwd and done
        self.player_rwds = tc.zeros((self.max_n_player, 1), dtype=self.dtype, device=self.device)
        self.player_dones = tc.zeros((self.max_n_player, 1), dtype=tc.bool, device=self.device)

    
    def next(self, act_key):
        # look up the act in act_dict
        act_key = act_key.to(tc.int64).item() #to python int64
        assert act_key in self.act_dict
        act = self.act_dict[act_key]

        # do act
        if act['code'] == 'F':
            self.__fold()
        elif act['code'] == 'X':
            self.__check()
        elif act['code'] == 'C':
            self.__call()
        elif act['code'] == 'A':
            self.__shove()
        else: # act['code'] == 'B':
            self.__bet(act['bet_size'])

        is_over = self.__check_game_state()
        if is_over:
            self.__check_winner()
            self.reset()

    def to_player_pov(self, i=None):
        i = self.whose_turn if i is None else i
        obs = tc.zeros((self.n_obs), dtype=self.dtype, device=self.device)

        # hand card
        obs[0:2] = self.player_hands[i]
        
        # board card
        if self.street > 0 or self.showdown:
            obs[2:5] = self.flop
        if self.street > 1 or self.showdown:
            obs[5:6] = self.turn
        if self.street > 2 or self.showdown:
            obs[6:7] = self.river

        eff_pot = self.count_eff_pot(i)
        obs[7:] = tc.cat([tc.tensor([i, eff_pot]), self.player_bets, self.player_chips, self.player_lines])
        valid_act = self.check_valid_act(i)

        return obs, self.player_rwds[i], self.player_dones[i], valid_act
    
    def count_eff_pot(self, i):
        my_total_bet = self.player_puts[i]
        who_cover_me = self.player_puts >= my_total_bet
        # effective pot = sum of the chips whose covered by me + num of players including myself whose chips cover me * my total bet
        eff_pot = self.player_puts[~who_cover_me].sum() + who_cover_me.sum() * my_total_bet 
        return eff_pot
    
    def check_valid_act(self, i):
        eff_pot = self.count_eff_pot(i)
        my_pre_bet = self.player_bets[i] #.clone()
        my_chip = self.player_chips[i] #.clone()
        alive_opps = ~self.who_fold
        alive_opps[i] = False
        max_opp_chip = self.player_chips[alive_opps].max()
        add_to_call = self.max_bet - my_pre_bet
        my_bets = eff_pot * self.bet_sizes
        add_to_min_raise = self.min_raise - my_pre_bet

        can_fold = self.who_raise.any()
        can_check = ~can_fold
        can_call = my_chip > add_to_call
        valid_bets = (add_to_min_raise <= my_bets) & (my_bets < my_chip) & (my_bets < max_opp_chip)
        valid_act = tc.cat([tc.tensor([can_fold, can_check, can_call, True]), valid_bets]) # can_shove is always True
        return valid_act


    def __fold(self):
        i = self.whose_turn
        self.player_bets[i] = 0
        self.player_lines[i] *= -1
        self.who_fold[i] = True

        # for folded player, his game is done
        self.player_rwds[i, 0] = - self.player_puts[i]
        self.player_dones[i, 0] = True
    
    def __check(self):
        i = self.whose_turn
        self.player_lines[i] *= 2
        self.who_check[i] = True

    def __call(self):
        i = self.whose_turn
        add_to_call = self.max_bet - self.player_bets[i]
        self.player_bets[i] += add_to_call
        self.player_puts[i] += add_to_call
        self.player_chips[i] -= add_to_call
        self.player_lines[i] *= 2
        self.who_call[i] = True

    def __bet(self, bet_size):
        i = self.whose_turn
        eff_pot = self.count_eff_pot(i)
        my_bet = (eff_pot * bet_size).round(decimals=1)
        self.player_bets[i] += my_bet
        self.player_puts[i] += my_bet
        self.player_chips[i] -= my_bet
        self.player_lines[i] = self.player_lines[i] * 2 + 1
        self.who_check, self.who_call, self.who_raise = tc.zeros((3, self.max_n_player), dtype=tc.bool, device=self.device) # reset the temporary bet status
        self.who_raise[i] = True

    def __shove(self):
        i = self.whose_turn
        my_chip = self.player_chips[i].clone()
        self.player_bets[i] += my_chip
        self.player_puts[i] += my_chip
        self.player_chips[i] = 0
        self.who_shove[i] = True

        add_to_min_raise = self.min_raise - self.player_bets[i]
        if my_chip < add_to_min_raise:
            self.player_lines[i] *= 2
        else:
            self.player_lines[i] = self.player_lines[i] * 2 + 1
            self.who_check, self.who_call, self.who_raise = tc.zeros((3, self.max_n_player), dtype=tc.bool, device=self.device)
            self.who_raise[i] = True

    def __check_game_state(self):
        i = self.whose_turn
        who_wait = self.who_fold | self.who_shove
        all_fold = self.who_fold.sum() == (self.n_player - 1)
        all_check = (self.who_check | who_wait).sum() == self.n_player
        all_call = (self.who_raise | self.who_call | who_wait).sum() == self.n_player
        all_shove = (who_wait).sum() == self.n_player

        if all_fold:
            self.is_over = True
            return self.is_over
        elif all_shove:
            self.showdown = True
            self.is_over = True
            return self.is_over
        elif all_check or all_call:
            if self.street == 3:
                self.showdown = True
                self.is_over = True
                return self.is_over
            else:
                self.street += 1
                i = self.SB_idx
            self.player_bets = tc.zeros(self.max_n_player, dtype=self.dtype, device=self.device)
        else:
            i = (i + 1) % self.n_player

        # move the turn if one is fold, shove or no player
        while (who_wait | ~self.occupied)[i]:
            i = (i + 1) % self.n_player
        self.whose_turn = i
        return self.is_over


    def __check_winner(self):
        if self.showdown: # if all players raise all-in or it's post-river, showdown is true
            player_ranks = tc.zeros(self.max_n_player, dtype=tc.int64, device=self.device)
            for player_idx, hand in enumerate(self.player_hands[self.occupied]):
                hand_board_combo = tc.cat((hand, self.flop, self.turn, self.river))
                card_7c5 = tc.combinations(hand_board_combo, r=5)
                player_ranks[player_idx] = check_rank(card_7c5, return_highest=True)

            sort_ranks = player_ranks.unique(sorted=True, return_inverse=True)
            r = 0
            while self.player_puts.any():
                winner_idxs = self.player_idxs[(player_ranks==sort_ranks[r]) & ~self.who_fold]
                n_chop = winner_idxs.size(dim=0)

                winner_total_bets, chop_idxs = self.player_puts[winner_idxs].sort()
                for i in chop_idxs:
                    gain = 0
                    for j in self.player_idxs[self.occupied]:
                        if winner_total_bets[i] < self.player_puts[j]:
                            gain_from_player_j = winner_total_bets[i].clone()
                            self.player_puts[j] -= winner_total_bets[i]
                        else:
                            gain_from_player_j = self.player_puts[j].clone()
                            self.player_puts[j] = 0
                        gain += gain_from_player_j
                    self.player_chips[winner_idxs] += gain / n_chop
                r += 1
        else: # if all fold to one
            winner_idx = self.player_idxs[~self.who_fold & self.occupied]
            self.player_chips[winner_idx] += self.player_puts.sum()

        self.player_puts = tc.zeros(self.max_n_player, dtype=self.dtype, device=self.device)

        self.player_rwds = (self.player_chips - self.player_init_chips).reshape(self.max_n_player, 1, 1)
        self.player_dones = tc.ones((self.max_n_player, 1, 1), dtype=tc.bool, device=self.device)


game = GameTorchEnv(n_player=3)    
game.reset()
print(game.player_hands)



# %%
def encode_act_str(act_str='I'):
    assert isinstance(act_str, str)
    
    act_bin = '1' # default inaction
    for a in act_str:
        if a == 'C': # check, call, limp
            act_bin += '0'
        elif a == 'B': # open, bet, raise, all-in
            act_bin += '1'
        elif a == 'N': # no player
            act_bin = '0'
    
    code = int(act_bin, 2)
    if act_str[-1] == 'F':
        code *= -1
    return code