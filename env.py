import torch
from poker import check_rank

# obs = [hand[0:2], flop[2:5], turn[5:6], river[6:7], pot[7:8], bets[8:14], chips[14,20], line[20:26]]
# act = [fold[0:1], call[1:2], bets[2:21], all-in[21:22]]
DEFAULT_DTYPE = torch.float32

def make_act_dict(odd_incr=0.05):
    odds = torch.arange(start=odd_incr, end=1, step=odd_incr, dtype=torch.float32)
    bet_sizes = odds/(1-odds)

    reserved_num_of_act = 4
    num_of_act = reserved_num_of_act + bet_sizes.size(dim=0)
    act_dict = {'num_of_act': num_of_act}
    for i in range(num_of_act):
        if i == 0:
            act_dict[i] = {'code': 'F', 'name': 'fold'}
        elif i == 1:
            act_dict[i] = {'code': 'X', 'name': 'check'}
        elif i == 2:
            act_dict[i] = {'code': 'C', 'name': 'call'}
        elif i == 3:
            act_dict[i] = {'code': 'A', 'name': 'all in'}
        else:
            act_dict[i] = {'code': 'R', 'name': f'bet {(bet_sizes[i-reserved_num_of_act]*100).round(decimals=0).to(torch.int64)}% of pot', 'bet_size': bet_sizes[i-reserved_num_of_act]}
    return act_dict

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


ACT_DICT = make_act_dict(odd_incr=0.05)
class GameTorchEnv():
    def __init__(self, num_of_player=2, act_dict=ACT_DICT, rotation=False, dtype=torch.float32, device='cpu'):
        ### dtype and working device of this game env
        self.dtype = dtype
        self.device = device

        ### obs
        self.max_num_of_player = 6
        self.num_of_obs = 11 + 3 * self.max_num_of_player
        # 2 for hand, 5 for board, 1 for position, 1 for eff_pot, 1 for add_to_call, 1 for add_to_min_raise
        # and 3x6 for player's bet, chip and encoded act line
        
        ### act
        bet_sizes = []
        for k in act_dict.keys():
            if k == 'num_of_act':
                self.num_of_act = act_dict['num_of_act']
            else:
                if act_dict[k]['code'] == 'R':
                    bet_size = act_dict[k]['bet_size'].to(device=device)
                    bet_sizes.append(bet_size.reshape(1))
                    act_dict[k]['bet_size'] = bet_size
        
        self.bet_sizes = torch.cat(bet_sizes, dim=0)
        self.act_dict = act_dict
        
        ### player idxes
        assert num_of_player in range(2, self.max_num_of_player + 1)
        self.num_of_player = num_of_player
        self.player_idxes = torch.arange(self.max_num_of_player, dtype=torch.int64, device=device)
        self.has_player = self.player_idxes < self.num_of_player
        
        self.rotation = rotation
        self.num_of_game = 0

    def reset(self):
        self.num_of_game += 1

        ### position idx
        self.sb_idx = (self.num_of_game - 1) % self.num_of_player if self.rotation else 0
        self.bb_idx = (self.sb_idx + 1) % self.num_of_player
        self.whose_turn = (self.sb_idx + 2) % self.num_of_player # BTN_idx for heads-up, UTG_idx otherwise
        
        ### hand and board cards
        deck = torch.randperm(52, dtype=torch.int64, device=self.device) + 1 # reserve 0 for not dealed card
        self.player_hand = deck[:self.max_num_of_player * 2].reshape(self.max_num_of_player, 2).sort(dim=-1).values
        self.flop = deck[12:15].sort().values
        self.turn = deck[15:16]
        self.river = deck[16:17]
        
        ### player chip, bet-to, and pot
        self.player_initial_chip = 100 * self.has_player.to(dtype=self.dtype)
        self.player_bet = torch.zeros(self.max_num_of_player, dtype=self.dtype, device=self.device)
        self.player_bet[self.sb_idx] = 0.5
        self.player_bet[self.bb_idx] = 1
        self.player_total_bet = self.player_bet.clone()
        self.player_chip = self.player_initial_chip - self.player_total_bet

        self.total_pot = self.player_total_bet.sum()

        ### encoded act line
        self.player_line = self.has_player.to(dtype=torch.int64)

        ### status of game
        self.over = False
        self.street = 0
        self.showdown = False
        
        ### call/raise limitation
        self.biggest_bet = 1 # limp to 1BB
        self.min_raise = 1 # raise to 2BB

        ### player's status
        self.who_fold, self.who_check, self.who_call, self.who_raise, self.who_all_in = torch.zeros((5, self.max_num_of_player), dtype=torch.bool, device=self.device)
        self.who_raise[self.bb_idx] = True

        # player's rwd and done
        self.player_rwd = torch.zeros((self.max_num_of_player, 1), dtype=self.dtype, device=self.device)
        self.player_done = torch.zeros((self.max_num_of_player, 1), dtype=torch.bool, device=self.device)

    def to_player_pov(self, i=None):
        i = self.whose_turn if i is None else i
        obs = torch.zeros((self.num_of_obs), dtype=self.dtype, device=self.device)

        ### hand card
        obs[0:2] = self.player_hand[i]
        
        ### board card
        if self.street > 0 or self.showdown:
            obs[2:5] = self.flop
        if self.street > 1 or self.showdown:
            obs[5:6] = self.turn
        if self.street > 2 or self.showdown:
            obs[6:7] = self.river

        eff_pot = self.count_eff_pot(i)
        my_pre_bet = self.player_bet[i].clone()
        add_to_call = self.biggest_bet - my_pre_bet
        add_to_min_raise = self.biggest_bet + self.min_raise - my_pre_bet
        obs[7:] = torch.cat([torch.tensor([i, eff_pot, add_to_call, add_to_min_raise]), self.player_bet, self.player_chip, self.player_line])

        valid_act = self.determine_valid_act(i)

        return obs, self.player_rwd[i], self.player_done[i], valid_act
    
    def count_eff_pot(self, i):
        my_total_bet = self.player_total_bet[i]
        who_cover_me = self.player_total_bet >= my_total_bet
        # effective pot = sum of the chips whose covered by me + num of players including myself whose chips cover me * my total bet
        eff_pot = self.player_total_bet[~who_cover_me].sum() + who_cover_me.sum() * my_total_bet 
        return eff_pot
    
    def determine_valid_act(self, i):
        eff_pot = self.count_eff_pot(i)
        my_pre_bet = self.player_bet[i].clone()
        my_chip = self.player_chip[i].clone()
        add_to_call = self.biggest_bet - my_pre_bet
        my_raise = eff_pot * self.bet_sizes
        add_to_min_raise = self.biggest_bet + self.min_raise - my_pre_bet

        can_fold = self.who_raise.any()
        can_check = ~can_fold
        can_call = my_chip > add_to_call
        valid_raise = (add_to_min_raise <= my_raise) & (my_raise < my_chip)
        valid_act = torch.cat([torch.tensor([can_fold, can_check, can_call, True]), valid_raise])
        return valid_act
    
    def next(self, act_key):
        ### look up the act in act_dict
        act_key = act_key.to(torch.int64).item() #to python int64
        assert act_key in self.act_dict
        act = self.act_dict[act_key]

        ### do act
        i = self.whose_turn
        if act['code'] == 'F':
            self.__fold(i)
        elif act['code'] == 'X':
            self.__check(i)
        elif act['code'] == 'C':
            add_to_call = self.biggest_bet - self.player_bet[i]
            self.__call(i, add_to_call)
        elif act['code'] == 'A':
            add_to_min_raise = self.biggest_bet + self.min_raise - self.player_bet[i]
            my_chip = self.player_chip[i].clone()
            self.__all_in(i, my_chip, add_to_min_raise)
        else: # act['code'] == 'R':
            eff_pot = self.count_eff_pot(i)
            my_raise = (eff_pot * act['bet_size']).round(decimals=1)
            self.__raise(i, my_raise)

        self.__determine_game_state(i)

    def __determine_game_state(self, i):
        all_fold_to_one = self.who_fold.sum() == (self.num_of_player - 1)
        all_check = (self.who_check|self.who_fold|self.who_all_in).sum() == self.num_of_player
        all_call_to_one = (self.who_raise|self.who_call|self.who_fold|self.who_all_in).sum() == self.num_of_player
        all_shove = self.who_all_in.sum() == self.num_of_player
        all_shove_to_one = (self.who_all_in.sum() == (self.num_of_player - 1)) and (self.player_bet[~self.who_all_in] == self.biggest_bet)
        if all_fold_to_one:
            self.over = True
        elif all_shove or all_shove_to_one:
            self.over = True
        elif all_check or all_call_to_one:
            if self.street == 3:
                self.over = True
            else:
                self.street += 1
                i = self.sb_idx
            self.player_bet = torch.zeros(self.max_num_of_player, dtype=self.dtype, device=self.device)
        else:
            i = (i + 1) % self.num_of_player
        while (self.who_fold|self.who_all_in|~self.has_player)[i]:
            i = (i + 1) % self.num_of_player
        self.whose_turn = i

    def __fold(self, i):
        # player_total_bet and player_chip have already updated in other __act(), so no need to update here
        self.player_bet[i] = 0 
        self.player_line[i] *= -1
        self.who_fold[i] = True
        # for folded player, his game is done
        self.player_rwd[i, 0] = - self.player_total_bet[i]
        self.player_done[i, 0] = True
    
    def __check(self, i):
        self.player_line[i] *= 2
        self.who_check[i] = True

    def __call(self, i, add_to_call):
        self.player_bet[i] += add_to_call
        self.player_total_bet[i] += add_to_call
        self.player_chip[i] -= add_to_call
        self.player_line[i] *= 2
        self.who_call[i] = True

    def __raise(self, i, my_raise):
        self.player_bet[i] += my_raise
        self.player_total_bet[i] += my_raise
        self.player_chip[i] -= my_raise
        self.player_line[i] = self.player_line[i] * 2 + 1
        self.who_call[self.who_call] = False
        self.who_check, self.who_call, self.who_raise = torch.zeros((3, self.max_num_of_player), dtype=torch.bool, device=self.device)
        self.who_raise[i] = True

    def __all_in(self, i, my_chip, add_to_min_raise):
        self.player_bet[i] += my_chip
        self.player_total_bet[i] += my_chip
        self.player_chip[i] = 0
        self.who_all_in[i] = True
        if my_chip < add_to_min_raise:
            self.player_line[i] *= 2
            self.who_call[i] = True
        else:
            self.player_line[i] = self.player_line[i] * 2 + 1
            self.who_check, self.who_call, self.who_raise = torch.zeros((3, self.max_num_of_player), dtype=torch.bool, device=self.device)
            self.who_raise[i] = True


    def __determine_winner(self):
        if self.showdown: # if all players raise all-in or it's post-river, showdown is true
            player_hand_ranks = torch.zeros(self.max_num_of_player, dtype=torch.int64, device=self.device)
            for player_idx, hand in enumerate(self.player_hands[~self.no_players]):
                hand_with_board = torch.cat((hand, self.flop, self.turn, self.river))
                card_7_choose_5 = torch.combinations(hand_with_board, r=5)
                player_hand_ranks[player_idx] = check_rank(card_7_choose_5, return_highest=True)

            sort_uniq_hand_ranks = player_hand_ranks.unique(sorted=True)
            for rank_idx in reversed(range(sort_uniq_hand_ranks.size(dim=0))):
                winner_idxes = self.player_idxes[(player_hand_ranks==sort_uniq_hand_ranks[rank_idx]) & ~self.player_folds]
                num_of_winner = winner_idxes.size(dim=0)
                
                winner_chips_in_settled_pot, sub_winner_idxes = self.player_chips_in_settled_pot[winner_idxes].sort()
                for i in sub_winner_idxes:
                    win_chip = 0
                    for j in self.player_idxes[~self.no_players]:
                        if winner_chips_in_settled_pot[i] < self.player_chips_in_settled_pot[j]:
                            win_chip_from_player_j = winner_chips_in_settled_pot[i].clone()
                            self.player_chips_in_settled_pot[j] -= winner_chips_in_settled_pot[i]
                        else:
                            win_chip_from_player_j = self.player_chips_in_settled_pot[j].clone()
                            self.player_chips_in_settled_pot[j] = 0
                        win_chip += win_chip_from_player_j
                    self.player_chips[winner_idxes] += win_chip / num_of_winner
        else:
            winner_idx = self.player_idxes[~self.player_folds & ~self.no_players]
            self.player_chips[winner_idx] += self.settled_pot
        self.player_chips_in_settled_pot = torch.zeros(self.max_num_of_player, dtype=self.dtype, device=self.device)
        self.settled_pot = 0

        self.player_rwds = (self.player_chips - self.player_initial_chips).reshape(self.max_num_of_player, 1, 1)
        self.player_dones = torch.ones((self.max_num_of_player, 1, 1), dtype=torch.bool, device=self.device)

if __name__ == '__main__':
    game = GameTorchEnv(num_of_player=3)    
    game.reset()
    while not game.over:
        i = game.whose_turn
        s = game.street
        o, r, d, va = game.to_player_pov()
        print(s, i, o)
        a = torch.randint(0, 23, torch.Size([1]))
        game.next(a)