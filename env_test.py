# %%
import torch as tc
from env import GameTorchEnv

# %%
def obs_code_to_text(obs):
    print('=================================')
    print('')
    print(f'hands = {obs[0:2]}')
    print(f'board = {obs[2:7]}')
    print(f'position = {obs[7]}')
    print(f'eff pot = {obs[8]}')
    print(f'bet this street = {obs[9:15]}')
    print(f'all remain chips = {obs[15:21]}')
    print(f'all action line = {obs[21:]}')
    print('')

# %%
if __name__ == '__main__':

    n_player = 3
    init_chip = 100

    game = GameTorchEnv(n_player=n_player)
    game.reset(init_chip)

    print(game.player_puts)
    print(game.player_bets)
    print(game.player_chips)
    print(game.act_dict)
    

    # %%
    for i in range(n_player):
        obs, rwd, dones, valid = game.to_player_pov(i)
        print('=================================')
        print(game.check_valid_act(i))
        obs_code_to_text(obs)
        
    


    
# %%
