import numpy as np
import torch
from TicTacToe import TicTacToe
from AlphaMCTS import AlphaMCTS
from model import ResNet
from ConnectFour import ConnectFour

def main():
    game_name = 'ConnectFour'
    player = 1
    args = {
        'c': 2, 
        'num_searches': 1000,
        'dirichlet_epsilon': 0.25,
        'dirichlet_alpha': 0.3
    }
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    if game_name == 'TicTacToe':
        game = TicTacToe()
        model = ResNet(game, 4, 64, device)
        model.load_state_dict(torch.load('checkpoints/tictactoe/model_2_TicTacToe.pt', map_location=device))
    elif game_name == 'ConnectFour':
        game = ConnectFour()
        model = ResNet(game, 9, 128, device) # connectfour
        model.load_state_dict(torch.load('checkpoints/connectfour/model_7_ConnectFour.pt', map_location=device))
    model.eval()
    mcts = AlphaMCTS(game, args, model)
    state = game.get_initial_state()

    while True:
        print(state)
        if player == 1:
            valid_moves = game.get_valid_moves(state)
            print("Valid moves:", [i for i in range(game.action_size) if valid_moves[i] == 1])

            action = int(input(f"{player}:"))
            if valid_moves[action] == 0:
                print("action not valid")
                continue
        else:
            neutral_state = game.change_game_perspective(state, player)
            mcts_probabilities = mcts.search(neutral_state)
            action = np.argmax(mcts_probabilities)


        state = game.get_next_state(state, action, player)
        value, _terminated = game.get_value_and_terminated(state, action)
        if _terminated:
            print(state)
            if value == 1: print(f"Player {player} wins!")
            else: print("It's a draw!")
            break

        player = game.get_opponent(player)  # Switch to the opponent player


if __name__ == "__main__":
    main()