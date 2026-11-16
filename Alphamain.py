import numpy as np
from TicTacToe import TicTacToe
from AlphaMCTS import AlphaMCTS
from model import ResNet

def main():
    game = TicTacToe()
    player = 1
    args = {
        'c': 2, 
        'num_searches': 1000
    }

    model = ResNet(game, 4, 64)
    model.eval()
    mcts = AlphaMCTS(game, args, model)
    state = game.get_initial_state()

    while True:
        print(state)
        if player == 1:
            valid_moves = game.get_valid_moves(state)
            print("Valid moves:", [i for i in range(game.action_size) if valid_moves[i] == 1])

            action = int(input("Enter your move (0-8): "))
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