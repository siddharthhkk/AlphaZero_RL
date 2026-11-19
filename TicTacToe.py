import numpy as np
from mcts import MCTS

class TicTacToe:
    def __init__(self):
        self.row_count = 3
        self.column_count = 3
        self.action_size = self.row_count * self.column_count

    def __repr__(self):
        return "TicTacToe"

    def get_initial_state(self):
        return np.zeros((self.row_count, self.column_count), dtype=int)

    def get_next_state(self, state, action, player): # action is an integer from 0 to 8
        row = action // self.column_count  # Calculate the row index 6 / 3 = 1
        col = action % self.column_count  # Calculate the column index 6 % 3 = 0 so action 6 corresponds to (1, 0)
        state[row, col] = player
        return state

    def get_valid_moves(self, state):
        return (state.reshape(-1) == 0).astype(np.uint8)

    def check_win(self, state, action):
        if action == None: return False
        row = action // self.column_count
        col = action % self.column_count
        player = state[row, col]

        # Check the row
        if np.sum(state[row, :]) == player * self.column_count:
            return True
        elif np.sum(state[:, col]) == player * self.row_count:
            return True
        elif np.sum(np.diag(state)) == player * self.row_count:
            return True
        elif np.sum(np.diag(np.flip(state, axis=0))) == player * self.row_count:
            return True
        else:
            return False
            
    def get_value_and_terminated(self, state, action):
        if self.check_win(state, action):
            return 1, True  # Current player wins
        elif np.sum(self.get_valid_moves(state)) == 0:
            return 0, True  # Draw
        return 0, False  # Game continues

    def get_opponent(self, player):
        return -player # returns the opponent player (-1 for player 1 and 1 for player -1)
    def get_opponent_value(self, value): 
        return -value

    def change_game_perspective(self, state, player):
        return state * player  # Perspective of the current player (1 for player 1, -1 for player -1)

    def get_encoded_state(self, state):
        encoded_state =  np.stack(
            (state == -1, state == 0, state == 1)
        ).astype(np.float32)

        if len(state.shape) == 3:
            encoded_state = np.swapaxes(encoded_state, 0, 1)  # for parallel training
        return encoded_state



#mcts main.py
def main():
    game = TicTacToe()
    player = 1

    mcts = MCTS(game, {'c': 1.41, 'num_searches': 1000})
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