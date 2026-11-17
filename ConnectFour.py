import numpy as np

class ConnectFour:
    def __init__(self):
        self.row_count = 6
        self.column_count = 7
        self.action_size = self.column_count
        self.in_a_row = 4

    def __repr__(self):
        return "ConnectFour"

    def get_initial_state(self):
        return np.zeros((self.row_count, self.column_count), dtype=int)

    def get_next_state(self, state, action, player): # action is an integer from 0 to 8
        row = np.max(np.where(state[:, action] == 0))
        state[row, action] = player    #as column is action!
        return state

    def get_valid_moves(self, state):
        return (state[0] == 0).astype(np.uint8)

    def check_win(self, state, action):
        if action == None: return False
        row = np.min(np.where(state[:, action] != 0))
        player = state[row][action]

        def count(offset_row, offset_col):
            for i in range(1, self.in_a_row):
                r = row + offset_row * i
                c = action + offset_col * i
                if(r < 0 or r >= self.row_count or c < 0 or c >= self.column_count or state[r][c] != player): return i - 1
            return self.in_a_row - 1

        # Check the row
        if count(1, 0) >= self.in_a_row - 1:
            return True
        elif (count(0, 1) + count(0, -1)) >= self.in_a_row - 1:
            return True
        elif (count(1, 1) + count(-1, -1)) >= self.in_a_row - 1:
            return True
        elif (count(1, -1) + count(-1, 1)) >= self.in_a_row - 1:
            return True
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
        return np.stack(
            (state == -1, state == 0, state == 1)
        ).astype(np.float32)