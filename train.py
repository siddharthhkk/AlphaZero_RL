import torch
import os
import numpy as np
import random
from tqdm import trange
from AlphaMCTS import AlphaMCTS
from TicTacToe import TicTacToe
from model import ResNet
from ConnectFour import ConnectFour


class AlphaZero:
    def __init__(self, model, optimizer, game, args, game_name):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = AlphaMCTS(game, args, model)

        base_dir = os.path.dirname(os.path.abspath(__file__))
        if game_name == 'TicTacToe':
            self.save_dir = os.path.join(base_dir, "checkpoints/tictactoe")
        elif game_name == 'ConnectFour':
            self.save_dir = os.path.join(base_dir, "checkpoints/connectfour")
        os.makedirs(self.save_dir, exist_ok=True)

    def selfPlay(self):
        memory = []
        player = 1
        state = self.game.get_initial_state()

        while True:
            neutral_state = self.game.change_game_perspective(state, player)
            action_probs = self.mcts.search(neutral_state)

            memory.append((neutral_state, action_probs, player))
            # temperature = squeezes the probs much closer, so we do more exploration (instead of max prob also pick second max value sometimes)
            temperature_action_probs = action_probs ** (1 / self.args['temperature'])
            temperature_action_probs /= np.sum(temperature_action_probs)
            action = np.random.choice(self.game.action_size, p=temperature_action_probs)
            state = self.game.get_next_state(state, action, player)

            value, is_terminal = self.game.get_value_and_terminated(state, action)

            if is_terminal:
                returnMemory = []

                for hist_neutral_state, hist_action_probs, hist_player in memory:
                    hist_outcome = value if hist_player == player else self.game.get_opponent_value(value)
                    returnMemory.append((
                        self.game.get_encoded_state(hist_neutral_state),
                        hist_action_probs, hist_outcome
                    ))
                return returnMemory

            player = self.game.get_opponent(player)

    def train(self, memory):
        random.shuffle(memory)
        total_loss = 0.0
        num_batches = 0
        for batchidx in range(0, len(memory), self.args['batch_size']):
            sample = memory[batchidx: min(len(memory) - 1, batchidx + self.args['batch_size'])]
            state, policy_targets, value_targets = zip(*sample)

            state, policy_targets, value_targets = np.array(state), np.array(policy_targets), np.array(value_targets).reshape(-1,1)
            state = torch.tensor(state, dtype=torch.float32, device=self.model.device)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=self.model.device)
            value_targets = torch.tensor(value_targets, dtype=torch.float32, device=self.model.device)

            out_policy, out_value = self.model(state)
            policy_loss = torch.nn.functional.cross_entropy(out_policy, policy_targets)
            value_loss = torch.nn.functional.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1
        print(f"Average Loss: {total_loss / num_batches:.4f}")


    def learn(self):
        for iter in range(self.args['num_iterations']):
            memory = []

            self.model.eval()
            for selfPlay_iter in trange(self.args['num_selfPlay_iterations']):  #trange for progress bar
                memory += self.selfPlay()

            self.model.train()
            for epoch in trange(self.args['num_epochs']):
                self.train(memory)

            torch.save(self.model.state_dict(), os.path.join(self.save_dir, f"model_{iter}_{self.game}.pt"))
            torch.save(self.optimizer.state_dict(), os.path.join(self.save_dir, f"optimizer_{iter}_{self.game}.pt"))



game_name = 'TicTacToe'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if game_name == 'TicTacToe':
    game = TicTacToe()
    model = ResNet(game, 4, 64, device)
    args = {
        'c': 2,
        'num_searches': 120,
        'num_iterations': 3,
        'num_selfPlay_iterations': 500,
        'num_epochs': 4,
        'batch_size': 64,
        'temperature': 1.25,  #ranges from 0-3!
        'dirichlet_epsilon': 0.25,
        'dirichlet_alpha': 0.3
    }

elif game_name == 'ConnectFour':
    game = ConnectFour()
    model = ResNet(game, 9, 128, device)
    args = {
        'c': 2,
        'num_searches': 600,
        'num_iterations': 8,
        'num_selfPlay_iterations': 500,
        'num_epochs': 4,
        'batch_size': 128,
        'temperature': 1.25,
        'dirichlet_epsilon': 0.25,
        'dirichlet_alpha': 0.3
    }

opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

alphaZero = AlphaZero(model, optimizer=opt, game=game, args=args, game_name=game_name)
alphaZero.learn()