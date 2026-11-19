import torch
import os
import numpy as np
import random
from tqdm import tqdm, trange
from AlphaMCTS import AlphaNode
from TicTacToe import TicTacToe
from ConnectFour import ConnectFour
from model import ResNet


class AlphaParallelMCTS:
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model

    @torch.no_grad() #as we just want to get the prediction
    def search(self, states, self_play_games):
        policy, _ = self.model(
            torch.tensor(self.game.get_encoded_state(states), device=self.model.device)
        )
        policy = torch.softmax(policy, axis=1).cpu().numpy()
        #adding noise
        policy = (1 - self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon'] * np.random.dirichlet(
            [self.args['dirichlet_alpha']] * self.game.action_size, size=policy.shape[0]
        )

        for i, spg in enumerate(self_play_games):
            spg_policy = policy[i]
            valid_moves = self.game.get_valid_moves(states[i])
            spg_policy *= valid_moves
            spg_policy /= np.sum(spg_policy)

            spg.root = AlphaNode(self.game, self.args, states[i], visit_count=1)
            spg.root.expand(spg_policy)

        
        for search in range(self.args['num_searches']):
            for spg in self_play_games:
                spg.node = None
                node = spg.root
                # Selection
                while node.is_fully_expanded():
                    node = node.select()
                # Expansion
                value, is_terminal = self.game.get_value_and_terminated(node.state, node.action_taken)
                value = self.game.get_opponent_value(value)

                if is_terminal:
                    node.backprop(value)
                else:
                    spg.node = node

                expandable_spGames = [mappingidx for mappingidx in range(len(self_play_games)) if self_play_games[mappingidx].node is not None]

                if len(expandable_spGames) > 0:
                    states = np.stack([self_play_games[mappingidx].node.state for mappingidx in expandable_spGames])
                    # for prediction from the trained model!
                    policy, value = self.model(
                        torch.tensor(self.game.get_encoded_state(states), device=self.model.device)
                    )
                    policy = torch.softmax(policy, axis=1).cpu().numpy()


                for i, mappingidx in enumerate(expandable_spGames):
                    node = self_play_games[mappingidx].node

                    spg_policy, spg_value = policy[i], value[i]
                    valid_moves = self.game.get_valid_moves(node.state)
                    spg_policy *= valid_moves
                    spg_policy /= np.sum(spg_policy)

                    node.expand(spg_policy)
                    node.backprop(spg_value)




class AlphaParallelZero:
    def __init__(self, model, optimizer, game, args, game_name):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = AlphaParallelMCTS(game, args, model)

        base_dir = os.path.dirname(os.path.abspath(__file__))
        if game_name == 'TicTacToe':
            self.save_dir = os.path.join(base_dir, "checkpoints/parallel/tictactoe")
        elif game_name == 'ConnectFour':
            self.save_dir = os.path.join(base_dir, "checkpoints/parallel/connectfour")
        os.makedirs(self.save_dir, exist_ok=True)

    def selfPlay(self, game_progress=None):
        return_memory = []
        player = 1
        self_play_games = [SPG(self.game) for  spg in range(self.args['num_parallel_games'])]

        while len(self_play_games) > 0:
            states = np.stack([spg.state for spg in self_play_games])
            neutral_states = self.game.change_game_perspective(states, player)

            self.mcts.search(neutral_states, self_play_games)

            for i in range(len(self_play_games))[::-1]:
                spg = self_play_games[i]
                action_probs = np.zeros(self.game.action_size)
                for c in spg.root.children:
                    action_probs[c.action_taken] = c.visit_count
                action_probs /= np.sum(action_probs)
                
                spg.memory.append((spg.root.state, action_probs, player))
                #adding temp
                temperature_action_probs = action_probs ** (1 / self.args['temperature'])
                temperature_action_probs /= np.sum(temperature_action_probs)
                action = np.random.choice(self.game.action_size, p=temperature_action_probs)

                spg.state = self.game.get_next_state(spg.state, action, player)

                value, is_terminal = self.game.get_value_and_terminated(spg.state, action)

                if is_terminal:
                    for hist_neutral_state, hist_action_probs, hist_player in spg.memory:
                        hist_outcome = value if hist_player == player else self.game.get_opponent_value(value)
                        return_memory.append((
                            self.game.get_encoded_state(hist_neutral_state),
                            hist_action_probs, hist_outcome
                        ))
                    del self_play_games[i]
                    if game_progress is not None:
                        game_progress.update(1)
                    
            player = self.game.get_opponent(player)
        return return_memory

    
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
            num_batches = self.args['num_selfPlay_iterations'] // self.args['num_parallel_games']
            for _ in trange(num_batches, desc="Self-play batches"):
                with tqdm(total=self.args['num_parallel_games'], desc="Completed games", leave=True) as game_progress:
                    memory += self.selfPlay(game_progress)

            self.model.train()
            for epoch in trange(self.args['num_epochs']):
                self.train(memory)

            torch.save(self.model.state_dict(), os.path.join(self.save_dir, f"model_{iter}_{self.game}.pt"))
            torch.save(self.optimizer.state_dict(), os.path.join(self.save_dir, f"optimizer_{iter}_{self.game}.pt"))


class SPG:
    def __init__(self, game):
        self.state = game.get_initial_state()
        self.memory = []
        self.root = None
        self.node = None


game_name = 'ConnectFour'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if game_name == 'TicTacToe':
    game = TicTacToe()
    model = ResNet(game, 4, 64, device)
    args = {
        'c': 2,
        'num_searches': 120,
        'num_iterations': 3,
        'num_selfPlay_iterations': 500,
        'num_parallel_games': 100,
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
        'num_searches': 200,   #600 for a strong model
        'num_iterations': 8,
        'num_selfPlay_iterations': 300,  #1000 good
        'num_parallel_games': 100,  #300 good
        'num_epochs': 4,
        'batch_size': 64,
        'temperature': 1.25,
        'dirichlet_epsilon': 0.25,
        'dirichlet_alpha': 0.3
    }

opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

alphaZero = AlphaParallelZero(model, optimizer=opt, game=game, args=args, game_name=game_name)
alphaZero.learn()