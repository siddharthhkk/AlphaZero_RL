import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from TicTacToe import TicTacToe

class ResNet(nn.Module):
    def __init__(self, game, num_resBlocks, num_hidden, device):
        super().__init__()
        self.device = device
        self.startBlock = nn.Sequential(
            nn.Conv2d(3, num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU()
        )
        self.backbone = nn.ModuleList([ResBlock(num_hidden) for i in range(num_resBlocks)])

        self.policy_head = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * game.row_count *  game.column_count, game.action_size)
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(num_hidden, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * game.row_count * game.column_count, 1),
            nn.Tanh()
        )
        self.to(device)

    def forward(self, x):
        x = self.startBlock(x)
        for resblock in self.backbone:
            x = resblock(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value

#ResBlock
class ResBlock(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm2d(num_hidden)

    def forward(self, x):
        residual = x
        x = F.relu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x)) + residual
        return F.relu(x)


def main():
    game = TicTacToe()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = game.get_initial_state()
    state = game.get_next_state(state, 2, -1)
    state = game.get_next_state(state, 4, -1)
    state = game.get_next_state(state, 6, 1)
    state = game.get_next_state(state, 8, 1)
    #print(state)

    encoded_state = game.get_encoded_state(state)
    #print(encoded_state)

    state_tensor = torch.tensor(encoded_state, device=device).unsqueeze(0)

    model = ResNet(game, 4, 64, device=device)
    model.load_state_dict(torch.load('checkpoints/model_2.pt', map_location=device))
    model.eval()
    policy, value = model.forward(state_tensor)
    policy = torch.softmax(policy, axis=1).squeeze(0).detach().cpu().numpy()
    print(policy, value.item())
    plt.bar(range(game.action_size), policy) # show the policy distribution for each action (how likely the model thinks each action is the best move)
    plt.show()

if __name__ == "__main__":
    main()