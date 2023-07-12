# Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Actor(nn.Module):
    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            init_w: float = 3e-3,
    ):
        """Initialize."""
        super(Actor, self).__init__()
        self.hidden1 = nn.Linear(in_dim, 128)
        self.hidden2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, out_dim)
        self.out.weight.data.uniform_(-init_w, init_w)
        self.out.bias.data.uniform_(-init_w, init_w)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        x = F.relu(self.hidden1(state))
        x = F.tanh(self.hidden2(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden2(x))
        x = F.tanh(self.hidden2(x))
        x = F.tanh(self.hidden2(x))
        action = self.out(x).sigmoid()  # Everything should in range [0,1]

        return action


class Critic(nn.Module):
    def __init__(
            self,
            in_dim: int,
            init_w: float = 3e-3,
    ):
        """Initialize."""
        super(Critic, self).__init__()

        self.hidden1 = nn.Linear(in_dim, 128)
        self.hidden2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, 1)

        self.out.weight.data.uniform_(-init_w, init_w)
        self.out.bias.data.uniform_(-init_w, init_w)

    def forward(
            self, state: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """Forward method implementation."""
        x = torch.cat((state, action), dim=-1)
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden2(x))
        value = self.out(x)

        return value


"""
    Resnet-9 for Actor/Critic
"""


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        residual = x
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        out += residual
        out = F.relu(out)
        return out


class Actor_Resnet9(nn.Module):
    def __init__(self, in_dim, out_dim, init_w=3e-3):
        super(Actor_Resnet9, self).__init__()
        self.fc = nn.Linear(in_dim, 128)
        self.residual_blocks = nn.Sequential(
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128)
        )
        self.out = nn.Linear(128, out_dim)
        self.out.weight.data.uniform_(-init_w, init_w)
        self.out.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = F.relu(self.fc(state))
        x = self.residual_blocks(x)
        action = self.out(x).sigmoid()
        return action


class Critic_Resnet9(nn.Module):
    def __init__(self, in_dim, init_w=3e-3):
        super(Critic_Resnet9, self).__init__()
        self.fc = nn.Linear(in_dim, 128)
        self.residual_blocks = nn.Sequential(
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128)
        )
        self.out = nn.Linear(128, 1)
        self.out.weight.data.uniform_(-init_w, init_w)
        self.out.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=-1)
        x = F.relu(self.fc(x))
        x = self.residual_blocks(x)
        value = self.out(x)
        return value
