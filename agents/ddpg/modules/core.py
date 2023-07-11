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
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class Actor_Resnet9(nn.Module):
    def __init__(self, in_dim, out_dim, init_w=3e-3):
        super(Actor, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.residual_blocks = nn.Sequential(
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
            ResidualBlock(64, 64)
        )
        self.out = nn.Linear(64, out_dim)
        self.out.weight.data.uniform_(-init_w, init_w)
        self.out.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = F.relu(self.bn1(self.conv1(state)))
        x = self.residual_blocks(x)
        x = F.avg_pool2d(x, kernel_size=x.size()[2:])
        x = x.view(x.size(0), -1)
        action = self.out(x).sigmoid()
        return action


class Critic_Resnet9(nn.Module):
    def __init__(self, in_dim, init_w=3e-3):
        super(Critic, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.residual_blocks = nn.Sequential(
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
            ResidualBlock(64, 64)
        )
        self.out = nn.Linear(64, 1)
        self.out.weight.data.uniform_(-init_w, init_w)
        self.out.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x = F.relu(self.bn1(self.conv1(state)))
        x = self.residual_blocks(x)
        x = F.avg_pool2d(x, kernel_size=x.size()[2:])
        x = x.view(x.size(0), -1)
        x = torch.cat((x, action), dim=-1)
        value = self.out(x)
        return value
