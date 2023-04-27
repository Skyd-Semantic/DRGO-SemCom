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
        x = F.tanh(self.hidden2(x))
        x = F.tanh(self.hidden2(x))
        x = F.tanh(self.hidden2(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden2(x))
        value = self.out(x)

        return value
