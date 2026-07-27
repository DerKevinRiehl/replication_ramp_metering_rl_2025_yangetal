import torch
import torch.nn as nn
from torch.distributions import Normal

class SharedActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim=1, log_std=-0.5):
        super(SharedActorCritic, self).__init__()

        layer_size = 256

        self.shared = nn.Sequential(
            nn.Linear(state_dim, layer_size),
            nn.Tanh(),
            nn.Linear(layer_size, layer_size),
            nn.Tanh(),
            nn.Linear(layer_size, layer_size),
            nn.Tanh()
        )

        self.actor_mean = nn.Sequential(
            nn.Linear(layer_size, layer_size),
            nn.Tanh(),
            nn.Linear(layer_size, action_dim),
            nn.Sigmoid()
        )

        self.actor_log_std = nn.Parameter(torch.full((action_dim,), log_std))

        self.critic = nn.Sequential(
            nn.Linear(layer_size, layer_size),
            nn.Tanh(),
            nn.Linear(layer_size, 1)
        )

    def forward(self, state):
        shared_features = self.shared(state)

        state_value = self.critic(shared_features)

        action_mean = self.actor_mean(shared_features)
        action_std = self.actor_log_std.exp()

        dist = Normal(action_mean, action_std)

        return dist, state_value
