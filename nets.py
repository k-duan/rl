import torch
from torch import nn
from torch.distributions import Categorical


class CategoricalPolicy(nn.Module):
    def __init__(self, n_observations: int, n_actions: int, n_layers: int, hsize: int):
        super().__init__()
        self.input = nn.Linear(n_observations, hsize)
        self.hidden_layers = nn.ModuleList([nn.Linear(hsize, hsize) for _ in range(n_layers)])
        self.output = nn.Linear(hsize, n_actions)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        out = nn.functional.relu(self.input(obs))
        for layer in self.hidden_layers:
            out = nn.functional.relu(layer(out))
        return nn.functional.relu(self.output(out))

    def get_action(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward(observation)
        m = Categorical(logits=logits)
        action = m.sample()
        return action, m.log_prob(action)

