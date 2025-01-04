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

    def get_action(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward(obs)
        m = Categorical(logits=logits)
        acts = m.sample()
        return acts, m.log_prob(acts)

    def get_lob_probs(self, obs: torch.Tensor, acts: torch.Tensor) -> torch.Tensor:
        logits = self.forward(obs)
        m = Categorical(logits=logits)
        return m.log_prob(acts)


class ValueNet(nn.Module):
    def __init__(self, n_observations: int, n_layers: int, hsize: int):
        super().__init__()
        self.input = nn.Linear(n_observations, hsize)
        self.hidden_layers = nn.ModuleList([nn.Linear(hsize, hsize) for _ in range(n_layers)])
        self.output = nn.Linear(hsize, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        out = nn.functional.relu(self.input(obs))
        for layer in self.hidden_layers:
            out = nn.functional.relu(layer(out))
        return nn.functional.relu(self.output(out))
