from typing import Callable, Optional

import numpy as np
import gymnasium as gym
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from nets import CategoricalPolicy, ValueNet


def get_rewards_fn(reward_type: str) -> Callable:
   # rewards: BxT
   # pylint: disable=unused-argument
   def _rewards(rewards: torch.Tensor, r: float = 0.99) -> torch.Tensor:
      return torch.sum(rewards, dim=-1).unsqueeze(dim=-1).repeat([1, rewards.size(-1)])

   # rewards: BxT
   def _discounted_rewards(rewards: torch.Tensor, r: float = 0.99) -> torch.Tensor:
      d = 1.0
      out = torch.zeros(size=rewards.size(), dtype=torch.float32)
      for i in range(rewards.size(-1)):
         out[:, i] = d * rewards[:, i]
         d *= r
      return torch.sum(out, dim=-1).unsqueeze(dim=-1).repeat([1, out.size(-1)])

   # rewards: BxT
   def _rewards_to_go(rewards: torch.Tensor, r: float = 0.99) -> torch.Tensor:
      out = torch.zeros(size=rewards.size(), dtype=torch.float32)
      out[:, -1] = rewards[:, -1]
      for i in range(rewards.size(-1) - 1)[::-1]:
         out[:, i] = rewards[:, i] + r * out[:, i + 1]
      return out

   reward_fn = {
      "vanilla": _rewards,
      "discounted": _discounted_rewards,
      "rewards_to_go": _rewards_to_go,
   }
   return reward_fn[reward_type]

@torch.no_grad()
def _normalize(t: torch.Tensor) -> torch.Tensor:
   return (t - t.mean()) / (t.std() + 1e-8)

@torch.no_grad()
def _normalize_v2(nt: torch.Tensor) -> torch.Tensor:
   """
   nt: a nested tensor with leading dim as batch dim
   Normalize along batch dim
   """
   assert nt.is_nested
   return torch.nested.nested_tensor([(t - t.mean()) / (t.std() + 1e-8) for t in nt.unbind()])

def compute_policy_loss(logits: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
   return -torch.mean(logits * rewards)

def compute_policy_loss_v2(logits: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
   assert logits.is_nested
   assert rewards.is_nested
   assert logits.size(0) == rewards.size(0)

   weighted_logits = logits * rewards
   return -torch.mean(torch.stack([torch.mean(wl) for wl in weighted_logits.unbind()]))

def compute_value_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
   return torch.nn.functional.mse_loss(logits, target)

def compute_value_loss_v2(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
   assert logits.is_nested
   assert target.is_nested
   assert logits.size(0) == target.size(0)

   return torch.mean(torch.stack([torch.nn.functional.mse_loss(l, t) for l, t in zip(logits.unbind(), target.unbind())]))

def _log(train: bool, it: int, writer: SummaryWriter, batch_stats: dict):
   for k, v in batch_stats.items():
      writer.add_scalar(f'{'train' if train else 'test'}/{k}', v, it)

def slice_nested(nt: torch.Tensor, start: int, end: Optional[int]) -> torch.Tensor:
   """
   :param nt: a nested tensor
   :param start: start dim of each tensor in t
   :param end: end dim of each tensor in t
   :return: a new nested tensor with sliced dim
   """
   assert nt.is_nested
   return torch.nested.nested_tensor([t[start:end] for t in nt.unbind()], requires_grad=nt.requires_grad)

def squeeze_nested(nt: torch.Tensor, dim: int) -> torch.Tensor:
   assert nt.is_nested
   return torch.nested.nested_tensor([t.squeeze(dim=dim) for t in nt.unbind()], requires_grad=nt.requires_grad)

def mean_nested(nt: torch.Tensor) -> torch.Tensor:
   assert nt.is_nested
   return torch.mean(torch.cat([t if t.size() else t.unsqueeze(dim=0) for t in nt.unbind()]))

def std_nested(nt: torch.Tensor) -> torch.Tensor:
   assert nt.is_nested
   return torch.std(torch.cat([t if t.size() else t.unsqueeze(dim=0) for t in nt.unbind()]))


class Batch:
   """
   Flattened batch of shape (1x(sum of T))
   """
   def __init__(self):
      self.data = {k: torch.tensor([], dtype=torch.float32) for k in self.keys()}

   def keys(self):
      return ["logits", "rewards", "rewards_to_go", "episode_lengths", "observations", "next_observations"]

   def add(self, key: str, val: torch.Tensor):
      self.data[key] = torch.cat([self.data[key], val], dim=-1)

   def __len__(self):
      return self.data["episode_lengths"].size(-1)

   def __getitem__(self, key):
      return self.data[key]

   def clear(self):
      self.data = {k: torch.tensor([], dtype=torch.float32) for k in self.keys()}


class BatchV2:
   """
   Allows variable dimension on T to accommodate different episodes
   logits: BxT
   rewards: BxT
   observations: BxTxN
   """
   def __init__(self):
      self.data = self._new()

   def _new(self) -> dict[str, torch.Tensor]:
      return {k: torch.nested.nested_tensor([], dtype=torch.float32) for k in self.keys()}

   def keys(self):
      return ["logits", "rewards", "rewards_to_go", "episode_lengths", "observations"]

   def add(self, key: str, val: torch.Tensor):
      """
      :param key: name of the tensor
      :param val: new tensor without leading batch dimension
      """
      self.data[key] = torch.nested.nested_tensor(list(self.data[key].unbind()) + [val], requires_grad=val.requires_grad)

   def __len__(self):
      return self.data["logits"].size(dim=0)

   def __getitem__(self, key):
      return self.data[key]

   def clear(self):
      self.data = self._new()


def main():
   max_episode_steps = 500
   env = gym.make("CartPole-v1", render_mode="human", max_episode_steps=max_episode_steps)
   writer = SummaryWriter()
   rewards_to_go_fn = get_rewards_fn("rewards_to_go")
   policy_net = CategoricalPolicy(n_observations=4, n_actions=2, n_layers=2, hsize=32)
   policy_optimizer = optim.AdamW(params=policy_net.parameters(), lr=0.01)
   value_net = ValueNet(n_observations=4, n_layers=2, hsize=32)
   value_optimizer = optim.AdamW(params=value_net.parameters(), lr=0.01)
   batch_size = 8
   observation, info = env.reset(seed=42)
   # Create batch
   batch = BatchV2()

   for i in range(1000):
      logits, rewards, observations = [], [], []

      # Sample one episode
      while True:
         # Action
         action, log_prob = policy_net.get_action(torch.tensor(observation, dtype=torch.float32).unsqueeze(dim=0))
         logits.append(log_prob.squeeze(dim=0))

         # step (transition) through the environment with the action
         # receiving the *next* observation, reward and if the episode has terminated or truncated
         observations.append(observation)
         observation, reward, terminated, truncated, info = env.step(torch.squeeze(action).tolist())
         rewards.append(reward)

         if terminated or truncated:
            observation, info = env.reset()
            break

      # Add episode to batch
      batch.add("logits", torch.stack(logits, dim=-1))
      batch.add("rewards", torch.tensor(rewards, dtype=torch.float32))
      batch.add("rewards_to_go", rewards_to_go_fn(torch.tensor([rewards], dtype=torch.float32), r=0.99).squeeze(dim=0))
      batch.add("episode_lengths", torch.tensor(len(rewards), dtype=torch.float32))
      batch.add("observations", torch.tensor(np.array(observations), dtype=torch.float32))

      # Number of episode in the batch
      if len(batch) >= batch_size:
         # Compute value estimates
         # A = r(s_t, a_t) + \lambda * v(s_t) - v(s_{t+1})
         values = value_net(batch["observations"])
         values = squeeze_nested(values, dim=-1)
         current_values, next_values = slice_nested(values, 0, -1), slice_nested(values, 1, None)
         advantages = slice_nested(batch["rewards"], 0, -1) + 0.99 * current_values - next_values

         # Policy net optimization
         # Normalize batch rewards and calculate loss
         # https://datascience.stackexchange.com/questions/20098/why-do-we-normalize-the-discounted-rewards-when-doing-policy-gradient-reinforcem
         policy_loss = compute_policy_loss_v2(slice_nested(batch["logits"], 0, -1), _normalize_v2(advantages))
         policy_optimizer.zero_grad()
         policy_loss.backward()
         policy_optimizer.step()

         # Value net optimization
         value_loss = compute_value_loss_v2(values, batch["rewards_to_go"])
         value_optimizer.zero_grad()
         value_loss.backward()
         value_optimizer.step()

         # Log stats
         _log(train=True, it=i, writer=writer, batch_stats={
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "rewards/mean": mean_nested(batch["rewards"]),
            "rewards/std": std_nested(batch["rewards"]),
            "rewards_to_go/mean": mean_nested(batch["rewards_to_go"]),
            "rewards_to_go/std": std_nested(batch["rewards_to_go"]),
            "episode_length/mean": mean_nested(batch["episode_lengths"]),
         })

         # Clear this batch
         batch.clear()

   writer.close()
   env.close()


if __name__ == "__main__":
   torch.manual_seed(123)
   main()

