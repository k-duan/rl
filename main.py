from typing import Callable

import gymnasium as gym
import torch
from tensorboard.compat.tensorflow_stub.dtypes import float32
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from nets import CategoricalPolicy


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

def _normalize_q_values(q_values: torch.Tensor) -> torch.Tensor:
   return (q_values - q_values.mean()) / (q_values.std() + 1e-8)

def _compute_loss(logits: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
   return -torch.mean(logits * rewards)

def _log(train: bool, it: int, writer: SummaryWriter, batch_stats: dict):
   for k, v in batch_stats.items():
      writer.add_scalar(f'{'train' if train else 'test'}/{k}', v, it)

class Batch:
   def __init__(self):
      self.data = {k: torch.tensor([], dtype=torch.float32) for k in self.keys()}

   def keys(self):
      return ["logits", "q_values", "episode_lengths"]

   def add(self, key: str, val: torch.Tensor):
      self.data[key] = torch.cat([self.data[key], val], dim=-1)

   def size(self):
      return self.data["episode_lengths"].size(-1)

   def __getitem__(self, key):
      return self.data[key]

   def clear(self):
      self.data = {k: torch.tensor([], dtype=torch.float32) for k in self.keys()}


def main():
   max_episode_steps = 500
   env = gym.make("CartPole-v1", render_mode="human", max_episode_steps=max_episode_steps)
   writer = SummaryWriter()
   rewards_fn = get_rewards_fn("rewards_to_go")
   policy = CategoricalPolicy(n_observations=4, n_actions=2, n_layers=2, hsize=32)
   optimizer = optim.AdamW(params=policy.parameters(), lr=0.01)
   batch_size = 8
   observation, info = env.reset(seed=42)
   # Create batch
   batch = Batch()

   for i in range(1000):
      logits, rewards = [], []

      # Sample one episode
      while True:
         # Action
         action, log_prob = policy.get_action(torch.tensor(observation, dtype=torch.float32).unsqueeze(dim=0))
         logits.append(log_prob)

         # step (transition) through the environment with the action
         # receiving the next observation, reward and if the episode has terminated or truncated
         observation, reward, terminated, truncated, info = env.step(torch.squeeze(action).tolist())
         rewards.append(reward)

         if terminated or truncated:
            observation, info = env.reset()
            break

      # Add episode to batch
      batch.add("logits", torch.stack(logits, dim=-1))
      batch.add("q_values", rewards_fn(torch.tensor([rewards], dtype=torch.float32), r=0.99))
      batch.add("episode_lengths", torch.tensor([[len(rewards)]], dtype=torch.float32))

      # Number of episode in the batch
      if batch.size() >= batch_size:
         # Normalize batch rewards and calculate loss
         # https://datascience.stackexchange.com/questions/20098/why-do-we-normalize-the-discounted-rewards-when-doing-policy-gradient-reinforcem
         loss = _compute_loss(batch["logits"], _normalize_q_values(batch["q_values"]))

         # Log stats
         _log(train=True, it=i, writer=writer, batch_stats={
            "batch_loss": loss,
            "batch_q_mean": batch["q_values"].mean(),
            "batch_q_std": batch["q_values"].std(),
            "batch_episode_length": torch.mean(batch["episode_lengths"]),
         })
         optimizer.zero_grad()
         loss.backward()
         optimizer.step()

         # Clear this batch
         batch.clear()

   writer.close()
   env.close()


if __name__ == "__main__":
   torch.manual_seed(123)
   main()

