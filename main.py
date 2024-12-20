from typing import Callable, Optional

import numpy as np
import gymnasium as gym
import torch
from torch import optim
import torch.nn.functional as F
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
def _normalize(t: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
   if mask is not None:
      return (t - t[mask].mean()) / (t[mask].std() + 1e-8)
   return (t - t.mean()) / (t.std() + 1e-8)

def compute_policy_loss(logits: torch.Tensor, rewards: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
   if mask is not None:
      return -torch.mean(logits[mask] * rewards[mask])
   return -torch.mean(logits * rewards)

def compute_value_loss(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
   if mask is not None:
      return torch.nn.functional.mse_loss(logits[mask], target[mask])
   return torch.nn.functional.mse_loss(logits, target)

def _log(train: bool, it: int, writer: SummaryWriter, batch_stats: dict):
   for k, v in batch_stats.items():
      writer.add_scalar(f'{'train' if train else 'test'}/{k}', v, it)


class Batch:
   """
   observations: BxTxN
   logits: BxT
   rewards: BxT
   episode_masks: BxT
   """
   def __init__(self, max_episode_len: int, n_observations: int, n_actions: int):
      self.data = {
         "observations": torch.zeros(size=(0, max_episode_len, n_observations), dtype=torch.float32),
         "logits": torch.zeros(size=(0, max_episode_len), dtype=torch.float32),
         "rewards": torch.zeros(size=(0, max_episode_len), dtype=torch.float32),
         "rewards_to_go": torch.zeros(size=(0, max_episode_len), dtype=torch.float32),
         "episode_masks": torch.zeros(size=(0, max_episode_len), dtype=torch.bool),
      }
      self._max_episode_len = max_episode_len
      self._n_observations = n_observations
      self._n_actions = n_actions

   def add(self, kv: dict[str, torch.Tensor]):
      """
      :param kv: dictionary of key (tensor name) and value (tensor). Each tensor's first two leading dims are 1 and T
      :return:
      """
      for k, v in kv.items():
         assert v.size(0) == 1
         pad_size = (0, self._max_episode_len - v.size(1))  # 1xT
         if len(v.shape) == 3: # 1xTxN
            pad_size = (0, 0, 0, self._max_episode_len - v.size(1))
         v = F.pad(v, pad=pad_size)  # zero pad to max sequence length T
         self.data[k] = torch.cat([self.data[k], v])

   def __len__(self):
      return self.data["logits"].size(0)

   def __getitem__(self, key):
      return self.data[key]


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
   batch = Batch(max_episode_len=max_episode_steps, n_observations=4, n_actions=2)

   for i in range(5000):
      logits, rewards, observations = [], [], []

      # Sample one episode
      while True:
         # Action
         action, log_prob = policy_net.get_action(torch.tensor(observation, dtype=torch.float32).unsqueeze(dim=0))
         logits.append(log_prob)

         # step (transition) through the environment with the action
         # receiving the *next* observation, reward and if the episode has terminated or truncated
         observations.append(observation)
         observation, reward, terminated, truncated, info = env.step(torch.squeeze(action).tolist())
         rewards.append(reward)

         if terminated or truncated:
            observation, info = env.reset()
            break

      # Add episode to batch
      batch.add(kv={
         "observations": torch.tensor(np.array(observations), dtype=torch.float32).unsqueeze(dim=0),
         "logits": torch.stack(logits, dim=-1),
         "rewards": torch.tensor(rewards, dtype=torch.float32).unsqueeze(dim=0),
         "rewards_to_go": rewards_to_go_fn(torch.tensor([rewards], dtype=torch.float32), r=0.99),
         "episode_masks": torch.ones((1,len(rewards)), dtype=torch.bool)
      })

      # Number of episode in the batch
      if len(batch) >= batch_size:
         # Normalized rewards to go
         nrtg = _normalize(batch["rewards_to_go"], batch["episode_masks"])

         # import pdb; pdb.set_trace()
         # Compute value estimates
         # A = r(s_t, a_t) + \lambda * v(s_t) - v(s_{t+1})
         # (B, T, 4) -> (BxT, 4) -> (BxT, 1)
         values = value_net(batch["observations"].view(batch_size*max_episode_steps, -1))
         values = values.view(batch_size, max_episode_steps) # TODO optimize this

         # Value net optimization (normalized rewards to go)
         # (BxT, 1), (BxT, 1) -> (1,)
         # Choices of the target:
         # 1. Rewards to go (normalized)
         targets = nrtg

         # 2. Bootstrap estimate
         # targets = torch.zeros_like(values, dtype=torch.float32)
         # with torch.no_grad():
         #    targets[:, :-1] = batch["rewards"][:, :-1] + 0.99 * values[:, 1:]
         #    targets[:, -1] = batch["rewards"][:, -1]

         # Compute value loss
         value_loss = compute_value_loss(values, targets, batch["episode_masks"])
         value_optimizer.zero_grad()
         value_loss.backward()
         value_optimizer.step()

         # Compute advantage estimates
         advantages = torch.zeros_like(batch["logits"], dtype=torch.float32)
         with torch.no_grad():
            # Estimated "rewards to go":
            # Reward of current start, plus discounted value of future state:
            #  i.e. batch["rewards"][:, :-1] + 0.99 * values[:, 1:]
            # Baseline:
            # Value of current state:
            #  i.e. values[: , :-1]
            # 1. Advantage estimate of first T-1 states:
            # advantages[:, :-1] = batch["rewards"][:, :-1] + 0.99 * values[:, 1:] - values[: , :-1]
            # 2. Advantage estimate of the last state
            # advantages[:, -1] = batch["rewards"][:, -1] - values[:, -1] # edge case
            # TODO the issue here is that value function target is normalized while batch rewards above are not.
            #  What does this end up with?
            advantages[:, :-1] = nrtg[:, :-1] - values[:, :-1]
            advantages[:, -1] = nrtg[:, -1]

         # Policy net optimization
         # Normalize batch rewards and calculate loss
         # https://datascience.stackexchange.com/questions/20098/why-do-we-normalize-the-discounted-rewards-when-doing-policy-gradient-reinforcem
         policy_loss = compute_policy_loss(batch["logits"], _normalize(advantages, batch["episode_masks"]),
                                           batch["episode_masks"])
         policy_optimizer.zero_grad()
         policy_loss.backward()
         policy_optimizer.step()

         # Log stats
         _log(train=True, it=i, writer=writer, batch_stats={
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "rewards/mean": torch.mean(batch["rewards"][batch["episode_masks"]]),
            "rewards/std": torch.std(batch["rewards"][batch["episode_masks"]]),
            "rewards_to_go/mean": torch.mean(batch["rewards_to_go"][batch["episode_masks"]]),
            "rewards_to_go/std": torch.std(batch["rewards_to_go"][batch["episode_masks"]]),
            "episode_length/mean": batch["episode_masks"].sum() / len(batch),
         })

         # Clear this batch
         batch = Batch(max_episode_len=max_episode_steps, n_observations=4, n_actions=2)

   writer.close()
   env.close()


if __name__ == "__main__":
   torch.manual_seed(123)
   main()

