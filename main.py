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

def get_advantage_fn(advantage_type: str) -> Callable:

   @torch.no_grad()
   def _rewards_to_go(batch: Batch, **kwargs) -> torch.Tensor:
      return batch["rewards_to_go"]

   @torch.no_grad()
   def _monte_carlo_td_error(batch: Batch, values: torch.Tensor, **kwargs) -> torch.Tensor:
      return batch["rewards_to_go"] - values

   @torch.no_grad()
   def _td_error(batch: Batch, values: torch.Tensor, r: float = 0.99, **kwargs) -> torch.Tensor:
      td_errors = torch.zeros_like(batch["rewards"], dtype=torch.float32)
      td_errors[:, :-1] = batch["rewards"][:, :-1] + r * values[:, 1:] - values[: , :-1]
      td_errors[:, -1] = batch["rewards"][:, -1] - values[:, -1] # edge case
      return td_errors

   @torch.no_grad()
   def _gae(batch: Batch, values: torch.Tensor, r: float = 0.99, l: float = 1.0) -> torch.Tensor:
      # GAE can be calculated recursively:
      # GAE(s_t, a_t) = td_error(t) + r * l * GAE(s_{t+1}, a_{t+1})
      # where GAE(s_T, a_T) = td_error(T)
      # GAE(l=1.0) -> `_monte_carlo_td_error`
      # GAE(l=0) -> `_td_error`
      td_errors = _td_error(batch, values)
      gae = torch.zeros_like(td_errors, dtype=torch.float32)
      gae[:, -1] = td_errors[:, -1]
      for i in range(gae.size(-1) - 1)[::-1]:
         gae[:, i] = td_errors[:, i] + r * l * gae[:, i + 1]
      return gae

   advantage_fn = {
      "rewards_to_go": _rewards_to_go,
      "monte_carlo_td_error": _monte_carlo_td_error,
      "td_error": _td_error,
      "gae": _gae,
   }

   return advantage_fn[advantage_type]

@torch.no_grad()
def _normalize(t: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
   mean, std = t.mean(), t.std()
   if mask is not None:
      mean, std = t[mask].mean(), t[mask].std()
   return (t - mean) / (std + 1e-8)

def get_policy_loss_fn(loss_type: str) -> Callable:
   def vpg_loss(logits: torch.Tensor, advantages: torch.Tensor, mask: torch.Tensor = None, **kwargs):
      if mask is None:
         mask = torch.ones_like(logits, dtype=torch.bool)
      return -torch.mean(logits[mask] * advantages[mask])
   def ppo_loss(logits: torch.Tensor, advantages: torch.Tensor, mask: torch.Tensor = None, old_logits: torch.Tensor = None, clip_ratio: float = 0.2):
      if mask is None:
         mask = torch.ones_like(logits, dtype=torch.bool)
      # Importance sampling probability ratio rho = \pi(a|s) / \pi_old(a|s)
      rho = torch.exp(logits - old_logits)
      # min(rho*advantages, clip(rho, 1-clip_ratio, 1+clip_ratio)*advantages)
      return torch.mean(torch.min(rho[mask] * advantages[mask], torch.clamp(rho[mask], 1-clip_ratio, 1+clip_ratio) * advantages[mask]))

   policy_loss_fn = {
      "vpg": vpg_loss,
      "ppo": ppo_loss,
   }

   return policy_loss_fn[loss_type]

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
         "actions": torch.zeros(size=(0, max_episode_len), dtype=torch.float32),
         "logits": torch.zeros(size=(0, max_episode_len), dtype=torch.float32),
         "rewards": torch.zeros(size=(0, max_episode_len), dtype=torch.float32),
         "rewards_to_go": torch.zeros(size=(0, max_episode_len), dtype=torch.float32),
         "episode_masks": torch.zeros(size=(0, max_episode_len), dtype=torch.bool),
      }
      self._max_episode_len = max_episode_len

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

   def __setitem__(self, key, value):
      self.data[key] = value


def main():
   max_episode_steps = 500
   env = gym.make("CartPole-v1", render_mode="human", max_episode_steps=max_episode_steps)
   writer = SummaryWriter()
   rewards_to_go_fn = get_rewards_fn("rewards_to_go")
   advantage_fn = get_advantage_fn("gae")
   policy_loss_fn = get_policy_loss_fn("vpg")
   policy_net = CategoricalPolicy(n_observations=4, n_actions=2, n_layers=2, hsize=32)
   policy_optimizer = optim.AdamW(params=policy_net.parameters(), lr=0.01)
   policy_update_steps = 1
   value_net = ValueNet(n_observations=4, n_layers=2, hsize=32)
   value_optimizer = optim.AdamW(params=value_net.parameters(), lr=0.01)
   value_update_steps = 1
   batch_size = 8
   observation, info = env.reset(seed=42)
   # Create batch
   batch = Batch(max_episode_len=max_episode_steps, n_observations=4, n_actions=2)

   for i in range(1000):
      actions, logits, rewards, observations = [], [], [], []

      # Sample one episode
      while True:
         # Action
         action, log_prob = policy_net.get_action(torch.tensor(observation, dtype=torch.float32).unsqueeze(dim=0))
         actions.append(action)
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
         "actions": torch.stack(actions, dim=-1),
         "logits": torch.stack(logits, dim=-1),
         "rewards": torch.tensor(rewards, dtype=torch.float32).unsqueeze(dim=0),
         "rewards_to_go": rewards_to_go_fn(torch.tensor([rewards], dtype=torch.float32), r=0.99),
         "episode_masks": torch.ones((1,len(rewards)), dtype=torch.bool)
      })

      # Number of episode in the batch
      if len(batch) >= batch_size:
         # Normalized rewards to go
         # batch["nrtg"] = _normalize(batch["rewards_to_go"], batch["episode_masks"])

         # Value net optimization
         for _ in range(value_update_steps):
            # Compute value estimates
            # (B, T, 4) -> (BxT, 4) -> (BxT, 1)
            values = value_net(batch["observations"].view(batch_size*max_episode_steps, -1))
            values = values.view(batch_size, max_episode_steps) # TODO optimize this

            # regression target: normalized rewards to go
            # (BxT, 1), (BxT, 1) -> (1,)
            targets = batch["rewards_to_go"]
            value_loss = compute_value_loss(values, targets, batch["episode_masks"])
            value_optimizer.zero_grad()
            value_loss.backward()
            value_optimizer.step()

         # Compute advantage estimates
         advantages = advantage_fn(batch, values, r=0.99, l=0.95)

         # Policy net optimization
         for _ in range(policy_update_steps):
            # Normalize batch rewards and calculate loss
            # https://datascience.stackexchange.com/questions/20098/why-do-we-normalize-the-discounted-rewards-when-doing-policy-gradient-reinforcem
            log_probs = policy_net.get_lob_probs(batch["observations"].view(batch_size*max_episode_steps, -1),
                                                    batch["actions"].view(-1))
            policy_loss = policy_loss_fn(logits=log_probs.view(batch_size, max_episode_steps),
                                         advantages=_normalize(advantages, batch["episode_masks"]),
                                         mask=batch["episode_masks"],
                                         old_logits=batch["logits"].detach(),
                                         clip_ratio=0.2)
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

