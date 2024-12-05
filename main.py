import gymnasium as gym
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from nets import CategoricalPolicy

# rewards: BxT
def _discounted_rewards(rewards: torch.Tensor, r: float = 0.99) -> torch.Tensor:
   d = 1.0
   out = torch.zeros(size=rewards.size(), dtype=torch.float32)
   for i in range(rewards.size(-1)):
      out[i] = d * rewards[i]
      d *= r
   return out

def _rewards_to_go(rewards: torch.Tensor, r: float = 0.99) -> torch.Tensor:
   out = torch.zeros(size=rewards.size(), dtype=torch.float32)
   out[-1] = rewards[-1]
   for i in range(rewards.size(-1)-1)[::-1]:
      out[i] = rewards[i] + r * out[i+1]
   return out

def _compute_loss(logits: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
   return -torch.mean(logits * rewards)

def _log(train: bool, it: int, writer: SummaryWriter, batch_stats: dict):
   for k, v in batch_stats.items():
      writer.add_scalar(f'{'train' if train else 'test'}/{k}', v, it)

def main():
   max_episode_steps = 500
   env = gym.make("CartPole-v1", render_mode="human", max_episode_steps=max_episode_steps)
   writer = SummaryWriter()
   policy = CategoricalPolicy(n_observations=4, n_actions=2, n_layers=2, hsize=32)
   optimizer = optim.AdamW(params=policy.parameters(), lr=0.01)
   batch_size = 8
   observation, info = env.reset(seed=42)
   # only need logits and rewards to compute the loss
   batch_logits, batch_rewards, batch_episode_lens = [], [], []

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
         reward = torch.unsqueeze(torch.tensor(reward, dtype=torch.float32), dim=0)
         rewards.append(reward)

         if terminated or truncated:
            observation, info = env.reset()
            break
      rewards_to_go = _rewards_to_go(torch.cat(rewards), r=0.99)
      batch_logits.extend(logits)
      batch_rewards.extend(rewards_to_go)
      batch_episode_lens.append(len(rewards_to_go))

      # Number of episode in the batch
      if len(batch_episode_lens) >= batch_size:
         # Normalize batch rewards
         rws = torch.stack(batch_rewards)
         loss = _compute_loss(torch.cat(batch_logits), (rws - rws.mean()) / (rws.std() + 1e-8))

         # Log stats
         _log(train=True, it=i, writer=writer, batch_stats={
            "batch_loss": loss.item(),
            "batch_reward_mean": rws.mean(),
            "batch_reward_std": rws.std(),
            "batch_episode_length": torch.mean(torch.tensor(batch_episode_lens, dtype=torch.float32))
         })
         optimizer.zero_grad()
         loss.backward()
         optimizer.step()

         # Clear this batch
         batch_logits, batch_rewards, batch_lens = [], [], []

   writer.close()
   env.close()


if __name__ == "__main__":
   torch.manual_seed(123)
   main()

