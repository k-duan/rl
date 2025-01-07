# rl

<img src="plots/cartpolev1.gif" height="250"/>

This repo is for learning RL basics. I intend to keep things simple, for example:

* Focus on simple gym environment
* Minimum dependencies and avoid advanced implementations in pytorch
* Only use implementation tricks that are necessary to make things work
* Tensorization as much as possible, e.g. episode masks to handle variable sequence length

## Implementations

* REINFORCE
* A2C
* PPO

## Notes

* A2C suffers from instability issues. A few helpful tricks:
  * advantage normalization
  * use a learning schedule
  * gradient norm clipping
  * carefully tune learning rate and other parameters
* PPO makes training much more stable. See A2C and PPO (both using GAE with same lambda to calculate advantages) comparison on CartPole-V1 (experimented using [8181d9f](https://github.com/k-duan/rl/commit/8181d9f9b30e35811a9f039947fc08cf6cb50ad8)
):

<img src="plots/Screenshot%202025-01-07%20at%2010.37.27.png" height="250"/>

* There are other approaches (not implemented in this repo), e.g. A3C, SAC to make actor critic method more stable.
* Reinforcement Learning materials:
  * [Sergey Levine's CS285 open course](https://rail.eecs.berkeley.edu/deeprlcourse/)
  * [OpenAI RL Spinning Up](https://spinningup.openai.com/en/latest/index.html)
