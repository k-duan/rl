# rl

## Vanilla Policy Gradient

1. Batch size makes a big difference (so we can lower the variance using 3) below). Policy gradient on single episode samples *DO NOT* work.
2. Network size makes a big difference (#hidden layers)

Check the plot below (2 hidden layers v.s. 1 hidden layer). 
With 2 hidden layers VPG is able to achieve max episode length (500) with less than 400 sampled episodes.

![Screenshot 2024-12-04 at 23.54.55.png](plots/Screenshot%202024-12-04%20at%2023.54.55.png)

3. Advantages must be normalized by mean and std. Without doing these policy gradient methods *DO NOT* work.
4. Compare different rewards functions

![Screenshot 2024-12-09 at 09.11.12.png](plots/Screenshot%202024-12-09%20at%2009.11.12.png)

 - All three implementations have identical hyperparameters (learning rate=`0.01`, batch size=`8`, policy network configs (two hidden layers, hidden layewr size=`32`, etc.) except the rewards calculation methods used.
 - #iterations taken to achieve max episode length: `rewards_to_go` (383) > `vanilla` (431) > `discounted rewards` (591). One hypothesis is that this CartPole task is capped to 500 max episode length, so the total rewards is still bounded (i.e. not a real infinite time sequence). Note that here `rewards_to_go` is implemented with discounting factor.
 - Batch reward statistics are not comparable when reward calculations are different. For example `vanilla` and `discounted_rewards` achieves close to zero std (which makes sense because each state in the time sequence share the same trajectory reward, and both methods sample episodes at max episode length in the end); `vanilla` has much higher mean because no discounting factors are used, but still bounded due to max episode length.
 - `rewards_to_go` looks better by showing a trend of the policy loss (compared with `vanilla` and `discounted_rewards` methods). However batch losses are still quite noisy for all three methods. *Reason: non-stationary distribution due to constantly changing policy.*
 - All three methods eventually converge to max episode length. However `rewards_to_go` oscillate a little bit while  `discounted` and `vanilla` completely stays at max episode length; their training loss both converge to zero. Since the loss is a (reward) weighted sum of negative log likelihood in action space, this indicates the policy model "overfit" the episode samples and becomes "deterministic" policy (outputting 100% or 0% on left/right actions).
 - *TODO: Any other metrics to monitor? Looks like the only reasonable metric is total rewards vs episode time steps.*

## Actor-Critic
1. Using TD errors:
 - a) TD error w/o MC returns + (batch mean/std) normalized `rewards_to_go` as value target + (batch mean/std) normalized `advantages`: never reached max episode length (can at most reach ~40 steps by normalizing raw rewards using mean/std of `rewards_to_go`). *Reason: Bias-Variance tradeoff?*
 - b) TD error with MC returns + (batch mean/std) normalized `rewards_to_go` as value target + (batch mean/std) normalized `advantages`: took 303 iterations to reach max episode length.

![Screenshot 2024-12-19 at 16.18.09.png](plots/Screenshot%202024-12-19%20at%2016.18.09.png)

 - c) TD error with MC returns + raw `rewards_to_go` as value target + (batch mean/std) normalized `advantages`:  took 375 iterations to reach max episode length.

![Screenshot 2024-12-23 at 11.45.22.png](plots/Screenshot%202024-12-23%20at%2011.45.22.png)

 - In both b) and c), value loss does not converge well. Reasons: correlated / non i.i.d. samples (parallel synchronization? replay buffer? target net?).
2. *TODO (Bootstrap estimate) Fit value net to `r(s_t,a_t) + 0.99*v^hat(s_{t+1})`*.
3. GAE: interpolation between 1a) and 1b):

![Screenshot 2024-12-25 at 18.06.32.png](plots/Screenshot%202024-12-25%20at%2018.06.32.png)
