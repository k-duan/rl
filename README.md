# rl

## Vanilla Policy Gradient Method

1. Batch size makes a big difference (so we can lower the variance using 3) below). Policy gradient on single episode samples *DO NOT* work.
2. Network size makes a big difference (#hidden layers)

Check the plot below (2 hidden layers v.s. 1 hidden layer). 
With 2 hidden layers VPG is able to achieve max episode length (500) with less than 400 sampled episodes.

![Screenshot 2024-12-04 at 23.54.55.png](plots/Screenshot%202024-12-04%20at%2023.54.55.png)

3. Rewards must be normalized by mean and std. Without doing this policy gradient methods *DO NOT*
4. Compare different rewards functions

![Screenshot 2024-12-09 at 09.11.12.png](plots/Screenshot%202024-12-09%20at%2009.11.12.png)

 - All three implementations have identical hyperparameters (learning rate=`0.01`, batch size=`8`, policy network configs (two hidden layers, hidden layewr size=`32`, etc.) except the rewards calculation methods used.
 - #iterations taken to achieve max episode length: `rewards_to_go` (383) > `vanilla` (431) > `discounted rewards` (591). One hypothesis is that this CartPole task is capped to 500 max episode length, so the total rewards is still bounded (i.e. not a real infinite time sequence). Note that here `rewards_to_go` is implemented with discounting factor.
 - Batch reward statistics are not comparable when reward calculations are different. For example `vanilla` and `discounted_rewards` achieves close to zero std (which makes sense because each state in the time sequence share the same trajectory reward, and both methods sample episodes at max episode length in the end); `vanilla` has much higher mean because no discounting factors are used, but still bounded due to max episode length.
 - `rewards_to_go` looks better by showing a decreasing loss trend (compared with `vanilla` and `discounted_rewards` methods). However batch losses are still quite noisy for all three methods.
 - All three methods eventually converge to max episode length. However `rewards_to_go` oscillate a little bit while  `discounted` and `vanilla` completely stays at max episode length; their training loss both converge to zero. Since the loss is a (reward) weighted sum of negative log likelihood in action space, this indicates the policy model "overfit" the episode samples and becomes "deterministic" policy (outputting 100% or 0% on left/right actions).
 - *TODO: Any other metrics to monitor?*
6. *TODO: Using value net and advantage estimates*
 - Fit value net with rewards to go
 - (Bootstrap estimate)Fit value net to `r(s_t,a_t) + v^hat(s_{t+1})`
 - N-step advantage estimates
