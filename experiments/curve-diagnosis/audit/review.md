# Read-only implementation audit

Inspected the exact `dist/rl.js` and preserved curve experiment. No concrete PPO, GAE, or normal CartPole-step implementation bug found that explains the seed-162066966 dip.

## Mathematical checks

- Network forward/backprop indexing and tanh chain rule are correct. Independent central finite differences checked all 193 weights: maximum absolute error 2.55e-11.
- Bernoulli clipped-PPO actor-loss gradient and entropy-loss sign are correct. Independent central finite differences checked 200 cases across actions, positive/negative advantages, and clipped/unclipped ratios: maximum absolute error 1.42e-9.
- GAE includes next-state bootstrap value, resets its continuation at episode boundaries, bootstraps the 500-step time-limit truncation, and zeros next value on physical failure. The rollout-end GAE cut is standard.
- Normal step equations match the standard CartPole parameterization: total mass 1.1, pole mass-length 0.05, pole half-length 0.5, Euler step 0.02. Drag physics is not used during training.
- The numeric probability clamp is inconsistent with a derivative of the clamped function at extreme logits, but is inactive in the implicated run: pre-update rollout logits at updates 8–28 had maximum absolute values 2.87–4.30, versus the clamp threshold near 13.815. No inspected rollout samples were clamped. No evidence it contributes here.

## Displayed-dip interpretation

Training point N is computed after learning update N, but its returns were collected before that parameter update. Completed episodes can span multiple policy updates. The recent-20 mean also retains older policies and cannot attribute a dip to the currently labeled checkpoint.

For this seed, rolling point 16 is 283.5, yet newly completed episodes at 16 already averaged 235.17. Point 17 is 230.7, with newly completed episodes averaging 161.17; points 18 and 19 are 216.45 and 204.4, with new-episode means 216.2 and 209.67. The visible peak therefore hides an earlier decline, and updates 12–18 deserve investigation.

## Attribution limits and discriminating diagnostics

1. Frozen policies evaluated on many fresh matched episodes can distinguish true performance deterioration from moving-average lag and small-sample noise. Ordinary `evaluate(sameSeed, count)` does not match episodes across policies after different trajectory lengths: one RNG stream supplies both resets and actions. Use separate per-episode initial states and a pre-sampled uniform sequence indexed by episode and step, or equivalent independent streams.
2. Track cart-boundary versus pole-angle termination, together with final x, velocity, angle, and angular velocity. Cart drift can explain the physical route to lost returns; it does not itself establish why the optimizer changed the policy.
3. Same-batch smaller-actor-LR counterfactuals, preserving optimizer moments and minibatch order, can establish that a more conservative parameter update avoids the damage in this batch. They do not alone prove a bad critic, excessive exploration, or a universally excessive learning rate.
4. Mean KL on old rollout states misses rare/high-impact states and occupancy changes. Report the KL distribution/max as well as mean, and evaluate decisions on successful/failing trajectory states. PPO clipping constrains the sampled surrogate; it is not a hard parameter or KL bound, and entropy gradients plus Adam momentum can move a weight after its direct policy-loss contribution is clipped.
5. To test momentum as the source, preserve the identical batch/order but zero actor Adam moments in a separate counterfactual. To test advantage estimation, compare against matched-policy long-rollout/Monte-Carlo estimates or refit the critic independently; poor critic explained variance alone is not causal proof.
6. Larger rollouts, fewer epochs, or multiple independent minibatch permutations address sampling/optimization sensitivity. Their recovery establishes sensitivity to those changes, not a unique root cause. Avoid assigning broad algorithmic claims from one selected dip.
