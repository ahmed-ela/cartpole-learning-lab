# Diagnosis of the CartPole learning dip

The largest first regression in the selected run was caused by an actor update that introduced a harmful directional bias. The cart increasingly drifted off the left edge while the pole stayed relatively upright. Repeated optimization improved PPO's objective on its training batch while making fresh balancing performance worse. The app's moving average displayed this loss several updates late.

This conclusion concerns the actual representative run with seed **162066966**, from the earlier 128-run experiment. It does not establish one universal explanation for every reinforcement-learning dip.

![Measurements and interventions](dip-diagnosis.png)

## The policy deteriorated before the chart dipped

Exact replay reproduced all 80 original training-score samples and every completed training episode. The app source and preserved experiment source both have SHA-256 `cf8745bbc2faf82b0b05ed817867d0bfd339c09455567b95dab5a767d63b67d3`.

Each frozen checkpoint was evaluated on 1,500 fresh episodes with identical starting states and per-timestep action uniforms. Evaluation never consumed the training random-number stream.

| PPO checkpoint | Fresh sampled-policy mean | 95% interval | Displayed training mean |
|---|---:|---:|---:|
| 14 | 329.80 | 324.29–335.31 | 267.90 |
| 15 | 181.21 | 179.18–183.25 | 274.35 |
| 16 | 165.37 | 163.46–167.28 | 283.50 |
| 18 | 316.61 | 311.60–321.62 | 216.45 |
| 19 | 284.11 | 279.47–288.75 | 204.40 |

The actual regression occurred at **14 → 15 → 16**. The displayed last-20-episodes average peaked at 16 and reached its subsequent trough at 19. Completed episodes contain experience from earlier policies, and training episodes at a plotted update precede that update's optimization. Consequently the chart was still falling while the current policy was recovering. Its derivatives describe the moving average, not instantaneous policy quality.

## A controlled test identifies the directional offset

Write the actor's right-action probability as `p(right | s) = sigmoid(z(s))`. Its zero-state logit changed from **0.10557 at checkpoint 14 to 0.51712 at checkpoint 15**. Thus right-action probability at a perfectly centered, motionless, upright state rose from **52.64% to 62.65%**.

To test that offset, inference was changed to `sigmoid(z(s) - z(0))`. This subtracts one constant; all other network weights, state dependence, training history, and physical equations remain fixed. It is an experimental controller intervention, not further learning.

The hypothesis was then validated on a separate 2,000-episode holdout, using matched episodes for both checkpoints and both conditions:

| Controller | Before: checkpoint 14 | After: checkpoint 15 | After-minus-before change |
|---|---:|---:|---:|
| Original | 328.52 | 181.71 | −146.80 |
| Constant offset removed | 350.90 | 347.43 | −3.47 |

The original decrease's paired 95% interval was **−151.27 to −142.34**. With the offset removed, it was **−7.16 to +0.21**. The paired difference between those two changes was **+143.33 [137.80, 148.86]**. Almost all of the first major regression disappears under this single controller intervention.

In that holdout, **1,907/2,000** original checkpoint-15 episodes failed only at the cart boundary, all leftward. Their mean terminal pole angle was **4.29°**, below the 12° falling threshold. After offset removal, cart-boundary failures split between left and right and 418 episodes survived to 500 steps. A preference for pushing right at exactly zero does not imply eventual rightward drift: the force also rotates the pole, and the coupled feedback loop determines the eventual motion.

These controllers remain imperfect. Removing the offset raises performance to roughly 347, not a guaranteed 500. A smaller additional decline at checkpoint 16 involves other policy changes: removing its offset alone raised the 1,500-episode mean from 165.37 to 296.73; also removing its cart-position input raised it to 326.66. The latter is an exploratory ablation, not independently confirmed here. By checkpoint 40, cart-position feedback helps rather than harms.

## PPO optimized a misleading training proxy

During update 15's eight optimization epochs, a separate 750-episode trace showed:

- Fresh mean return: **335.89 → 308.41 after epoch 1 → 181.97 after epoch 8**.
- Clipped batch surrogate: **approximately 0 → 0.01366**, improving every epoch.
- The full objective including entropy also improved every epoch.

The harmful direction was present in the first epoch and amplified by continued optimization. This was not exclusively late-epoch overshoot. Deterministic action selection also deteriorated, ruling out action sampling alone as the explanation.

Single-update counterfactuals held the original checkpoint, Adam state, 1,024-transition batch, eight minibatch permutations, and critic updates fixed. On 1,000 matched evaluation episodes, actor learning rates 0.001, 0.0005, 0.00025, and 0 yielded means **180.53, 207.08, 240.04, and 326.09** respectively, against a pre-update mean of 326.09. Smaller actor steps mitigated this regression. They did not improve every tested update.

The default update's mean exact Bernoulli KL on batch states was **0.01641**. About 33.6% of observed-action probability ratios lay outside the clipping interval. These numbers do not establish a universal dangerous KL threshold. PPO clipping is not a strict policy-change constraint, as established in [Truly Proximal Policy Optimization](https://proceedings.mlr.press/v115/wang20b.html).

## Scope of the conclusion

The experiments identify a harmful actor update, a major causal contribution from its directional offset, physical failure through cart drift, and delayed reporting by the moving average. They do not uniquely separate finite-rollout sampling error, critic/advantage-estimation error, and optimizer dynamics as the origin of the misleading update direction. Critic error against the fixed batch targets decreased, which does not prove those targets accurately represented future reward.

Intervals measure episode-evaluation uncertainty conditional on these frozen policies. They are not uncertainty across independently trained policies, and they do not assume rewards or learning curves are normally distributed. The bias result was confirmed on an independent holdout after declaring the comparison. Source-gradient checks and exact arithmetic/RNG replay found no implementation bug in the tested PPO, GAE, or CartPole calculations. These experiments did not change the app's training behavior.

The accompanying [numeric data](dip-diagnosis-data.json) contains the plotted summaries, holdout contrasts, optimization trace, and intervention results. [Reproduction scripts and instructions](../../experiments/README.md) regenerate the full raw checkpoint outcomes; compact numeric summaries and preserved checkpoint weights are included in the repository.
