# Conditional PPO update intervention study

Seed: `162066966`. Source: the immutable-for-this-analysis snapshot `experiments/curve-analysis/rl.snapshot.mjs`.

Reproduce from the repository root:

```sh
node experiments/curve-diagnosis/interventions/run.mjs
```

The script writes an accessor-only copy of the source RNG so complete trainer states can be cloned. Its numeric RNG body and all PPO operations remain identical. Full actor/critic weights, gradients, Adam moments, environment state, completed episodes, and step/update counts agree exactly between the original source and accessor copy through update 30. Original training episode histories also reproduce exactly.

## Method

For each original update 12 through 24, clone the original trainer before collecting its rollout. Collect one original 1,024-transition batch, including its original advantage estimates and fixed return targets. Clone the complete post-collection trainer state for each intervention. Use actor learning rates `.001`, `.0005`, `.00025`, and `0`; preserve critic learning rate `.001`, all eight epochs, all minibatch permutations, original Adam moments, and every other setting.

Each resulting policy is evaluated on 1,000 fresh episodes, with identical initial states and identical per-timestep action uniforms across policies. Evaluation randomness is separate from training. Shared utility: `experiments/curve-diagnosis/evaluation/eval.mjs`. Evaluation dataset seed: `cartpole-intervention-independent-v1`. Returns measure undiscounted episode survival, capped at 500. Paired 95% percentile bootstrap intervals use 2,000 resamples. These are pointwise intervals, not simultaneous intervals across the experiment.

The batch diagnostics report exact Bernoulli KL divergence from old to new policy, averaged over the fixed batch states; observed-action ratio clip fraction; the advantage-sign-dependent active clip fraction; clipped policy surrogate gain; and critic MSE against the batch's fixed targets. All measured policy probabilities were inside numerical clamp bounds.

Two multi-update branch experiments preserve all outcomes through update 30:

- Original checkpoint 11: before the earliest independently evaluated regression in the prespecified update range.
- Original checkpoint 14: before the main early regression independently identified by the frozen-checkpoint study.

Each branch uses actor learning rates `.001`, `.0005`, and `.00025`. After the first update, different policies naturally collect different data. Those branches are full training interventions, unlike the single-update tests that hold the data fixed.

## Main results

The original checkpoint 14 sampled-policy mean was **326.094**. Updating it on its original batch produced:

| Actor learning rate | Mean survival after update 15 | Change from checkpoint 14 |
| --- | ---: | ---: |
| .001 | 180.525 | -145.569 |
| .0005 | 207.077 | -119.017 |
| .00025 | 240.036 | -86.058 |
| 0 | 326.094 | 0 |

Default update 15's paired change interval was `[-151.859, -139.323]`. Thus the decline is an effect of this actor update, beyond uncertainty from a few evaluation episodes. Lowering the actor step attenuated this particular regression; freezing its weights preserved the previous performance while the critic still trained.

Despite declining survival, default update 15 increased the clipped training surrogate by `0.0136623`. Critic training MSE against fixed batch targets declined from `0.0027590` to `0.0012326`. Batch mean KL was `0.0164068`. Observed-action ratios ranged from `0.7406` to `1.4955`; 33.59% were outside the `[0.8, 1.2]` interval and 16.50% were actively clipped by the objective.

Update 16 caused a further 14.397-point decline; updates 17 and 18 caused substantial recovery. The displayed 20-episode training mean peaks at update 16 and drops through update 19, so its timing lags the actual policy regression.

For branches from original checkpoint 14, update 30 means were **389.828**, **500.000**, and **488.177** for default, half, and quarter actor learning rates. Half-rate minus default at update 30 was `110.172`, paired interval `[104.618, 115.559]`. Every one of the 1,000 sampled evaluation episodes survived to the cap under the half-rate policy at updates 26–30.

## Limits

These are causal interventions on one selected seed, its realized original batches, and fixed checkpoints. They do not establish a universally best actor learning rate. Smaller steps sometimes performed worse: the checkpoint-14 quarter-rate branch scored 249.374 at update 18 versus the default branch's 314.257, and quarter-rate single-update 23 performed substantially worse than that update's default outcome.

Positive batch surrogate gain together with negative independently measured survival demonstrates a mismatch between that training proxy and realized policy performance. These interventions do not separately identify the contributions of old critic error, sample noise, optimizer momentum, discounting, or changed state visitation. Improving critic MSE against fixed training targets is not proof of more accurate true values. Batch KL measures the rollout's states, not every possible state.

Running the script generates `results.json` with all raw evaluation return arrays, physical failure summaries, metrics, and branch outcomes. This bulk file and the generated accessor-only RNG source are ignored by Git. The committed `summary.json` is the compact numeric report; `manifest.json` records source hashes, settings, and selection rules. These experiments do not alter the live app files or settings.
