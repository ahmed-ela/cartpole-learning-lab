# Epoch diagnostic for seed 162066966

The learner was replayed exactly through checkpoint 14. At updates 15 and 16, the original actor update was wrapped solely to copy weights after each of the eight original optimization epochs. All actor/critic weights, gradients, Adam moments, optimizer times, environment state, episode returns, and recorded training curves matched an untouched control through update 16. No application file changed.

Every frozen policy was evaluated on 750 identical presampled initial states and step-index action uniforms. The sampled-policy mean is primary; deterministic evaluation provides a check against an explanation based solely on random exploration. Uncertainty below is a normal approximation for the paired episode-return difference.

Update 15 damages actual performance from the first epoch, while its batch surrogate and full objective including entropy both improve throughout. Later epochs amplify that harmful direction. This is a direct local demonstration of optimizing the rollout proxy without improving actual expected episode reward; it does not identify uniquely whether advantage error, sampling, discount/horizon, occupancy change, or optimizer momentum originated the proxy mismatch.

## Original update 15 (checkpoint 14 to 15)

| Epoch | Sampled mean | Deterministic mean | Surrogate | Exact KL old to new | Cart-only failures |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 335.89 | 488.16 | -0.000000 | 0.000000 | 81.5% |
| 1 | 308.41 | 403.44 | 0.003895 | 0.000663 | 87.6% |
| 2 | 248.38 | 279.43 | 0.009813 | 0.005181 | 96.3% |
| 3 | 221.08 | 247.43 | 0.010832 | 0.008944 | 97.9% |
| 4 | 208.60 | 230.36 | 0.012148 | 0.010125 | 97.1% |
| 5 | 199.02 | 220.22 | 0.012934 | 0.011754 | 96.4% |
| 6 | 189.96 | 210.23 | 0.013437 | 0.013612 | 96.1% |
| 7 | 184.74 | 204.65 | 0.013638 | 0.015347 | 95.3% |
| 8 | 181.97 | 201.19 | 0.013662 | 0.016407 | 94.7% |

First epoch paired change from before: -27.48; 95% interval [-34.57, -20.39]. Final paired change: -153.91; 95% interval [-161.22, -146.61].

## Original update 16 (checkpoint 15 to 16)

| Epoch | Sampled mean | Deterministic mean | Surrogate | Exact KL old to new | Cart-only failures |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 181.97 | 201.19 | -0.000000 | 0.000000 | 94.7% |
| 1 | 177.91 | 197.65 | 0.000614 | 0.000264 | 92.4% |
| 2 | 177.65 | 197.19 | 0.000832 | 0.000327 | 91.3% |
| 3 | 172.87 | 191.69 | 0.001490 | 0.001228 | 90.7% |
| 4 | 169.50 | 187.57 | 0.002053 | 0.002408 | 88.0% |
| 5 | 167.13 | 185.15 | 0.002484 | 0.003446 | 86.7% |
| 6 | 164.25 | 182.50 | 0.002652 | 0.004623 | 84.0% |
| 7 | 169.99 | 188.70 | 0.002621 | 0.002497 | 86.3% |
| 8 | 166.08 | 184.61 | 0.003087 | 0.004298 | 84.3% |

First epoch paired change from before: -4.06; 95% interval [-4.87, -3.25]. Final paired change: -15.89; 95% interval [-17.46, -14.33].

## Files

`epoch-trace.mjs` is the reproducible diagnostic. Running it generates `epoch-results.json` with full evaluations, per-epoch weights, and batch metrics; that bulk file is ignored by Git. The committed `epoch-summary.json` removes the episode-level bulk.
