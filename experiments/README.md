# Reproduce the learning-curve experiments

The app's PPO implementation is preserved in `curve-analysis/rl.snapshot.mjs`. The recorded experiments use this snapshot and do not change the live app. Run every command below from the repository root. Node scripts have no package dependencies; they were verified with Node 26.

The committed data includes the complete 128-run ensemble, its fixed seeds, compact diagnostic summaries, and actor/critic checkpoint states. Larger raw per-episode diagnostic arrays are generated on demand and ignored by Git.

## Curve ensemble

```sh
node experiments/curve-analysis/analyze.mjs
```

This reruns the 128 recorded seeds for 80 updates each and regenerates the curve data and three imposed k-medoids groups. It uses three worker threads. See `curve-analysis/README.md` for settings, score definitions, and interpretation limits.

## Frozen-policy dip diagnosis

```sh
node experiments/curve-diagnosis/evaluation/diagnose.mjs
node experiments/curve-diagnosis/evaluation/confirm.mjs
node experiments/curve-diagnosis/evaluation/center-holdout.mjs
```

Run these in order. The first verifies exact seed replay and evaluates 26 frozen checkpoints on 1,500 matched episodes each. The second generates the independent 2,000-episode holdout outcomes. The third uses those holdout outcomes to test removal of the constant policy logit. See `curve-diagnosis/evaluation/README.md` for the physical failure modes, chart lag, paired confidence intervals, and causal scope.

## Conditional optimizer tests and diagnostics

```sh
node experiments/curve-diagnosis/interventions/run.mjs
node experiments/curve-diagnosis/audit/epoch-trace.mjs
node experiments/curve-diagnosis/ablations.mjs
node experiments/curve-diagnosis/control-field.mjs
```

The intervention script reproduces same-batch actor learning-rate counterfactuals and multi-update branches. The epoch trace follows original updates 15 and 16 without changing their arithmetic. The remaining scripts reproduce frozen-policy input/logit interventions and local controller derivatives. These are conditional studies of the selected run, not universal tuning recommendations.

## Figures

```sh
python3 -m venv experiments/.venv
experiments/.venv/bin/python -m pip install -r experiments/requirements.txt
experiments/.venv/bin/python experiments/curve-analysis/plot_curves.py
experiments/.venv/bin/python experiments/curve-diagnosis/plot_diagnosis.py
```

Both plotting scripts read committed compact data and write PNG/SVG figures into `docs/learning-curves`. Pass `--output-dir PATH` to render elsewhere. Installing Python plotting dependencies is optional; all numerical Node experiments can run without them.
