# results

```
<dataset>/<split>/<method>/
  weights.pt, scores.npz
  curve_<calibrator>.json     alpha, fpr, tpr, tpr_early, on_time, ldr
  summary.json                area to FPR 40, extended flag, LTT point, config, seed
<dataset>/skipped.json        methods that could not run, with the reason
```

Plots are generated from these files only. Weights are git-ignored.
