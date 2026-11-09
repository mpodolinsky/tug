#!/usr/bin/env bash
# Start the SAFE-MLP / SAFE-LSTM sweeps (12 configurations x 20 splits per family) as two background workers; they keep running after the
# terminal closes.
#   PYTHON=/path/to/python bash scripts/launch_safe.sh
# Worker 1 runs SAFE as published (safe_mlp, safe_lstm); worker 2 the same models on rollouts cut at the onset (safe_mlp_pre, safe_lstm_pre:
# not SAFE as published, an ablation of the labels). Early stopping with patience 50, see docs/SAFE_BASELINES.md.
# Logs: results/logs/safe_*.log. Stop with: pkill -f "cli[.]run_baselines". Start it again to resume (finished runs are skipped).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/logs
python_bin="${PYTHON:-python}"
setsid nohup "$python_bin" -m cli.run_baselines --families safe_mlp safe_lstm > results/logs/safe_published.log 2>&1 < /dev/null &
sleep 45  # the first worker's preflight and feature loading finish before the second starts
setsid nohup "$python_bin" -m cli.run_baselines --families safe_mlp_pre safe_lstm_pre > results/logs/safe_pre_onset.log 2>&1 < /dev/null &
echo "started 2 workers; logs in results/logs/"
