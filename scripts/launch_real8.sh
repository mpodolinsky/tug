#!/usr/bin/env bash
# Train every probe and run every baseline that the real-robot data allows, on its 8 leave-one-task-out splits (the action-based baselines are
# skipped: its saved plans hold only the executed actions). Three background workers; they keep running after the terminal closes.
#   PYTHON=/path/to/python bash scripts/launch_real8.sh
# The probes of the main table (lstm_modes_clip, both labels) are trained first and keep their weights; the other architectures do not, to spare
# disk (26 MB each). Logs: results/logs/real8_*.log. Stop with: pkill -f "cli[.](train|run_baselines)". Start it again to resume (finished runs are skipped).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/logs
python_bin="${PYTHON:-python}"
setsid nohup bash -c "'$python_bin' -m cli.train --datasets real8_groot --probes lstm_modes_clip --variants onset_pre onset_traj && '$python_bin' -m cli.train --datasets real8_groot --no-weights" > results/logs/real8_probes.log 2>&1 < /dev/null &
sleep 20
setsid nohup "$python_bin" -m cli.run_baselines --datasets real8_groot --families mahalanobis euclidean pca_kmeans safe_mlp safe_lstm > results/logs/real8_baselines.log 2>&1 < /dev/null &
sleep 20
setsid nohup "$python_bin" -m cli.run_baselines --datasets real8_groot --families safe_mlp_pre safe_lstm_pre > results/logs/real8_safe_pre.log 2>&1 < /dev/null &
echo "started 3 workers; logs in results/logs/real8_*.log"
