#!/usr/bin/env bash
# Start N training workers in the background; they keep running after the terminal closes.
#   PYTHON=/path/to/python bash scripts/launch_training.sh 3
# Logs: results/logs/worker_<i>_of_<N>.log. Stop with: pkill -f "cli.train --worker". Start it again to resume (finished runs are skipped).
set -euo pipefail
workers="${1:-3}"
cd "$(dirname "$0")/.."
mkdir -p results/logs
for i in $(seq 0 $((workers - 1))); do
  setsid nohup "${PYTHON:-python}" -m cli.train --worker "$i/$workers" > "results/logs/worker_${i}_of_${workers}.log" 2>&1 < /dev/null &
  sleep 30  # stagger the starts: each LIBERO-X worker needs about 6.6 GB of RAM while it loads the features
done
echo "started $workers workers; logs in results/logs/"
