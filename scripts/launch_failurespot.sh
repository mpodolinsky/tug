#!/usr/bin/env bash
# Start the FailureSpot sweep (2 detectors x 9 (eta1, eta2) pairs x 20 splits = 360 runs) as one background worker; it keeps running after the
# terminal closes.
#   PYTHON=/path/to/python bash scripts/launch_failurespot.sh
# Log: results/logs/failurespot.log. Stop with: pkill -f "cli[.]run_baselines". Start it again to resume (finished runs are skipped).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/logs
setsid nohup "${PYTHON:-python}" -m cli.run_baselines --families failurespot_mlp failurespot_lstm > results/logs/failurespot.log 2>&1 < /dev/null &
echo "started; log in results/logs/failurespot.log"
