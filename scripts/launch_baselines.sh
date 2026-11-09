#!/usr/bin/env bash
# Start the embedding-distance baseline sweep (26 configurations x 20 splits) as two background workers; they keep running after the terminal closes.
#   PYTHON=/path/to/python bash scripts/launch_baselines.sh
# PCA-KMeans runs alone because it peaks at about 12 GB of RAM on LIBERO-X; Mahalanobis and Euclidean (about 7 GB) share the other worker.
# Logs: results/logs/baselines_*.log. Stop with: pkill -f "cli[.]run_baselines". Start it again to resume (finished runs are skipped).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/logs
python_bin="${PYTHON:-python}"
setsid nohup "$python_bin" -m cli.run_baselines --families pca_kmeans > results/logs/baselines_pca_kmeans.log 2>&1 < /dev/null &
sleep 30  # the first worker's preflight and feature loading finish before the second starts
setsid nohup "$python_bin" -m cli.run_baselines --families mahalanobis euclidean > results/logs/baselines_mahalanobis_euclidean.log 2>&1 < /dev/null &
echo "started 2 workers; logs in results/logs/"
