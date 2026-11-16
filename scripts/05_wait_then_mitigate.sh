#!/bin/zsh
# Waits until the 9B 8x run has finished (end of scripts/02_run_8x.sh 4B 9B-4bit), then runs the mitigate experiment.
cd "$(dirname "$0")/.." || exit 1
until [[ -f results/9B-4bit-think8x_summary.csv ]]; do sleep 60; done
sleep 30   # let the 9B process finish its W&B upload
exec scripts/03_run_mitigate.sh
