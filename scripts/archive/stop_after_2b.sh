#!/bin/zsh
# Ends the queue as soon as the 2B 8x run finishes, so the 4B 8x run never starts.
cd /Users/aw02305/Code/MacBenchmark/inference-testing
until [[ -f results/2B-think8x_summary.csv ]]; do sleep 2; done
pkill -f run_resume.sh                          # queue first, so nothing new starts
while pgrep -f "tag 2B-think8x" >/dev/null; do sleep 2; done   # let the 2B finish its W&B upload
pkill -f "tag 4B-think8x"                       # in case the 4B slipped in (nothing is lost)
echo "STOPPED_AFTER_2B $(date '+%F %T')"
