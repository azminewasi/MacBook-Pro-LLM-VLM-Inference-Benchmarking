#!/bin/zsh
# Ends the queue as soon as the 4B 8x run finishes writing its summary, so the 9B 8x run never starts.
cd /Users/aw02305/Code/MacBenchmark/inference-testing
until [[ -f results/4B-think8x_summary.csv ]]; do sleep 2; done
pkill -f run_resume.sh                          # stop the queue first (4B python keeps finishing its W&B upload)
sleep 1; pkill -f "tag 9B-4bit-think8x"         # in case the 9B just started
while pgrep -f "tag 4B-think8x" >/dev/null; do sleep 2; done
echo "STOPPED_AFTER_4B $(date '+%F %T')"
