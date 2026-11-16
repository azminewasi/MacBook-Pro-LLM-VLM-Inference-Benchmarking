#!/bin/zsh
# Mitigation experiment: re-run truncated+wrong thinking-on samples with top-k 20 + presence penalty 1.5. Run after 01/02, BEFORE 04_cleanup.
# 1x sets: up to 20 failures per task; 8x sets: up to 10 per task. Originals are never modified.
source "$(dirname "$0")/common.sh"
run_retry 0.8B Qwen/Qwen3.5-0.8B --max-per-task 20
run_retry 2B   Qwen/Qwen3.5-2B   --max-per-task 20
run_retry 4B   Qwen/Qwen3.5-4B   --max-per-task 20
run_retry 9B-4bit $NINE --dtype 4bit --max-per-task 20
run_retry 0.8B-think8x Qwen/Qwen3.5-0.8B --max-per-task 10
run_retry 2B-think8x   Qwen/Qwen3.5-2B   --max-per-task 10
run_retry 4B-think8x   Qwen/Qwen3.5-4B   --max-per-task 10
run_retry 9B-4bit-think8x $NINE --dtype 4bit --max-per-task 10
echo RETRY_DONE
