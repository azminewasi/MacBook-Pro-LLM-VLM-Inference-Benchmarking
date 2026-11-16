#!/bin/zsh
# Mitigation experiment: re-run truncated+wrong thinking-on samples with top-k 20 + presence penalty 1.5. Run after 01/02, BEFORE 04_cleanup.
# 1x sets: up to 40 failures per task; 8x sets: up to 10 per task. Originals are never modified.
source "$(dirname "$0")/common.sh"
run_mitigate 0.8B Qwen/Qwen3.5-0.8B --max-per-task 40
run_mitigate 2B   Qwen/Qwen3.5-2B   --max-per-task 40
run_mitigate 4B   Qwen/Qwen3.5-4B   --max-per-task 40
run_mitigate 9B-4bit $NINE --dtype 4bit --max-per-task 40
run_mitigate 0.8B-think8x Qwen/Qwen3.5-0.8B --max-per-task 10
run_mitigate 2B-think8x   Qwen/Qwen3.5-2B   --max-per-task 10
run_mitigate 4B-think8x   Qwen/Qwen3.5-4B   --max-per-task 10
run_mitigate 9B-4bit-think8x $NINE --dtype 4bit --max-per-task 10
echo MITIGATE_DONE
