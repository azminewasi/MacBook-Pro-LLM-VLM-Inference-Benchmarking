#!/bin/zsh
# Main benchmark: 400 inferences per model (GSM8K + AI2D, thinking off/on), small -> large. Resumable; finished models are skipped.
# Tip: caffeinate -ims scripts/01_run_main.sh > logs/main.log 2>&1 &
source "$(dirname "$0")/common.sh"
run_bench 0.8B Qwen/Qwen3.5-0.8B --dtype bf16
run_bench 2B   Qwen/Qwen3.5-2B   --dtype bf16
run_bench 4B   Qwen/Qwen3.5-4B   --dtype bf16
ensure_9b
run_bench 9B-4bit $NINE --dtype 4bit
echo MAIN_DONE
