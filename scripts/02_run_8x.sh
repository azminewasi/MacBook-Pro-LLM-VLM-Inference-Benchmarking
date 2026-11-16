#!/bin/zsh
# Long-thinking runs: thinking-on only, 8x token caps (24,576 GSM8K / 16,384 AI2D), first 20 samples per task. Resumable.
# Optional: pass model tags to run a subset, e.g.  scripts/02_run_8x.sh 4B 9B-4bit
source "$(dirname "$0")/common.sh"
want=("$@"); [[ ${#want} -eq 0 ]] && want=(0.8B 2B 4B 9B-4bit)
declare -A M=([0.8B]=Qwen/Qwen3.5-0.8B [2B]=Qwen/Qwen3.5-2B [4B]=Qwen/Qwen3.5-4B [9B-4bit]=$NINE)
for t in $want; do
  [[ $t == 9B-4bit ]] && { ensure_9b; D=4bit; } || D=bf16
  run_bench ${t}-think8x ${M[$t]} --dtype $D --think-only --cap-mult 8 --n 20
done
echo EIGHTX_DONE
