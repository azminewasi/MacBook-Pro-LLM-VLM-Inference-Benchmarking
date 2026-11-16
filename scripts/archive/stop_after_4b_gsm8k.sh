#!/bin/zsh
# Ends the queue once the 4B 8x run has finished all 20 GSM8K samples (its AI2D half is left for later).
cd /Users/aw02305/Code/MacBenchmark/inference-testing
until [[ -f results/4B-think8x.jsonl ]] && [[ $(grep -c '"task": "gsm8k_text"' results/4B-think8x.jsonl) -ge 20 ]]; do sleep 5; done
pkill -f run_resume.sh                      # queue first, so the 9B never starts
sleep 1; pkill -INT -f "tag 4B-think8x"     # SIGINT lets W&B close the run cleanly (marks it interrupted)
sleep 10; pkill -f "tag 9B-4bit-think8x"
echo "STOPPED_AFTER_4B_GSM8K $(date '+%F %T')"
