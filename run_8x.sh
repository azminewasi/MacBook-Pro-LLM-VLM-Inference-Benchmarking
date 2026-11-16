#!/bin/zsh
cd /Users/aw02305/Code/MacBenchmark/inference-testing && source .venv/bin/activate
export WANDB_API_KEY="$(cat .wandb_key)"
while ! grep -q ALL_DONE run_all.log; do sleep 60; done
python bench.py --model Qwen/Qwen3.5-0.8B --tag 0.8B-think8x --think-only --cap-mult 8 --n 20
python bench.py --model Qwen/Qwen3.5-2B --tag 2B-think8x --think-only --cap-mult 8 --n 20
python bench.py --model Qwen/Qwen3.5-4B --tag 4B-think8x --think-only --cap-mult 8 --n 20
python bench.py --model models/Qwen3.5-9B-4bit --tag 9B-4bit-think8x --dtype 4bit --think-only --cap-mult 8 --n 20
echo ALL8X_DONE
