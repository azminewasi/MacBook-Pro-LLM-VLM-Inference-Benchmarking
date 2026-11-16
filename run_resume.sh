#!/bin/zsh
# Continues after the 2B: 4B, 9B (4-bit), then the 8x-thinking runs (20 samples). Resumable: finished samples are skipped.
cd /Users/aw02305/Code/MacBenchmark/inference-testing && source .venv/bin/activate
export WANDB_API_KEY="$(cat .wandb_key)"
python bench.py --model Qwen/Qwen3.5-4B --tag 4B --dtype bf16
[ -d models/Qwen3.5-9B-4bit ] || python -m mlx_vlm convert --hf-path Qwen/Qwen3.5-9B --mlx-path models/Qwen3.5-9B-4bit -q --q-bits 4 --q-group-size 64
python bench.py --model models/Qwen3.5-9B-4bit --tag 9B-4bit --dtype 4bit
echo ALL_DONE
python bench.py --model Qwen/Qwen3.5-0.8B --tag 0.8B-think8x --think-only --cap-mult 8 --n 20
python bench.py --model Qwen/Qwen3.5-2B --tag 2B-think8x --think-only --cap-mult 8 --n 20
python bench.py --model Qwen/Qwen3.5-4B --tag 4B-think8x --think-only --cap-mult 8 --n 20
python bench.py --model models/Qwen3.5-9B-4bit --tag 9B-4bit-think8x --dtype 4bit --think-only --cap-mult 8 --n 20
echo ALL8X_DONE
