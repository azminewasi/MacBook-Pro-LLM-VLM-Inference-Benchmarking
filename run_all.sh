#!/bin/zsh
cd /Users/aw02305/Code/MacBenchmark/inference-testing && source .venv/bin/activate
export WANDB_API_KEY="$(cat .wandb_key)"
python bench.py --model Qwen/Qwen3.5-0.8B --tag 0.8B --dtype bf16
python bench.py --model Qwen/Qwen3.5-2B --tag 2B --dtype bf16
python bench.py --model Qwen/Qwen3.5-4B --tag 4B --dtype bf16
[ -d models/Qwen3.5-9B-4bit ] || python -m mlx_vlm convert --hf-path Qwen/Qwen3.5-9B --mlx-path models/Qwen3.5-9B-4bit -q --q-bits 4 --q-group-size 64
python bench.py --model models/Qwen3.5-9B-4bit --tag 9B-4bit --dtype 4bit
echo ALL_DONE
