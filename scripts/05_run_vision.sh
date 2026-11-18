#!/bin/zsh
# Vision arm: SpatialEval spatial-map (4-option multiple choice on map images), 100 questions, thinking off (cap 640) and on (cap 2048), per model.
# The same 100 evaluation questions, prompt and scoring as the post-training project, so base-model numbers carry over. Resumable; finished models are skipped.
# Results go to results/<tag>-spatial.jsonl (separate from the main benchmark). Run when nothing else is using the GPU.
source "$(dirname "$0")/common.sh"
run_bench 0.8B-spatial Qwen/Qwen3.5-0.8B --dtype bf16 --tasks spatial_vision
run_bench 2B-spatial   Qwen/Qwen3.5-2B   --dtype bf16 --tasks spatial_vision
run_bench 4B-spatial   Qwen/Qwen3.5-4B   --dtype bf16 --tasks spatial_vision
ensure_9b
run_bench 9B-4bit-spatial $NINE --dtype 4bit --tasks spatial_vision
echo VISION_DONE
