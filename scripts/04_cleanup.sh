#!/bin/zsh
# Removes temporary / downloaded things. DRY RUN by default (just lists sizes); pass --yes to delete.
# Keeps: results/, scripts, README, compare.py. Run only after 01-03 are finished.
cd "$(dirname "$0")/.." || exit 1
HF=~/.cache/huggingface/hub
T=(tmp models wandb __pycache__ $HF/models--Qwen--Qwen3.5-0.8B $HF/models--Qwen--Qwen3.5-2B $HF/models--Qwen--Qwen3.5-4B $HF/models--Qwen--Qwen3.5-9B $HF/datasets--lmms-lab--ai2d $HF/datasets--openai--gsm8k)
size() { # HF caches hold symlinks into blobs/, so sum the resolved file sizes
  if [[ -d $1/snapshots ]]; then find $1/snapshots -type l -exec stat -L -f %z {} + | awk '{s+=$1} END {printf "%.1f GB", s/1e9}' ; else du -sh "$1" | cut -f1; fi; }
for p in $T; do [[ -e $p ]] && echo "$(size $p)\t$p"; done
if [[ "$1" == "--yes" ]]; then for p in $T; do rm -rf "$p"; done; echo "deleted. (post-train-testing will re-download the models + GSM8K)"; else echo "dry run - nothing deleted. Re-run with --yes."; fi
