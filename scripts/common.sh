# Sourced by the other scripts: go to the project root, activate the venv, load the W&B key, define helpers.
cd "$(dirname "${(%):-%x}")/.." || exit 1
source .venv/bin/activate
export WANDB_API_KEY="$(cat .wandb_key)"
mkdir -p logs

# run_bench <tag> <model> [extra bench.py args]  -> skipped if results/<tag>_summary.csv exists (= finished).
# Interrupted runs have no summary, so they resume sample-by-sample.
run_bench() {
  local tag=$1 model=$2; shift 2
  [[ -f results/${tag}_summary.csv ]] && { echo "skip $tag (finished)"; return; }
  python bench.py --model "$model" --tag "$tag" "$@"
}
# run_mitigate <tag> <model> [extra args]  -> skipped if results/<tag>-mitigate_summary.csv exists.
run_mitigate() {
  local tag=$1 model=$2; shift 2
  [[ -f results/${tag}-mitigate_summary.csv ]] && { echo "skip mitigate $tag (finished)"; return; }
  python mitigate_truncated.py --model "$model" --tag "$tag" "$@"
}
NINE=models/Qwen3.5-9B-4bit
ensure_9b() {   # one-time 4-bit conversion of the 9B (19 GB bf16 does not fit in 16 GB RAM)
  [[ -d $NINE ]] || python -m mlx_vlm convert --hf-path Qwen/Qwen3.5-9B --mlx-path $NINE -q --q-bits 4 --q-group-size 64
}
