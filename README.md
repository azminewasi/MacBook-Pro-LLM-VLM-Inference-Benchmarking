# Mac LLM/VLM Inference Testing

How far can a single 16 GB MacBook go at running modern LLMs and VLMs locally? This project benchmarks
**Qwen3.5 (0.8B, 2B, 4B, 9B)** on an **Apple M5 MacBook** with [MLX](https://github.com/ml-explore/mlx), measuring
quality, speed, memory and system load for every single inference, and logging everything to Weights & Biases.

> **Status:** 0.8B and 2B are complete. 4B, 9B (4-bit) and the long-thinking (8x token budget) runs are in progress; the tables below will be extended as they finish.

## Overview
- **Hardware:** Apple M5, 10 CPU/GPU cores, 16 GB unified memory, macOS 26 (Darwin 25.6), plugged in. MLX 0.32, mlx-vlm 0.7.12.
- **Models** (all natively multimodal, so one set of weights does both tasks):
  `Qwen/Qwen3.5-0.8B`, `-2B`, `-4B` in bf16; `-9B` is ~19 GB in bf16 and cannot fit, so it is quantised to 4-bit.
- **Tasks (100 samples each):**
  - Text: **GSM8K** test (first 100), grade-school math, exact-match on the final number.
  - Vision: **AI2D** test (first 100), multiple-choice science-diagram questions, options numbered 1-4.
- **Settings:** thinking **off** and **on** for each task, so 400 inferences per model.
  Thinking-off is greedy; thinking-on uses temperature 0.6 / top-p 0.95 (Qwen's recommendation).
  Token caps: thinking-off 512 (GSM8K) / 32 (AI2D); thinking-on 3072 (GSM8K) / 2048 (AI2D).
  A follow-up run repeats thinking-on with **8x the caps on the first 20 samples** per task to separate capability from token budget.
  Samples that hit the cap without answering count as wrong, and the truncation rate is reported.
- **Original vs LoRA:** the first plan included LoRA variants; this was dropped in favour of the thinking on/off comparison. Post-training lives in the sibling project *Mac LLM/VLM Post-training Testing*.

## Metrics (logged per sample, aggregated per model/task/mode)
| Group | Metrics |
|---|---|
| Quality | accuracy (+95% Wilson CI, paired sign test between models), truncation rate, finish reason |
| Latency | time to first token (TTFT), end-to-end latency (mean, p95), time/tokens per correct answer |
| Throughput | prefill tokens/s, decode tokens/s, prompt tokens, generated tokens |
| Memory | peak and active MLX (GPU) memory, process RSS, system RAM used, swap used |
| Compute load | system CPU %, GPU device utilisation % (ioreg), thermal CPU speed limit (pmset) |
| Model | load time, load memory |

Not collected: power in watts (needs `sudo powermetrics`; battery readings are meaningless on AC).

## Results so far (100 samples per cell; ±~10 points at 95% confidence)
| Model | Task | Thinking | Accuracy | Truncated | Gen tokens | Decode tok/s | Latency | Peak mem |
|---|---|---|---|---|---|---|---|---|
| 0.8B | GSM8K | off | 54% | 21% | 357 | 73.5 | 4.9 s | 1.8 GB |
| 0.8B | GSM8K | on | 22% | 94% | 2960 | 70.6 | 42 s | 1.8 GB |
| 0.8B | AI2D | off | 41% | 5% | 4 | 140* | 0.2 s | 2.4 GB |
| 0.8B | AI2D | on | 48% | 43% | 1308 | 71.0 | 18.6 s | 2.5 GB |
| 2B | GSM8K | off | 77% | 18% | 349 | 31.5 | 11.2 s | 4.3 GB |
| 2B | GSM8K | on | 65% | 49% | 2320 | 30.4 | 76 s | 4.3 GB |
| 2B | AI2D | off | 46% | 0% | 3 | 56* | 0.3 s | 5.1 GB |
| 2B | AI2D | on | 73% | 34% | 1132 | 31.1 | 37 s | 5.1 GB |

\*AI2D thinking-off answers are ~3-4 tokens, so decode tok/s there is dominated by overhead.

**Findings so far**
- 2B beats 0.8B significantly everywhere except AI2D without thinking (both near guessing, 4 options = 25% chance; paired sign test p=0.53).
- Thinking helps vision (2B: 46% -> 73%) but *hurts* GSM8K at the 3072-token cap, mostly because of truncation (94% of 0.8B and 49% of 2B thinking runs never finish) - hence the 8x-cap experiment.
- Decode is memory-bandwidth bound: speed falls ~2.3x from 0.8B to 2B (~2.6x more weights). Both keep the GPU ~97-99% busy with CPU ~10%.
- No swap and no thermal throttling for 0.8B/2B. The 4B (8.5 GB in bf16) is close to the 16 GB limit and starts using swap; decode ~14 tok/s.

Full per-sample data is in `results/` (`<model>.jsonl`, `<model>_summary.csv`, `compare_*.csv`).

## Repository layout
```
bench.py               benchmark harness (one model per call; resumable via results/<tag>.jsonl)
compare.py             cross-model comparison: CIs, paired tests, cost per correct answer
mitigate_truncated.py     mitigation experiment: mitigate truncated+wrong thinking samples with top-k 20 + presence penalty
scripts/
  01_run_main.sh       main benchmark, 400 inferences/model, 0.8B -> 2B -> 4B -> 9B (4-bit)
  02_run_8x.sh         long-thinking runs (8x token caps, 20 samples/task); optional args select models
  03_run_mitigate.sh      mitigation experiment, 10 samples per task (run after 01/02); logs to W&B project macbook-llm-mitigation
  04_cleanup.sh        dry-run by default; --yes deletes tmp/, models/, wandb/ and the HF model cache
  common.sh            shared helpers (skip-if-finished, 9B conversion)
results/               per-sample records (.jsonl), summaries (.csv), comparison tables (kept)
logs/                  console logs (git-ignored)
```
All scripts are resumable and skip finished runs, so they can be re-run safely; use `caffeinate -ims scripts/01_run_main.sh > logs/main.log 2>&1 &` for long runs.

## Reproduce
```bash
# one-shot: scripts/01_run_main.sh && scripts/02_run_8x.sh && scripts/03_run_mitigate.sh   (or the manual steps below)
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install mlx-lm mlx-vlm wandb datasets huggingface_hub psutil pillow torch torchvision pandas
export WANDB_API_KEY=...        # runs log online to W&B project "macbook-llm-inference"
python bench.py --model Qwen/Qwen3.5-0.8B --tag 0.8B                 # 400 inferences
python bench.py --model Qwen/Qwen3.5-0.8B --tag 0.8B-think8x --think-only --cap-mult 8 --n 20
python -m mlx_vlm convert --hf-path Qwen/Qwen3.5-9B --mlx-path models/Qwen3.5-9B-4bit -q --q-bits 4 --q-group-size 64
python compare.py 0.8B 2B
```

## Caveats
- n=100 per cell gives wide intervals; trust paired tests and large gaps, not small differences.
- Token caps affect accuracy (truncated = wrong); both truncation rates and "accuracy among finished" are reported.
- Measurements reflect one laptop, plugged in, with other apps closed. Memory pressure on the 4B affects its numbers.
- Quantised 9B is not directly comparable to bf16 models (precision differs).
