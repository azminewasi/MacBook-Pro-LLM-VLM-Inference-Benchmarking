# Mac LLM/VLM Inference Testing

How far can a single 16 GB MacBook go at running modern LLMs and VLMs locally? This project benchmarks
**Qwen3.5 (0.8B, 2B, 4B, 9B)** on an **Apple M5 MacBook** with [MLX](https://github.com/ml-explore/mlx), measuring
quality, speed, memory and system load for every single inference, and logging everything to Weights & Biases.

> **Status:** all runs are complete: 4 models x 400 inferences, the 8x long-thinking runs, and the mitigation experiment. The full analysis with plots is in **[`results/report.html`](results/report.html)** (open it in a browser; regenerate with `python analysis/make_report.py`).

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

## Results (100 samples per cell; about ±10 points at 95% confidence)
| Model | GSM8K off | GSM8K on | AI2D off | AI2D on | Decode tok/s | Peak MLX mem |
|---|---|---|---|---|---|---|
| 0.8B (bf16) | 54% | 22% | 41% | 48% | 71-73 | 2.5 GB |
| 2B (bf16) | 77% | 65% | 46% | 73% | 30-31 | 5.1 GB |
| 4B (bf16) | 85% | 84% | 45%\* | 85% | 14 | 9.4 GB |
| 9B (4-bit) | 88% | 70% | 82% | 76% | 26 | 6.5 GB |

\*4B AI2D thinking-off is cut off by the 32-token cap in 49% of samples (it explains before answering), so it understates the model.

**Long thinking (8x token caps, first 20 questions, thinking on)**
| Model | GSM8K 1x -> 8x | AI2D 1x -> 8x | GSM8K still capped at 8x |
|---|---|---|---|
| 0.8B | 15% -> 35% | 40% -> 60% | 80% |
| 2B | 50% -> 80% | 60% -> 70% | 25% |
| 4B | 70% -> 95% | 85% -> 85% | 15% |
| 9B (4-bit) | 65% -> 95% | 75% -> 85% | 0% |

**Mitigation** (re-run truncated+wrong thinking samples with top-k 20 + presence penalty 1.5): rescues 20-35% of the re-run samples at 1x caps and 50-70% at 8x; there is no control arm and few samples, so treat as indicative.

**Findings**
- Quality rises with size (GSM8K without thinking 54 -> 77 -> 85 -> 88%). The 9B at 4-bit is better *and* faster (26 tok/s) than the 4B in bf16 (14 tok/s), so on 16 GB quantising a bigger model beats running a smaller one at full precision.
- Thinking helps vision for the smaller models but looks harmful on GSM8K at the 3,072-token cap because up to 94% of thinking runs never finish. With 8x caps thinking becomes the best mode (4B and 9B reach 95% on GSM8K).
- Remaining truncations at 8x are mostly repetition loops for 0.8B-4B; the 9B rarely loops. Wrong answers use about 2x the tokens of correct ones.
- Decode speed is memory-bandwidth bound: decode tok/s x weight size is ~110-145 GB/s for every model.
- The 4B in bf16 is the practical memory limit (9.4 GB peak; macOS used up to ~2.4 GB of swap without slowing decode). GPU utilisation was 97-99%, CPU 6-12%, and the thermal speed limit stayed at 100% (no throttling).

Per-sample data is in `results/` (`<model>.jsonl`, `<model>_summary.csv`, `*-think8x*`, `*-mitigate*`, `compare_*.csv`); the report ties everything together.

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
analysis/make_report.py  builds results/report.html (inline-SVG charts + tables, no extra dependencies)
results/               per-sample records (.jsonl), summaries (.csv), comparison tables, report.html (kept)
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
