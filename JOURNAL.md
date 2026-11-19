# Research journal: Mac LLM/VLM Inference Testing

A running, dated log of experiments, surprises, decisions and open questions. Newest entries at the bottom of the log.
Each entry: **what we tried → what happened → why we think so (and how sure we are) → what we decided**.
Times are local and approximate where marked `~`. Data: `results/` (per-sample `.jsonl`, summaries `.csv`); W&B projects `macbook-llm-inference` and `macbook-llm-mitigation`.

## Standing questions
| # | Question | Status |
|---|---|---|
| Q1 | How do Qwen3.5 0.8B/2B/4B/9B-4bit compare on GSM8K and AI2D, thinking off vs on, on a 16 GB M5? | answered (report.html) |
| Q2 | Is thinking's poor GSM8K accuracy a capability or a token-budget effect? | answered: budget (8x caps recover 4B/9B to 95% on 20 questions) |
| Q3 | Can top-k 20 + presence penalty rescue looping/truncated runs? | weakly supported: 20-35% (1x), 50-70% (8x), no control arm, tiny n |
| Q4 | Does batching change throughput on this Mac? | not measured |
| Q5 | Power draw and battery behaviour? | not measured |

## Log

### 2026-11-16 · Setup and design decisions
- Target: 4 models x (GSM8K + AI2D) x thinking off/on = 400 inferences/model, all metrics, W&B online. The LoRA arm of the plan was dropped in favour of thinking on/off. 9B in bf16 is 19 GB and does not fit, so it is converted to 4-bit.
- Early snags: a CLI flag that takes no value; AI2D scored 0 in the smoke test because lettered options were ambiguous (switched to numbered options 1-4); a battery-power metric was misleading on AC and was dropped.
- Known bias kept: **4B AI2D thinking-off** is cut off by the 32-token cap in 49% of samples (it explains before answering). We decided not to rerun.

### 2026-11-16 · Runs, stops and restarts
- 0.8B and 2B finished first; the run was stopped after the 2B so the laptop could cool, then resumed. The first 4B 8x attempt was started by a stop that took effect late and was killed after ~8 min (2 samples; kept in `logs/partial/`).
- 4B (8.5 GB in bf16) runs with ~2 GB of swap and no slowdown.
- Git: the old `.git` was removed and a fresh repo made; a stray `git add -A` committed a log, fixed with `git rm --cached`; pushes fail here for lack of credentials.

### 2026-11-16 22:24 · Estimate vs reality
- Estimated the remaining queue (4B 8x, 9B 8x, 1x and 8x mitigation) at ~15-16 h, finishing ~14:00.
- **Actual:** done by ~08:00 on 11-17 (~8-9 h). **Why we were off (hypothesis):** I over-counted samples that would run to the 24k-token cap and over-estimated how many mitigation samples there would be (only 2-16 per 8x set).

### 2026-11-17 ~08:00 · Mitigation results
- 1x sets (20 re-runs per model): rescued 6, 4, 6, 7 for 0.8B, 2B, 4B, 9B; 8x sets: 8/16, 5/7, 1/2, 2/3. Most 1x re-runs still hit the cap.
- **Caveat:** no control arm (a re-run with the original sampling settings); part of the rescue may be re-sampling luck. Not yet tested.

### 2026-11-17 ~08:10 · Cleanup
- Deleted tmp/, wandb/, the 9B cache (36 GB) and the 4-bit conversion, AI2D cache and the one-off scripts; kept results and the model caches for 0.8B-4B until post-training finishes.

### 2026-11-17 · Report, and a correction
- Built `results/report.html` (inline SVG, no dependencies). **Correction:** I had written the paired-test conclusions from memory and they were wrong; the actual tests show the 4B is significantly better than the 9B-4bit with thinking on (the 9B truncates more), and the 9B wins only on AI2D thinking-off (a format effect). Report and README were fixed; all claims now come from computed numbers.
- Also found: swap appeared in the 9B runs too (~2.3 GB with only 6.5 GB of MLX memory), so it is system-wide pressure, not model memory.
- Added a capability section (memory budget ≈6.5 GB OS + ≈9.5 GB usable; ~125 GB/s decode bandwidth; context adds <1 GB at 25k tokens; time per full benchmark) and restructured both reports as blog posts.

## Methodology notes
- n=100 per cell (±10 points), n=20 for 8x, 2-20 for mitigation; one machine, plugged in, apps closed; batch size 1.
- Loops are detected by zlib compression of the last 3,000 characters (<15%); a heuristic, not a human label.
- Decode-speed model: tokens/s ≈ 125 GB/s ÷ weight GB (within ~15% on the four measured models); larger-model rows in the report are extrapolations and are marked as such.

## Next / open
1. A control arm for the mitigation experiment (re-run with the original settings).
2. Batched generation throughput (Q4) and a powermetrics run for watts (Q5).
3. Cross-link with the post-training project (same models, same machine).

## 2026-11-18 (evening): cleanup
- Deleted the local `logs/` directory (earlier entries that mention files under `logs/` refer to files that no longer exist), local W&B caches and the Hugging Face model cache. Kept: per-inference results, summaries, reports and this journal.
