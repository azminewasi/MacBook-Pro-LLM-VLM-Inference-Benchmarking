"""Mitigation experiment: re-run ONLY the thinking-on samples that failed to finish (truncated and wrong) with
Qwen's recommended sampling (top-k 20) + a presence penalty to curb repetition loops. Same prompts, same caps.

  python retry_truncated.py --model Qwen/Qwen3.5-0.8B --tag 0.8B            # retries failures in results/0.8B.jsonl
  python retry_truncated.py --model Qwen/Qwen3.5-0.8B --tag 0.8B-think8x    # 8x-cap run (cap-mult auto = 8)

Writes results/<tag>-retry.jsonl (resumable) + results/<tag>-retry_summary.csv and logs to W&B (online) as run
'qwen3.5-<tag>-retry'. The ORIGINAL results are never modified; retries are extra rows to compare against.
NOTE: the models must still be available (run BEFORE the cleanup of the HF cache / models/ folder)."""
import argparse, json, os, re, subprocess, threading, time
import psutil, mlx.core as mx
from datasets import load_dataset
from mlx_vlm import load, stream_generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--tag", required=True)
ap.add_argument("--project", default="macbook-llm-inference"); ap.add_argument("--cap-mult", type=int, default=None)
ap.add_argument("--temperature", type=float, default=0.6); ap.add_argument("--top-p", type=float, default=0.95)
ap.add_argument("--top-k", type=int, default=20); ap.add_argument("--presence-penalty", type=float, default=1.5)
ap.add_argument("--presence-context", type=int, default=512, help="how many recent tokens the presence penalty looks back over")
ap.add_argument("--repetition-penalty", type=float, default=None)
ap.add_argument("--include-correct-truncated", action="store_true", help="also retry truncated samples that were already scored correct")
ap.add_argument("--seed", type=int, default=1234); ap.add_argument("--dtype", default="bf16"); ap.add_argument("--max-samples", type=int, default=0, help="cap on total samples retried")
ap.add_argument("--max-per-task", type=int, default=0, help="cap per task (random, seeded pick of the failures); 0 = all")
args = ap.parse_args()
if args.cap_mult is None: args.cap_mult = 8 if "think8x" in args.tag else 1
SRC, OUT = f"results/{args.tag}.jsonl", f"results/{args.tag}-retry.jsonl"
os.makedirs("tmp", exist_ok=True); mx.random.seed(args.seed)

# ---------- same prompts / scoring as bench.py ----------
GSM_SUFFIX = "\nSolve step by step. End with a final line: 'The answer is: <number>'."
CAP = {"gsm8k_text": 3072, "ai2d_vision": 2048}
def strip_think(t): return t.split("</think>")[-1] if "</think>" in t else (None if "<think>" in t else t)
def score_gsm(text, gold):
    a = strip_think(text)
    if a is None: return 0, None
    nums = re.findall(r"-?\d[\d,]*\.?\d*", a.replace("$", ""))
    if not nums: return 0, None
    p = nums[-1].replace(",", "").rstrip(".")
    try: return int(abs(float(p) - float(gold)) < 1e-4), p
    except Exception: return 0, p
def score_ai2d(text, gold):
    a = strip_think(text)
    if a is None: return 0, None
    m = re.search(r"\b([1-4])\b", a); p = m.group(1) if m else None
    return int(p == gold), p
SCORER = {"gsm8k_text": score_gsm, "ai2d_vision": score_ai2d}

def get_sample(task, i, cache={}):
    if task == "gsm8k_text":
        if task not in cache: cache[task] = load_dataset("openai/gsm8k", "main", split="test")
        r = cache[task][i]; return dict(q=r["question"] + GSM_SUFFIX, img=None, gold=r["answer"].split("####")[-1].strip().replace(",", ""))
    if task not in cache: cache[task] = load_dataset("lmms-lab/ai2d", split="test")
    r = cache[task][i]; p = f"tmp/ai2d_{i}.jpg"
    if not os.path.exists(p):
        img = r["image"].convert("RGB"); img.thumbnail((896, 896)); img.save(p, quality=90)
    L = "1234"[: len(r["options"])]
    q = r["question"] + "\n" + "\n".join(f"{l}. {o}" for l, o in zip(L, r["options"])) + "\nAnswer with the option number only."
    return dict(q=q, img=p, gold=L[int(r["answer"])])

# ---------- system sampler (same as bench.py) ----------
def gpu_util():
    try:
        o = subprocess.run(["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"], capture_output=True, text=True, timeout=3).stdout
        m = re.search(r'"Device Utilization %"=(\d+)', o); return int(m.group(1)) if m else None
    except Exception: return None
class Sampler(threading.Thread):
    def __init__(self): super().__init__(daemon=True); self.stop = False; self.s = []
    def run(self):
        psutil.cpu_percent(None)
        while not self.stop:
            time.sleep(0.5)
            self.s.append(dict(cpu=psutil.cpu_percent(None), ram=psutil.virtual_memory().used / 2**30, swap=psutil.swap_memory().used / 2**30, gpu=gpu_util()))
def agg(s, k):
    v = [x[k] for x in s if x.get(k) is not None]; return (sum(v) / len(v), max(v)) if v else (None, None)
def thermal():
    o = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True).stdout
    m = re.search(r"CPU_Speed_Limit\s*=\s*(\d+)", o); return int(m.group(1)) if m else 100

# ---------- select samples to retry ----------
src = [json.loads(l) for l in open(SRC)]
todo = [r for r in src if r["thinking"] and r["truncated"] and (args.include_correct_truncated or not r["correct"])]
if args.max_per_task:
    import random
    rng = random.Random(args.seed); picked = []
    for task in sorted({r["task"] for r in todo}):
        pool = [r for r in todo if r["task"] == task]; rng.shuffle(pool); picked += sorted(pool[: args.max_per_task], key=lambda r: r["id"])
    todo = picked
if args.max_samples: todo = todo[: args.max_samples]
n_think = sum(1 for r in src if r["thinking"])
print(f"{args.tag}: {len(todo)} of {n_think} thinking-on samples to retry (caps x{args.cap_mult})", flush=True)

model, processor = load(args.model); cfg = load_config(args.model)
import wandb
run = wandb.init(project=args.project, name=f"qwen3.5-{args.tag}-retry", reinit=True,
    tags=[args.tag.split("-")[0], args.dtype, "retry", "mitigation", "topk20", "presence-penalty", "8x-thinking" if args.cap_mult > 1 else "1x-thinking"], config=dict(
    model=args.model, source_tag=args.tag, kind="retry-truncated", cap_mult=args.cap_mult, n_retry=len(todo), dtype=args.dtype,
    temperature=args.temperature, top_p=args.top_p, top_k=args.top_k, presence_penalty=args.presence_penalty,
    presence_context=args.presence_context, repetition_penalty=args.repetition_penalty, seed=args.seed, backend="mlx", mlx=mx.__version__,
    chip="Apple M5", ram_gb=16))
done = set()
if os.path.exists(OUT): done = {(json.loads(l)["task"], json.loads(l)["id"]) for l in open(OUT)}
fout = open(OUT, "a"); ok_run = 0; k = 0
for r in todo:
    if (r["task"], r["id"]) in done: continue
    s = get_sample(r["task"], r["id"]); k += 1
    prompt = apply_chat_template(processor, cfg, s["q"], num_images=1 if s["img"] else 0, enable_thinking=True)
    kw = dict(max_tokens=CAP[r["task"]] * args.cap_mult, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
              presence_penalty=args.presence_penalty, presence_context_size=args.presence_context)
    if args.repetition_penalty: kw["repetition_penalty"] = args.repetition_penalty
    mx.reset_peak_memory(); smp = Sampler(); smp.start(); t0 = time.perf_counter(); ttft = None; text = ""; last = None
    for g in stream_generate(model, processor, prompt, image=s["img"], **kw):
        if ttft is None: ttft = time.perf_counter() - t0
        text += g.text; last = g
    total = time.perf_counter() - t0; smp.stop = True; smp.join(timeout=2)
    ok, pred = SCORER[r["task"]](text, s["gold"]); ok_run += ok
    cpu, gpu, ram, sw = agg(smp.s, "cpu"), agg(smp.s, "gpu"), agg(smp.s, "ram"), agg(smp.s, "swap")
    rec = dict(model=r["model"], task=r["task"], thinking=True, id=r["id"], correct=ok, pred=pred, gold=s["gold"], response=text,
               ttft_s=ttft, total_s=total, prompt_tokens=last.prompt_tokens, gen_tokens=last.generation_tokens, prompt_tps=last.prompt_tps,
               decode_tps=last.generation_tps, finish=last.finish_reason, truncated=int(last.finish_reason == "length"),
               peak_mem_gb=mx.get_peak_memory() / 2**30, active_mem_gb=mx.get_active_memory() / 2**30, cpu_avg=cpu[0], cpu_max=cpu[1],
               gpu_util_avg=gpu[0], gpu_util_max=gpu[1], ram_used_gb_avg=ram[0], ram_used_gb_max=ram[1], swap_gb_max=sw[1],
               rss_gb=psutil.Process().memory_info().rss / 2**30, thermal_speed_limit=thermal(),
               orig_correct=r["correct"], orig_truncated=r["truncated"], orig_gen_tokens=r["gen_tokens"], rescued=int(ok and not r["correct"]))
    fout.write(json.dumps(rec) + "\n"); fout.flush()
    wandb.log({f"retry/{key}": v for key, v in rec.items() if isinstance(v, (int, float)) and v is not None} | {"retry/running_rescue_rate": ok_run / k})
    print(f"{args.tag} retry {k}/{len(todo) - len(done)} {r['task']} id={r['id']} ok={ok} finish={last.finish_reason} gen={last.generation_tokens} {last.generation_tps:.1f}tok/s rescued_so_far={ok_run}", flush=True)

# ---------- summary: before vs after ----------
import pandas as pd
new = pd.DataFrame([json.loads(l) for l in open(OUT)]) if os.path.exists(OUT) and os.path.getsize(OUT) else pd.DataFrame()
base = pd.DataFrame([r for r in src if r["thinking"]])
rows = []
for task, b in base.groupby("task"):
    n = len(b); n_new = new[new.task == task] if len(new) else new
    resc = int(n_new.rescued.sum()) if len(n_new) else 0
    rows.append(dict(tag=args.tag, task=task, n_total=n, orig_acc=b.correct.mean(), orig_trunc=b.truncated.mean(), n_retried=len(n_new),
                     rescued=resc, rescue_rate=resc / len(n_new) if len(n_new) else None,
                     retry_still_truncated=n_new.truncated.mean() if len(n_new) else None,
                     acc_after_retry=(b.correct.sum() + resc) / n, acc_gain=resc / n,
                     retry_mean_gen_tokens=n_new.gen_tokens.mean() if len(n_new) else None, retry_mean_latency_s=n_new.total_s.mean() if len(n_new) else None))
summ = pd.DataFrame(rows); summ.to_csv(f"results/{args.tag}-retry_summary.csv", index=False)
wandb.log({"retry_summary": wandb.Table(dataframe=summ)})
for _, x in summ.iterrows():
    for key, v in x.items():
        if isinstance(v, (int, float)) and v == v: run.summary[f"{x.task}/{key}"] = v
run.finish(); print(summ.T.to_string())
