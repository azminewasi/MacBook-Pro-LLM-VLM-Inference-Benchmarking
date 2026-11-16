"""Benchmark Qwen3.5 on Apple Silicon (MLX): GSM8K (text) + AI2D (vision), thinking on/off.
Usage: python bench.py --model Qwen/Qwen3.5-0.8B --tag 0.8B [--n 100]
Per-sample records go to results/<tag>.jsonl (resumable) and to wandb (online)."""
import argparse, json, os, re, subprocess, threading, time, io
import psutil, mlx.core as mx
from datasets import load_dataset
from mlx_vlm import load, stream_generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--tag", required=True)
ap.add_argument("--n", type=int, default=100); ap.add_argument("--project", default="macbook-llm-inference")
ap.add_argument("--dtype", default="bf16")
ap.add_argument("--think-only", action="store_true"); ap.add_argument("--cap-mult", type=int, default=1)
args = ap.parse_args()
os.makedirs("results", exist_ok=True); os.makedirs("tmp", exist_ok=True)
OUT = f"results/{args.tag}.jsonl"

# ---------- system sampler (CPU, RAM, swap, GPU util, battery power) ----------
def ioreg_stats():
    d = {}
    try:
        o = subprocess.run(["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"], capture_output=True, text=True, timeout=3).stdout
        m = re.search(r'"Device Utilization %"=(\d+)', o); d["gpu_util"] = int(m.group(1)) if m else None
    except Exception: d["gpu_util"] = None
    try:
        o = subprocess.run(["ioreg", "-rn", "AppleSmartBattery"], capture_output=True, text=True, timeout=3).stdout
        a = re.search(r'"InstantAmperage" = (\d+)', o); v = re.search(r'"Voltage" = (\d+)', o)
        if a and v:
            amp = int(a.group(1)); amp = amp - 2**64 if amp > 2**63 else amp  # signed
            d["batt_mw"] = amp * int(v.group(1)) / 1000.0  # negative = discharging
    except Exception: pass
    return d

class Sampler(threading.Thread):
    def __init__(self): super().__init__(daemon=True); self.stop = False; self.s = []
    def run(self):
        psutil.cpu_percent(None)
        while not self.stop:
            time.sleep(0.5)
            r = {"cpu": psutil.cpu_percent(None), "ram_used_gb": psutil.virtual_memory().used / 2**30,
                 "swap_gb": psutil.swap_memory().used / 2**30}
            r.update(ioreg_stats()); self.s.append(r)
def agg(s, k):
    v = [x[k] for x in s if x.get(k) is not None]
    return (sum(v) / len(v), max(v)) if v else (None, None)
def thermal():
    o = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True).stdout
    m = re.search(r"CPU_Speed_Limit\s*=\s*(\d+)", o)
    return int(m.group(1)) if m else 100

# ---------- data ----------
def load_gsm(n):
    ds = load_dataset("openai/gsm8k", "main", split="test").select(range(n))
    return [dict(id=i, q=r["question"], gold=r["answer"].split("####")[-1].strip().replace(",", "")) for i, r in enumerate(ds)]
def load_ai2d(n):
    ds = load_dataset("lmms-lab/ai2d", split="test").select(range(n)); out = []
    for i, r in enumerate(ds):
        img = r["image"].convert("RGB"); img.thumbnail((896, 896)); p = f"tmp/ai2d_{i}.jpg"; img.save(p, quality=90)
        L = "1234"[: len(r["options"])]
        q = r["question"] + "\n" + "\n".join(f"{l}. {o}" for l, o in zip(L, r["options"])) + "\nAnswer with the option number only."
        out.append(dict(id=i, q=q, img=p, gold=L[int(r["answer"])]))
    return out

def strip_think(t):
    return t.split("</think>")[-1] if "</think>" in t else (None if "<think>" in t else t)
def score_gsm(text, gold):
    a = strip_think(text)
    if a is None: return 0, None
    nums = re.findall(r"-?\d[\d,]*\.?\d*", a.replace("$", ""))
    if not nums: return 0, None
    p = nums[-1].replace(",", "").rstrip(".")
    try: return int(abs(float(p) - float(gold)) < 1e-4), p
    except: return 0, p
def score_ai2d(text, gold):
    a = strip_think(text)
    if a is None: return 0, None
    m = re.search(r"\b([1-4])\b", a)
    p = m.group(1) if m else None
    return int(p == gold), p

# ---------- model ----------
mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"]) if hasattr(mx, "set_wired_limit") else None
t0 = time.time(); model, processor = load(args.model); cfg = load_config(args.model); load_s = time.time() - t0
load_mem = mx.get_active_memory() / 2**30
print(f"loaded {args.model} in {load_s:.1f}s, {load_mem:.2f} GB", flush=True)

def run_one(prompt_text, img, think, max_tokens):
    msgs = [{"role": "user", "content": prompt_text}]
    prompt = apply_chat_template(processor, cfg, prompt_text, num_images=1 if img else 0, enable_thinking=think)
    kw = dict(max_tokens=max_tokens, temperature=0.6 if think else 0.0, top_p=0.95 if think else 1.0)
    mx.reset_peak_memory(); smp = Sampler(); smp.start()
    t_start = time.perf_counter(); ttft = None; text = ""; last = None
    for r in stream_generate(model, processor, prompt, image=img, **kw):
        if ttft is None: ttft = time.perf_counter() - t_start
        text += r.text; last = r
    total = time.perf_counter() - t_start; smp.stop = True; smp.join(timeout=2)
    cpu = agg(smp.s, "cpu"); gpu = agg(smp.s, "gpu_util"); ram = agg(smp.s, "ram_used_gb"); sw = agg(smp.s, "swap_gb")
    return text, dict(
        ttft_s=ttft, total_s=total, prompt_tokens=last.prompt_tokens, gen_tokens=last.generation_tokens,
        prompt_tps=last.prompt_tps, decode_tps=last.generation_tps, finish=last.finish_reason,
        truncated=int(last.finish_reason == "length"),
        peak_mem_gb=mx.get_peak_memory() / 2**30, active_mem_gb=mx.get_active_memory() / 2**30,
        cpu_avg=cpu[0], cpu_max=cpu[1], gpu_util_avg=gpu[0], gpu_util_max=gpu[1],
        ram_used_gb_avg=ram[0], ram_used_gb_max=ram[1], swap_gb_max=sw[1],
        rss_gb=psutil.Process().memory_info().rss / 2**30, thermal_speed_limit=thermal())

# ---------- run ----------
import wandb
run = wandb.init(project=args.project, name=f"qwen3.5-{args.tag}", reinit=True, config=dict(
    model=args.model, tag=args.tag, n=args.n, dtype=args.dtype, cap_mult=args.cap_mult, backend="mlx", mlx=mx.__version__,
    chip="Apple M5", ram_gb=16, load_time_s=load_s, load_mem_gb=load_mem))
wandb.define_metric("sample_idx")
done = set()
if os.path.exists(OUT):
    for l in open(OUT): d = json.loads(l); done.add((d["task"], d["thinking"], d["id"]))
fout = open(OUT, "a")
tasks = {"gsm8k_text": (load_gsm(args.n), None, score_gsm), "ai2d_vision": (load_ai2d(args.n), True, score_ai2d)}
CAP = {("gsm8k_text", False): 512, ("gsm8k_text", True): 3072, ("ai2d_vision", False): 32, ("ai2d_vision", True): 2048}
GSM_SUFFIX = "\nSolve step by step. End with a final line: 'The answer is: <number>'."
rows = []
for task, (data, has_img, scorer) in tasks.items():
    for think in ((True,) if args.think_only else (False, True)):
        step = 0; cum = 0
        for s in data:
            step += 1
            if (task, think, s["id"]) in done: continue
            q = s["q"] + (GSM_SUFFIX if task == "gsm8k_text" else "")
            text, m = run_one(q, s.get("img"), think, CAP[(task, think)] * args.cap_mult)
            ok, pred = scorer(text, s["gold"]); cum += ok
            rec = dict(model=args.tag, task=task, thinking=think, id=s["id"], correct=ok, pred=pred, gold=s["gold"], response=text, **m)
            fout.write(json.dumps(rec) + "\n"); fout.flush()
            wandb.log({f"{task}/think={think}/{k}": v for k, v in m.items() if isinstance(v, (int, float)) and v is not None}
                      | {f"{task}/think={think}/correct": ok, f"{task}/think={think}/running_acc": cum / step})
            print(f"{args.tag} {task} think={think} {step}/{len(data)} ok={ok} gen={m['gen_tokens']} {m['decode_tps']:.1f}tok/s ttft={m['ttft_s']:.2f}s acc={cum/step:.2f}", flush=True)

# ---------- summary ----------
import pandas as pd
df = pd.DataFrame([json.loads(l) for l in open(OUT)])
num = df.select_dtypes("number").columns.drop(["id"])
summ = df.groupby(["task", "thinking"])[list(num)].mean().rename(columns={"correct": "accuracy"})
summ["n"] = df.groupby(["task", "thinking"]).size()
summ.to_csv(f"results/{args.tag}_summary.csv")
wandb.log({"summary_table": wandb.Table(dataframe=summ.reset_index()),
           "samples_table": wandb.Table(dataframe=df.drop(columns=["response"]).assign(response=df["response"].str[:2000]))})
for (t, th), r in summ.iterrows():
    for k, v in r.items(): run.summary[f"{t}/think={th}/{k}"] = v
run.finish()
print(summ.T.to_string())
