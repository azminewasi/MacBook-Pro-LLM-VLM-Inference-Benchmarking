"""Compare models from results/*.jsonl. Usage: python compare.py 0.8B 2B"""
import sys, json, numpy as np, pandas as pd
tags = sys.argv[1:]
df = pd.concat([pd.DataFrame([json.loads(l) for l in open(f"results/{t}.jsonl")]) for t in tags])
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z*z/n; c = p + z*z/(2*n); w = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)); return (c-w)/d, (c+w)/d
rows = []
for (t, task, th), g in df.groupby(["model", "task", "thinking"], sort=False):
    k, n = int(g.correct.sum()), len(g); lo, hi = wilson(k, n)
    done = g[g.truncated == 0]
    rows.append(dict(model=t, task=task, think=th, acc=k/n, ci95=f"{lo:.2f}-{hi:.2f}", acc_finished=done.correct.mean(), trunc=g.truncated.mean(),
        gen_tok=g.gen_tokens.mean(), dec_tps=g.decode_tps.mean(), prefill_tps=g.prompt_tps.mean(), ttft=g.ttft_s.mean(), lat=g.total_s.mean(),
        lat_p95=g.total_s.quantile(.95), sec_per_correct=g.total_s.sum()/max(k,1), tok_per_correct=g.gen_tokens.sum()/max(k,1),
        peak_gb=g.peak_mem_gb.max(), cpu=g.cpu_avg.mean(), gpu=g.gpu_util_avg.mean(), ram=g.ram_used_gb_max.max(), swap=g.swap_gb_max.max(), therm=g.thermal_speed_limit.min()))
R = pd.DataFrame(rows); pd.set_option("display.width", 250); pd.set_option("display.float_format", "{:.2f}".format)
print(R.to_string(index=False))
a, b = tags[0], tags[1]; print(f"\nPaired comparison {a} -> {b} (same 100 questions):")
for (task, th), g in df.groupby(["task", "thinking"]):
    p = g.pivot(index="id", columns="model", values="correct")
    only_b, only_a = int(((p[b] == 1) & (p[a] == 0)).sum()), int(((p[a] == 1) & (p[b] == 0)).sum())
    from math import comb
    n = only_a + only_b; pv = min(1, 2*sum(comb(n, i) for i in range(0, min(only_a, only_b)+1)) / 2**n) if n else 1
    print(f"  {task:12s} think={th!s:5s}: {b} only right={only_b:2d}, {a} only right={only_a:2d}, both={int(((p[a]==1)&(p[b]==1)).sum()):2d}, neither={int(((p[a]==0)&(p[b]==0)).sum()):2d}  sign-test p={pv:.4f}")
R.to_csv(f"results/compare_{'_vs_'.join(tags)}.csv", index=False)
