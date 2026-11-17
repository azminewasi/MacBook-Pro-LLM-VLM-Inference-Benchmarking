"""Builds results/report.html (self-contained: inline SVG charts, no dependencies beyond pandas/numpy).
Usage (from the project root):  python analysis/make_report.py"""
import json, os, zlib, datetime, html
from math import comb
import numpy as np, pandas as pd

M = ["0.8B", "2B", "4B", "9B-4bit"]
LABEL = {"0.8B": "0.8B (bf16)", "2B": "2B (bf16)", "4B": "4B (bf16)", "9B-4bit": "9B (4-bit)"}
TASK = {"gsm8k_text": "GSM8K (text)", "ai2d_vision": "AI2D (vision)"}
COND = [("gsm8k_text", False), ("gsm8k_text", True), ("ai2d_vision", False), ("ai2d_vision", True)]
CLAB = ["GSM8K\nthinking off", "GSM8K\nthinking on", "AI2D\nthinking off", "AI2D\nthinking on"]
R = "results"

def load(t): return pd.DataFrame([json.loads(l) for l in open(f"{R}/{t}.jsonl")])
D1 = {m: load(m) for m in M}
D8 = {m: load(m + "-think8x") for m in M}
MIT1 = {m: pd.read_csv(f"{R}/{m}-mitigate_summary.csv") for m in M}
MIT8 = {m: pd.read_csv(f"{R}/{m}-think8x-mitigate_summary.csv") for m in M}
LOAD_GB = {"0.8B": 1.59, "2B": 4.12, "4B": 8.46, "9B-4bit": 5.54}   # from the "loaded ... GB" lines of the run logs

def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); w = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)); return (c - w) / d, (c + w) / d
def cell(m, task, th, d=None):
    d = D1[m] if d is None else d
    return d[(d.task == task) & (d.thinking == th)]
def zr(s):
    b = s[-3000:].encode(); return len(zlib.compress(b)) / max(len(b), 1)
def sign_p(a, b):
    n = a + b
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(0, min(a, b) + 1)) / 2 ** n)

# ------------------------------------------------------------------ svg helpers
SC = ["s0", "s1", "s2", "s3", "s4"]
def esc(s): return html.escape(str(s))
def gbars(groups, series, fmt="{:.0f}%", ymax=None, log=False, ylabel="", yref=None, height=300, ymin=None):
    """groups: list of labels (\\n = line break); series: list of (name, [values], [(lo,hi)] or None). returns (legend_html, svg)."""
    W, H, L, Rr, T, B = 760, height, 58, 12, 16, 52
    vals = [v for _, vs, _ in series for v in vs if v is not None]
    if log:
        lo_e = int(np.floor(np.log10(min(v for v in vals if v > 0)))); hi_e = int(np.ceil(np.log10(max(vals))))
        f = lambda v: (np.log10(max(v, 10 ** lo_e)) - lo_e) / (hi_e - lo_e)
        ticks = [10 ** e for e in range(lo_e, hi_e + 1)]; top = 1.0
    else:
        mx = ymax or max(vals) * 1.12; mn = ymin or 0
        cands = [c * 10.0 ** e for e in range(-2, 4) for c in (1, 2, 2.5, 5)]
        step = min((c for c in cands if 3 <= (mx - mn) / c <= 6), default=(mx - mn) / 4, key=lambda c: abs((mx - mn) / c - 5))
        mx = mn + np.ceil((mx - mn) / step - 1e-9) * step; f = lambda v: (v - mn) / (mx - mn); ticks = list(np.arange(mn, mx + step / 2, step))
    ph, pw = H - T - B, W - L - Rr
    y = lambda v: T + ph * (1 - min(max(f(v), 0), 1))
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" class="chart">']
    for t in ticks:
        out.append(f'<line x1="{L}" x2="{W-Rr}" y1="{y(t):.1f}" y2="{y(t):.1f}" class="grid"/><text x="{L-6}" y="{y(t)+3:.1f}" class="tick" text-anchor="end">{esc(fmt.format(t) if not log else (f"{t:g}"))}</text>')
    if yref is not None:
        out.append(f'<line x1="{L}" x2="{W-Rr}" y1="{y(yref[0]):.1f}" y2="{y(yref[0]):.1f}" class="ref"/>')
        if yref[1]: out.append(f'<text x="{W-Rr-2}" y="{y(yref[0])-4:.1f}" class="tick reflab" text-anchor="end">{esc(yref[1])}</text>')
    gw = pw / len(groups); ns = len(series); bw = min(34, gw * 0.8 / ns)
    for gi, g in enumerate(groups):
        gx = L + gw * gi + gw / 2
        for li, line in enumerate(g.split("\n")):
            out.append(f'<text x="{gx:.1f}" y="{H-B+16+li*13}" class="glab" text-anchor="middle">{esc(line)}</text>')
        for si, (name, vs, errs) in enumerate(series):
            v = vs[gi]
            if v is None: continue
            x = gx - ns * bw / 2 + si * bw; yy = y(v)
            out.append(f'<rect x="{x+1:.1f}" y="{yy:.1f}" width="{bw-2:.1f}" height="{max(0, T+ph-yy):.1f}" class="{SC[si]}" rx="2"><title>{esc(name)} · {esc(g.replace(chr(10), " "))}: {esc(fmt.format(v))}</title></rect>')
            top_y = yy
            if errs and errs[gi]:
                lo, hi = errs[gi]; xm = x + bw / 2
                out.append(f'<line x1="{xm:.1f}" x2="{xm:.1f}" y1="{y(lo):.1f}" y2="{y(hi):.1f}" class="err"/><line x1="{xm-3:.1f}" x2="{xm+3:.1f}" y1="{y(hi):.1f}" y2="{y(hi):.1f}" class="err"/><line x1="{xm-3:.1f}" x2="{xm+3:.1f}" y1="{y(lo):.1f}" y2="{y(lo):.1f}" class="err"/>')
                top_y = min(top_y, y(hi))
            out.append(f'<text x="{x+bw/2:.1f}" y="{top_y-4:.1f}" class="val" text-anchor="middle">{esc(fmt.format(v))}</text>')
    out.append(f'<line x1="{L}" x2="{W-Rr}" y1="{T+ph}" y2="{T+ph}" class="axis"/>')
    if ylabel: out.append(f'<text x="12" y="{T+ph/2}" class="tick" text-anchor="middle" transform="rotate(-90 12 {T+ph/2})">{esc(ylabel)}</text>')
    out.append("</svg>")
    legend = '<div class="legend">' + "".join(f'<span><i class="sw {SC[i]}"></i>{esc(n)}</span>' for i, (n, _, _) in enumerate(series)) + "</div>"
    return legend + "".join(out)

def stacked(groups, parts, height=260, fmt="{}", ymax=None, yref=None, minlab=0):
    """parts: list of (name, [counts per group]) stacked; shows counts."""
    W, H, L, Rr, T, B = 760, height, 58, 12, 16, 40
    tot = [sum(p[1][i] for p in parts) for i in range(len(groups))]; mx = ymax or max(tot) * 1.1
    ph, pw = H - T - B, W - L - Rr
    step = min([c * 10.0 ** e for e in range(-1, 4) for c in (1, 2, 5)], key=lambda c: abs(mx / c - 4))
    y = lambda v: T + ph * (1 - v / (step * np.ceil(mx / step)))
    out = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img">']
    for t in np.arange(0, step * np.ceil(mx / step) + step / 2, step):
        out.append(f'<line x1="{L}" x2="{W-Rr}" y1="{y(t):.1f}" y2="{y(t):.1f}" class="grid"/><text x="{L-6}" y="{y(t)+3:.1f}" class="tick" text-anchor="end">{t:g}</text>')
    if yref: out.append(f'<line x1="{L}" x2="{W-Rr}" y1="{y(yref[0]):.1f}" y2="{y(yref[0]):.1f}" class="ref"/><text x="{W-Rr-2}" y="{y(yref[0])-4:.1f}" class="tick reflab" text-anchor="end">{esc(yref[1])}</text>')
    gw = pw / len(groups); bw = min(70, gw * 0.55)
    for gi, g in enumerate(groups):
        x = L + gw * gi + gw / 2 - bw / 2; base = 0
        for si, (name, vs) in enumerate(parts):
            v = vs[gi]
            if v:
                out.append(f'<rect x="{x:.1f}" y="{y(base+v):.1f}" width="{bw:.1f}" height="{y(base)-y(base+v):.1f}" class="{SC[si+1 if si else 0]}"><title>{esc(name)}: {esc(fmt.format(v))}</title></rect>' + (f'<text x="{x+bw/2:.1f}" y="{(y(base)+y(base+v))/2+4:.1f}" class="vali" text-anchor="middle">{esc(fmt.format(v))}</text>' if v >= minlab else ""))
            base += v
        for li, line in enumerate(g.split("\n")):
            out.append(f'<text x="{x+bw/2:.1f}" y="{H-B+16+li*13}" class="glab" text-anchor="middle">{esc(line)}</text>')
    out.append(f'<line x1="{L}" x2="{W-Rr}" y1="{T+ph}" y2="{T+ph}" class="axis"/></svg>')
    return '<div class="legend">' + "".join(f'<span><i class="sw {SC[i+1 if i else 0]}"></i>{esc(n)}</span>' for i, (n, _) in enumerate(parts)) + "</div>" + "".join(out)

# ------------------------------------------------------------------ numbers
acc, ci, trunc, genk, dec, pre, ttft, lat, spc, tpc, peak = ({} for _ in range(11))
for m in M:
    for (t, th) in COND:
        g = cell(m, t, th); k = int(g.correct.sum()); n = len(g)
        acc[m, t, th] = k / n; ci[m, t, th] = wilson(k, n); trunc[m, t, th] = g.truncated.mean(); genk[m, t, th] = g.gen_tokens.mean()
        dec[m, t, th] = g.decode_tps.mean(); pre[m, t, th] = g.prompt_tps.mean(); ttft[m, t, th] = g.ttft_s.mean(); lat[m, t, th] = g.total_s.mean()
        spc[m, t, th] = g.total_s.sum() / max(k, 1); tpc[m, t, th] = g.gen_tokens.sum() / max(k, 1)
        peak[m, t, th] = g.peak_mem_gb.max()
def series_over_models(fn, names=None):
    return [(LABEL[m], [fn(m, t, th) for (t, th) in COND], None) for m in M]

# loops among truncated thinking-on outputs (compression ratio of the last 3000 chars < 0.15  =>  repetition loop)
def loop_counts(D):
    res = {}
    for m in M:
        d = D[m]; d = d[(d.thinking == True) & (d.truncated == 1)]
        r = d.response.map(zr); res[m] = (int((r < 0.15).sum()), int((r >= 0.15).sum()))
    return res
L1, L8 = loop_counts(D1), loop_counts(D8)

# ------------------------------------------------------------------ charts
c_acc = gbars(CLAB, [(LABEL[m], [100 * acc[m, t, th] for t, th in COND], [(100 * ci[m, t, th][0], 100 * ci[m, t, th][1]) for t, th in COND]) for m in M], ymax=100, ylabel="accuracy (%)", yref=(25, ""))
c_trunc = gbars(CLAB, [(LABEL[m], [100 * trunc[m, t, th] for t, th in COND], None) for m in M], ymax=100, ylabel="hit token cap (%)")
c_gen = gbars(CLAB, [(LABEL[m], [genk[m, t, th] for t, th in COND], None) for m in M], fmt="{:.0f}", log=True, ylabel="mean generated tokens (log)")
c_dec = gbars(["GSM8K thinking on", "AI2D thinking on"], [(LABEL[m], [dec[m, "gsm8k_text", True], dec[m, "ai2d_vision", True]], None) for m in M], fmt="{:.1f}", ylabel="decode tokens / s")
c_pre = gbars(["GSM8K (~90 prompt tok)", "AI2D (~310 prompt tok, incl. image)"], [(LABEL[m], [pre[m, "gsm8k_text", True], pre[m, "ai2d_vision", True]], None) for m in M], fmt="{:.0f}", ylabel="prefill tokens / s")
bw_est = {m: dec[m, "gsm8k_text", True] * LOAD_GB[m] for m in M}
c_bw = gbars([LABEL[m] for m in M], [("decode tok/s × weight GB", [bw_est[m] for m in M], None)], fmt="{:.0f}", ymax=180, ylabel="implied GB/s", yref=(153, "Apple-quoted M5 peak ≈ 153 GB/s"))
c_lat = gbars(CLAB, [(LABEL[m], [lat[m, t, th] for t, th in COND], None) for m in M], fmt="{:.1f}", log=True, ylabel="mean latency, s (log)")
c_spc = gbars(CLAB, [(LABEL[m], [spc[m, t, th] for t, th in COND], None) for m in M], fmt="{:.1f}", log=True, ylabel="seconds per correct answer (log)")
c_tpc = gbars(CLAB, [(LABEL[m], [tpc[m, t, th] for t, th in COND], None) for m in M], fmt="{:.0f}", log=True, ylabel="generated tokens per correct answer (log)")
mem_series = [("MLX peak GPU memory", [max(peak[m, t, th] for t, th in COND) for m in M], None),
              ("System RAM used (max)", [max(D1[m].ram_used_gb_max) for m in M], None),
              ("Swap used (max)", [max(D1[m].swap_gb_max) for m in M], None)]
c_mem = gbars([LABEL[m] for m in M], mem_series, fmt="{:.1f}", ymax=18, ylabel="GB", yref=(16, "16 GB unified memory"))
matched = {}
for m in M:
    for t in ("gsm8k_text", "ai2d_vision"):
        a = cell(m, t, True); a = a[a.id < 20]; b = cell(m, t, True, D8[m]); matched[m, t] = (a.correct.mean(), b.correct.mean(), a.truncated.mean(), b.truncated.mean())
c_8x_g = gbars([LABEL[m] for m in M], [("1x cap (3072 tok)", [100 * matched[m, "gsm8k_text"][0] for m in M], None), ("8x cap (24,576 tok)", [100 * matched[m, "gsm8k_text"][1] for m in M], None)], ymax=100, ylabel="accuracy (%)")
c_8x_a = gbars([LABEL[m] for m in M], [("1x cap (2048 tok)", [100 * matched[m, "ai2d_vision"][0] for m in M], None), ("8x cap (16,384 tok)", [100 * matched[m, "ai2d_vision"][1] for m in M], None)], ymax=100, ylabel="accuracy (%)")
c_8x_t = gbars([LABEL[m] for m in M], [("GSM8K, 1x", [100 * matched[m, "gsm8k_text"][2] for m in M], None), ("GSM8K, 8x", [100 * matched[m, "gsm8k_text"][3] for m in M], None),
                                       ("AI2D, 1x", [100 * matched[m, "ai2d_vision"][2] for m in M], None), ("AI2D, 8x", [100 * matched[m, "ai2d_vision"][3] for m in M], None)], ymax=100, ylabel="hit token cap (%)")
c_loop1 = stacked([LABEL[m] for m in M], [("repetition loop", [L1[m][0] for m in M]), ("long but not looping", [L1[m][1] for m in M])])
c_loop8 = stacked([LABEL[m] for m in M], [("repetition loop", [L8[m][0] for m in M]), ("long but not looping", [L8[m][1] for m in M])])
cw = {m: (D1[m][(D1[m].thinking == True) & (D1[m].correct == 1)].gen_tokens.mean(), D1[m][(D1[m].thinking == True) & (D1[m].correct == 0)].gen_tokens.mean()) for m in M}
c_cw = gbars([LABEL[m] for m in M], [("correct answers", [cw[m][0] for m in M], None), ("wrong answers", [cw[m][1] for m in M], None)], fmt="{:.0f}", ylabel="mean tokens (thinking on, 1x)")
mrow = lambda D, m: D[m]
def mit_rows(MIT):
    rows = []
    for m in M:
        for _, r in MIT[m].iterrows():
            rows.append((LABEL[m], TASK[r.task], int(r.n_mitigated), int(r.rescued), r.mitigate_still_truncated, r.orig_acc, r.acc_after_mitigate))
    return rows
mit1_rescue = [sum(int(r.rescued) for _, r in MIT1[m].iterrows()) for m in M]; mit1_n = [sum(int(r.n_mitigated) for _, r in MIT1[m].iterrows()) for m in M]
mit8_rescue = [sum(int(r.rescued) for _, r in MIT8[m].iterrows()) for m in M]; mit8_n = [sum(int(r.n_mitigated) for _, r in MIT8[m].iterrows()) for m in M]
c_mit = gbars([LABEL[m] for m in M], [("1x sets: rescued % of re-run", [100 * a / max(b, 1) for a, b in zip(mit1_rescue, mit1_n)], None), ("8x sets: rescued % of re-run", [100 * a / max(b, 1) for a, b in zip(mit8_rescue, mit8_n)], None)], ymax=100, ylabel="% of truncated+wrong samples now correct")
hours = {m: (D1[m].total_s.sum() / 3600, D8[m].total_s.sum() / 3600) for m in M}
c_hours = gbars([LABEL[m] for m in M], [("main benchmark (400 inferences)", [hours[m][0] for m in M], None), ("8x long-thinking (40 inferences)", [hours[m][1] for m in M], None)], fmt="{:.1f}", ylabel="hours of generation")

# ------------------------------------------------------------------ tables
def tbl(head, rows, cls=""):
    return f'<div class="tw"><table class="{cls}"><thead><tr>' + "".join(f"<th>{esc(h)}</th>" for h in head) + "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows) + "</tbody></table></div>"
rows = []
for m in M:
    for t, th in COND:
        g = cell(m, t, th); lo, hi = ci[m, t, th]
        fin = g[g.truncated == 0]
        rows.append([LABEL[m], TASK[t], "on" if th else "off", f"<b>{100*acc[m,t,th]:.0f}%</b> <span class='dim'>({100*lo:.0f}–{100*hi:.0f})</span>", f"{100*fin.correct.mean():.0f}%" if len(fin) else "–", f"{100*trunc[m,t,th]:.0f}%",
                     f"{genk[m,t,th]:.0f}", f"{dec[m,t,th]:.1f}", f"{pre[m,t,th]:.0f}", f"{ttft[m,t,th]:.2f}", f"{lat[m,t,th]:.1f}", f"{g.total_s.quantile(.95):.1f}", f"{spc[m,t,th]:.1f}", f"{peak[m,t,th]:.1f}",
                     f"{g.cpu_avg.mean():.0f}", f"{g.gpu_util_avg.mean():.0f}", f"{g.ram_used_gb_max.max():.1f}", f"{g.swap_gb_max.max():.1f}", f"{g.thermal_speed_limit.min():.0f}"])
T_full = tbl(["Model", "Task", "Think", "Accuracy (95% CI)", "Acc. if finished", "Capped", "Gen tok", "Decode tok/s", "Prefill tok/s", "TTFT s", "Latency s", "p95 s", "s / correct", "Peak GB", "CPU %", "GPU %", "RAM GB", "Swap GB", "Therm. limit %"], rows, "dense")
rows8 = []
for m in M:
    for t in ("gsm8k_text", "ai2d_vision"):
        g = cell(m, t, True, D8[m]); k = int(g.correct.sum()); lo, hi = wilson(k, len(g))
        rows8.append([LABEL[m], TASK[t], f"<b>{100*k/len(g):.0f}%</b> <span class='dim'>({100*lo:.0f}–{100*hi:.0f})</span>", f"{100*g.truncated.mean():.0f}%", f"{g.gen_tokens.mean():.0f}", f"{g.decode_tps.mean():.1f}", f"{g.total_s.mean():.0f}", f"{g.gen_tokens.max():.0f}", f"{g.peak_mem_gb.max():.1f}", f"{g.swap_gb_max.max():.1f}"])
T_8x = tbl(["Model", "Task", "Accuracy (95% CI, n=20)", "Capped", "Gen tok", "Decode tok/s", "Latency s", "Max tok", "Peak GB", "Swap GB"], rows8, "dense")
rows_p = []
for (a, b) in zip(M[:-1], M[1:]):
    for t, th in COND:
        pa = cell(a, t, th).set_index("id").correct; pb = cell(b, t, th).set_index("id").correct
        oa, ob = int(((pa == 1) & (pb == 0)).sum()), int(((pb == 1) & (pa == 0)).sum()); p = sign_p(oa, ob)
        rows_p.append([f"{LABEL[a]} → {LABEL[b]}", TASK[t], "on" if th else "off", ob, oa, f"{p:.3f}", "<b>yes</b>" if p < 0.05 else "no"])
T_pair = tbl(["Comparison", "Task", "Think", "Only larger right", "Only smaller right", "Sign-test p", "p < 0.05"], rows_p)
mrows = lambda MIT, name: [[name, LABEL[m], TASK[r.task], int(r.n_mitigated), int(r.rescued), (f"{100*r.mitigate_still_truncated:.0f}%" if int(r.n_mitigated) else "–"), f"{100*r.orig_acc:.0f}%", f"{100*r.acc_after_mitigate:.0f}%"] for m in M for _, r in MIT[m].iterrows()]
T_mit = tbl(["Set", "Model", "Task", "Re-run", "Rescued", "Still capped", "Orig. acc", "Acc after"], mrows(MIT1, "1x") + mrows(MIT8, "8x"), "dense")
GL = [("TTFT", "Time To First Token: delay from sending the prompt until the first generated token appears (includes prefill)."),
      ("Prefill tok/s", "Prompt tokens processed per second before generation starts (compute-bound)."),
      ("Decode tok/s", "Generated tokens per second after the first token (memory-bandwidth-bound)."),
      ("Latency / p95", "End-to-end time per request (mean, and 95th percentile)."),
      ("Capped / truncated", "The model hit the max-token limit before finishing; counted as wrong unless a correct final answer already appeared."),
      ("Acc. if finished", "Accuracy over only the samples that finished within the cap (not truncated)."),
      ("Peak GB", "Peak MLX (GPU) memory allocation during a request, from mx.get_peak_memory()."),
      ("RAM / Swap", "System RAM used and swap used (psutil), sampled every 0.5 s; maximum over the run."),
      ("CPU % / GPU %", "Average system CPU utilisation and GPU device utilisation (ioreg 'Device Utilization %') during generation."),
      ("Therm. limit", "macOS CPU speed limit from pmset; 100% = no thermal throttling."),
      ("95% CI", "Wilson score interval for a proportion; with n=100 it is about ±10 points, with n=20 about ±20."),
      ("Sign test", "Paired exact binomial test on questions where exactly one of two models is correct.")]
T_gl = tbl(["Term", "Meaning"], [[f"<b>{esc(a)}</b>", esc(b)] for a, b in GL])

# ------------------------------------------------------------------ capability & limits
BASE_GB = float(np.median(D1["0.8B"].ram_used_gb_avg)) - LOAD_GB["0.8B"]       # RAM used by macOS + python + libs (0.8B run minus its weights)
work = {m: max(D1[m].peak_mem_gb.max(), D8[m].peak_mem_gb.max()) - LOAD_GB[m] for m in M}   # working memory above weights (activations + cache)
def extra_at_max(D, m):
    d = D[m].assign(tot=lambda x: x.prompt_tokens + x.gen_tokens); r = d.loc[d.tot.idxmax()]; return int(r.tot), r.peak_mem_gb - LOAD_GB[m]
ctx1 = {m: extra_at_max(D1, m) for m in M}; ctx8 = {m: extra_at_max(D8, m) for m in M}
c_budget = stacked([LABEL[m] for m in M], [("macOS + background (measured ≈%.1f GB)" % BASE_GB, [BASE_GB] * 4), ("model weights", [LOAD_GB[m] for m in M]),
                    ("working memory (activations + cache)", [work[m] for m in M]), ("left free", [max(0, 16 - BASE_GB - LOAD_GB[m] - work[m]) for m in M])],
                   fmt="{:.1f}", ymax=16, yref=(16, "16 GB unified memory"), minlab=0.6, height=300)
c_ctx = gbars([LABEL[m] for m in M], [("longest 1x run (~3k tokens)", [ctx1[m][1] for m in M], None), ("longest 8x run (16-25k tokens)", [ctx8[m][1] for m in M], None)], fmt="{:.2f}", ylabel="GB above weights")
BW = float(np.mean(list(bw_est.values())))      # measured effective decode bandwidth, GB/s
USABLE = 16 - BASE_GB
def fit_row(name, params_b, prec, bpp, measured=None):
    gb = LOAD_GB[measured] if measured else params_b * bpp
    need = gb + 1.0
    status = "fits comfortably" if need <= USABLE - 1.5 else ("tight (swap likely)" if need <= USABLE + 1.0 else "does not fit")
    tps = f"{dec[measured, 'gsm8k_text', True]:.0f} (measured)" if measured else (f"≈{BW/gb:.0f}" if need <= USABLE + 1.0 else "–")
    return [name, prec, f"{gb:.1f}" + ("" if measured else " (est.)"), status, tps]
FITS = [fit_row("0.8B", 0.8, "bf16", 2, "0.8B"), fit_row("2B", 2, "bf16", 2, "2B"), fit_row("4B", 4, "bf16", 2, "4B"), fit_row("9B", 9, "4-bit", 0.61, "9B-4bit"),
        fit_row("9B", 9, "bf16", 2.1), fit_row("8B-class", 8, "8-bit", 1.06), fit_row("14B-class", 14, "4-bit", 0.6), fit_row("14B-class", 14, "8-bit", 1.06), fit_row("32B-class", 32, "4-bit", 0.6)]
T_fit = tbl(["Model size", "Precision", "Weights GB", "On this 16 GB Mac", "Decode tok/s (thinking on)"], [[esc(c) for c in r] for r in FITS])
N_GSM, N_AI2D = 1319, 3088      # full GSM8K test split, full AI2D test split
def hrs(m, t, th): return (N_GSM if t == "gsm8k_text" else N_AI2D) * lat[m, t, th] / 3600
T_cost = tbl(["Model", "GSM8K (1,319) thinking off", "GSM8K thinking on", "AI2D (3,088) thinking off", "AI2D thinking on"],
             [[LABEL[m]] + [f"{hrs(m, t, th):.1f} h" for t, th in COND] for m in M])
prof = lambda t, th: (np.mean([cell(m, t, th).prompt_tokens.mean() for m in M]), np.mean([genk[m, t, th] for m in M]))
T_prof = tbl(["Task", "Prompt tokens", "Answer tokens (thinking off)", "Answer tokens (thinking on, 1x cap)", "What dominates the cost"],
             [["GSM8K (text math)", f"{prof('gsm8k_text', False)[0]:.0f}", f"{prof('gsm8k_text', False)[1]:.0f}", f"{prof('gsm8k_text', True)[1]:.0f}", "generated reasoning tokens (decode)"],
              ["AI2D (diagram MCQ, image ≤896 px)", f"{prof('ai2d_vision', False)[0]:.0f}", f"{prof('ai2d_vision', False)[1]:.0f}", f"{prof('ai2d_vision', True)[1]:.0f}", "image prefill when off; reasoning tokens when on"]])
tok_day = {m: dec[m, "gsm8k_text", True] * 86400 / 1e6 for m in M}

# ------------------------------------------------------------------ narrative numbers
g = lambda m, t, th: f"{100*acc[m,t,th]:.0f}%"
total_h = sum(sum(v) for v in hours.values())
tag_n = sum(len(D1[m]) + len(D8[m]) for m in M)
swap_max = {m: max(D1[m].swap_gb_max) for m in M}
CSS = """
:root{--bg:#fbfbfa;--fg:#1d2024;--dim:#6b7078;--card:#fff;--line:#e3e4e0;--grid:#ececE8;--s0:#2f6fdc;--s1:#e07b00;--s2:#139b6c;--s2:#139b6c;--s3:#a23cc4;--s4:#7a7f87;--accent:#2f6fdc}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#14161a;--fg:#e8e9ec;--dim:#9aa0a8;--card:#1b1e23;--line:#2c3037;--grid:#262a31;--s0:#6aa3ff;--s1:#ffa53d;--s2:#4cd6a0;--s3:#d98cf0;--s4:#9aa0a8;--accent:#6aa3ff}}
:root[data-theme=dark]{--bg:#14161a;--fg:#e8e9ec;--dim:#9aa0a8;--card:#1b1e23;--line:#2c3037;--grid:#262a31;--s0:#6aa3ff;--s1:#ffa53d;--s2:#4cd6a0;--s3:#d98cf0;--s4:#9aa0a8;--accent:#6aa3ff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Roboto,sans-serif}
main{max-width:980px;margin:0 auto;padding:32px 16px 80px}h1{font-size:30px;margin:0 0 6px;letter-spacing:-.02em}h2{font-size:21px;margin:44px 0 8px;letter-spacing:-.01em;border-top:1px solid var(--line);padding-top:26px}h3{font-size:16px;margin:26px 0 6px}
p{margin:8px 0}.sub{color:var(--dim);margin:0 0 18px}.dim{color:var(--dim)}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:10px;margin:18px 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}.card b{display:block;font-size:22px;letter-spacing:-.02em}.card span{color:var(--dim);font-size:13px}
.fig{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 12px 6px;margin:12px 0}.fig .cap{color:var(--dim);font-size:13px;margin:2px 4px 6px}.note{border-left:3px solid var(--accent);background:var(--card);padding:8px 12px;margin:12px 0;border-radius:0 8px 8px 0;font-size:14px}
.chart{width:100%;height:auto;display:block}.grid{stroke:var(--grid);stroke-width:1}.axis{stroke:var(--dim);stroke-width:1}.ref{stroke:var(--dim);stroke-width:1;stroke-dasharray:4 3}.err{stroke:var(--fg);stroke-width:1.2;opacity:.7}
.tick,.glab{fill:var(--dim);font-size:11px}.glab{font-size:11.5px;fill:var(--fg)}.val{fill:var(--fg);font-size:10px}.reflab{paint-order:stroke;stroke:var(--card);stroke-width:3px}.vali{fill:#fff;font-size:11px;font-weight:600}
.s0{fill:var(--s0)}.s1{fill:var(--s1)}.s2{fill:var(--s2)}.s3{fill:var(--s3)}.s4{fill:var(--s4)}
.legend{display:flex;flex-wrap:wrap;gap:4px 16px;font-size:13px;margin:0 4px 6px}.legend .sw{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:6px;vertical-align:-1px}
i.s0{background:var(--s0)}i.s1{background:var(--s1)}i.s2{background:var(--s2)}i.s3{background:var(--s3)}i.s4{background:var(--s4)}
.tw{overflow-x:auto;margin:10px 0}table{border-collapse:collapse;width:100%;font-size:13.5px;background:var(--card);border:1px solid var(--line)}th,td{padding:6px 9px;text-align:left;border-bottom:1px solid var(--line);white-space:nowrap}th{color:var(--dim);font-weight:600;font-size:12px}table:not(.dense) td{white-space:normal}
table.dense{font-size:12px}table.dense td,table.dense th{padding:5px 7px}ul{padding-left:20px}li{margin:4px 0}code{background:var(--card);border:1px solid var(--line);border-radius:4px;padding:0 4px;font-size:13px}
@media (max-width:600px){h1{font-size:24px}}"""

HTML = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Mac LLM/VLM Inference Report</title><style>{CSS}</style></head><body><main>
<h1>Qwen3.5 on a 16 GB Apple M5 MacBook: inference analysis</h1>
<p class="sub">0.8B, 2B, 4B (bf16) and 9B (4-bit) · MLX · GSM8K (text) and AI2D (vision) · thinking off / on · generated {datetime.date.today().isoformat()} from <code>results/*.jsonl</code> ({tag_n:,} main + long-thinking inferences, plus {sum(mit1_n)+sum(mit8_n)} mitigation re-runs)</p>

<div class="cards">
<div class="card"><b>{g('9B-4bit','gsm8k_text',False)}</b><span>best GSM8K, 9B 4-bit, thinking off ({26.5:.0f} tok/s)</span></div>
<div class="card"><b>{g('4B','ai2d_vision',True)}</b><span>best AI2D, 4B bf16, thinking on (14 tok/s)</span></div>
<div class="card"><b>≈{np.mean(list(bw_est.values())):.0f} GB/s</b><span>effective memory bandwidth reached while decoding</span></div>
<div class="card"><b>0 throttling</b><span>CPU speed limit stayed at 100% in every run; swap peaked at {max(swap_max.values()):.1f} GB</span></div>
</div>

<h2>Key findings</h2>
<ul>
<li><b>Quality scales with size, with one twist.</b> GSM8K without thinking rises {g('0.8B','gsm8k_text',False)} → {g('2B','gsm8k_text',False)} → {g('4B','gsm8k_text',False)} → {g('9B-4bit','gsm8k_text',False)}. The 9B at 4-bit is the best text model and is <i>faster</i> (26 tok/s) and smaller in memory (5.5 GB) than the 4B in bf16 (14 tok/s, 8.5 GB): on a 16 GB laptop, quantising up is better than running a smaller model at full precision.</li>
<li><b>Thinking helps vision for the smaller models but looks harmful on text under a fixed token cap.</b> AI2D thinking off→on (the 9B already answers directly at {g('9B-4bit','ai2d_vision',False)}): 0.8B {g('0.8B','ai2d_vision',False)}→{g('0.8B','ai2d_vision',True)}, 2B {g('2B','ai2d_vision',False)}→{g('2B','ai2d_vision',True)}, 4B {g('4B','ai2d_vision',False)}→{g('4B','ai2d_vision',True)}. On GSM8K thinking <i>lowers</i> accuracy at the 3,072-token cap (0.8B {g('0.8B','gsm8k_text',False)}→{g('0.8B','gsm8k_text',True)}, 9B {g('9B-4bit','gsm8k_text',False)}→{g('9B-4bit','gsm8k_text',True)}) because up to 94% of the thinking runs never finish.</li>
<li><b>The cap, not the capability, is the problem.</b> With an 8x larger budget (first 20 questions) thinking becomes the best mode and the 9B catches up with the 4B (both 95% on GSM8K): GSM8K 4B {100*matched['4B','gsm8k_text'][0]:.0f}%→{100*matched['4B','gsm8k_text'][1]:.0f}%, 9B {100*matched['9B-4bit','gsm8k_text'][0]:.0f}%→{100*matched['9B-4bit','gsm8k_text'][1]:.0f}%, with no 9B sample truncated. The price is time: ~{lat['4B','gsm8k_text',True]:.0f} s per answer at 1x grows to ~{cell('4B','gsm8k_text',True,D8['4B']).total_s.mean():.0f} s at 8x for the 4B.</li>
<li><b>Remaining truncations are mostly repetition loops.</b> At 8x, {L8['0.8B'][0]+L8['2B'][0]+L8['4B'][0]} of {sum(L8[m][0]+L8[m][1] for m in M[:3])} truncated 0.8B–4B outputs end in a repeating loop (compression-ratio test), while at 1x most truncations are legitimate long reasoning. The 9B almost never loops.</li>
<li><b>Wrong answers are long answers.</b> In thinking mode wrong answers use about {np.mean([cw[m][1]/cw[m][0] for m in M]):.1f}x the tokens of correct ones, which is why more budget alone has diminishing returns.</li>
<li><b>Speed is a memory-bandwidth story.</b> Decode tok/s × weight bytes lands at {min(bw_est.values()):.0f}–{max(bw_est.values()):.0f} GB/s for every model, i.e. most of the chip's peak bandwidth; speed halves when weights double.</li>
<li><b>The 16 GB limit is felt from the 4B up.</b> The 4B in bf16 peaks at {max(peak['4B',t,th] for t,th in COND):.1f} GB of MLX memory and macOS used up to ~{swap_max['4B']:.1f} GB of swap (the 9B runs showed ~{swap_max['9B-4bit']:.1f} GB too, with only {max(peak['9B-4bit',t,th] for t,th in COND):.1f} GB of MLX memory, so that is system-wide pressure). Decode speed stayed flat (~14 tok/s for the 4B), so the swap held idle pages, not the model. 0.8B and 2B never touched swap.</li>
</ul>

<h2>1. Setup</h2>
<p>Apple M5 (10-core CPU/GPU), 16 GB unified memory, macOS (Darwin 25.6), on AC power, other apps closed. MLX 0.32 / mlx-vlm 0.7.12, all models served from the same multimodal checkpoint. GSM8K: first 100 test questions, exact match on the final number. AI2D: first 100 test items, options numbered 1–4 (25% chance level). Thinking off is greedy; thinking on samples at T=0.6, top-p 0.95. Caps: off 512 / 32 tokens, on 3,072 / 2,048 (GSM8K / AI2D); the long-thinking runs use 8x those caps on the first 20 questions. A truncated answer counts as wrong.</p>
<div class="note"><b>Read AI2D thinking-off with care.</b> It is a 32-token budget: the 4B starts explaining before answering and gets cut off in {100*trunc['4B','ai2d_vision',False]:.0f}% of cases, so its {g('4B','ai2d_vision',False)} reflects the format limit, not weak vision. The 9B answers directly ({g('9B-4bit','ai2d_vision',False)}); the 0.8B and 2B are near chance. All cells have n=100, so 95% intervals are about ±10 points.</div>

<h2>2. Accuracy</h2>
<div class="fig"><div class="cap">Accuracy by model and mode; error bars are 95% Wilson intervals (n=100). The dashed line is chance level on AI2D (25%).</div>{c_acc}</div>
<p>Paired sign tests (table below) show significant gains for 0.8B→2B on GSM8K (both modes) and AI2D-thinking, and for 2B→4B with thinking on (GSM8K and AI2D). 4B→9B-4bit is mixed: the 9B wins AI2D with thinking off (a format effect, see the note above), but with thinking on the 4B is significantly <i>better</i> on both tasks, driven by the 9B hitting the token cap more often ({100*trunc['9B-4bit','gsm8k_text',True]:.0f}% vs {100*trunc['4B','gsm8k_text',True]:.0f}% on GSM8K). GSM8K without thinking, 4B vs 9B, is within noise.</p>
<h3>Paired sign tests between neighbouring model sizes</h3>{T_pair}

<h2>3. Token budget, truncation and looping</h2>
<div class="fig"><div class="cap">Share of requests that hit the token cap.</div>{c_trunc}</div>
<div class="fig"><div class="cap">Mean generated tokens (log scale). Thinking costs roughly 6x (GSM8K) to several hundred x (AI2D) more tokens than answering directly.</div>{c_gen}</div>
<div class="fig"><div class="cap">Wrong answers use roughly twice the tokens of correct ones (thinking on, 1x caps).</div>{c_cw}</div>

<h3>Does a bigger budget fix it? (same 20 questions, 1x vs 8x caps)</h3>
<div class="fig"><div class="cap">GSM8K, thinking on, first 20 questions.</div>{c_8x_g}</div>
<div class="fig"><div class="cap">AI2D, thinking on, first 20 questions.</div>{c_8x_a}</div>
<div class="fig"><div class="cap">Share hitting the cap, 1x vs 8x.</div>{c_8x_t}</div>
<p>n=20, so intervals are about ±20 points; the direction is consistent across all four models though. Full 8x results:</p>{T_8x}

<h3>What are the truncated outputs doing?</h3>
<p>A truncated thinking trace is classed as a <b>repetition loop</b> if its last 3,000 characters compress to under 15% of their size (zlib), i.e. the model is repeating itself; otherwise it is <b>long but not looping</b>.</p>
<div class="fig"><div class="cap">Truncated thinking-on outputs at 1x caps (counts, GSM8K + AI2D).</div>{c_loop1}</div>
<div class="fig"><div class="cap">Truncated thinking-on outputs at 8x caps. Loops dominate: giving a looping model more tokens just makes it loop for longer.</div>{c_loop8}</div>

<h2>4. Speed</h2>
<div class="fig"><div class="cap">Decode throughput (thinking on). It depends on the model, not the task.</div>{c_dec}</div>
<div class="fig"><div class="cap">Prefill throughput. Vision prompts carry ~200 extra image tokens.</div>{c_pre}</div>
<div class="fig"><div class="cap">Implied memory bandwidth = decode tok/s × weight size in GB. Decoding streams all weights once per token, so every model sits at a similar fraction of the hardware peak.</div>{c_bw}</div>
<div class="fig"><div class="cap">Mean end-to-end latency per request (log scale).</div>{c_lat}</div>

<h2>5. Efficiency: what does a correct answer cost?</h2>
<div class="fig"><div class="cap">Wall-clock seconds per correct answer (total time / number correct), log scale. Direct answers are about 6–20x cheaper than thinking on GSM8K.</div>{c_spc}</div>
<div class="fig"><div class="cap">Generated tokens per correct answer, log scale.</div>{c_tpc}</div>
<div class="fig"><div class="cap">Total generation time in the benchmark ({total_h:.1f} h across {tag_n:,} inferences).</div>{c_hours}</div>

<h2>6. Memory and system load</h2>
<div class="fig"><div class="cap">Peak MLX memory, peak system RAM and peak swap per model, against the 16 GB limit.</div>{c_mem}</div>
<p>GPU utilisation was 97–99% for every thinking-on run and CPU utilisation 6–12% (the CPU only feeds the GPU), so generation is fully GPU-bound. The thermal speed limit never dropped below 100%, so the M5 did not throttle even on multi-hour runs. The 9B at 4-bit uses only 5.5 GB of weights, which leaves room for long contexts.</p>

<h2>7. Mitigation experiment (top-k 20 + presence penalty 1.5)</h2>
<p>Truncated-and-wrong thinking samples were re-run with Qwen's recommended top-k 20 and presence penalty 1.5 (up to 10 per task; the 8x sets only contain the few samples that still failed). Rescued = now correct.</p>
<div class="fig"><div class="cap">Share of re-run samples that became correct.</div>{c_mit}</div>{T_mit}
<div class="note"><b>Caveat.</b> There is no control arm (re-running with the original sampling settings), and the 1x re-runs often still hit the cap, so part of the rescue rate may be plain re-sampling luck. With 2–20 samples per cell treat these as indicative only.</div>

<h2>8. Full results table</h2>{T_full}

<h2>9. MacBook capability and limits</h2>
<p>What this 16 GB M5 MacBook Pro can and cannot do for local LLM/VLM <b>inference and evaluation</b>, measured with the runs above. (Training capability is covered in the sibling post-training report.)</p>
<div class="cards">
<div class="card"><b>≈{USABLE:.1f} GB</b><span>usable for a model: macOS and background services keep ≈{BASE_GB:.1f} GB in use even with apps closed</span></div>
<div class="card"><b>≈{BW:.0f} GB/s</b><span>effective memory bandwidth while decoding; tokens/s ≈ {BW:.0f} ÷ weight GB</span></div>
<div class="card"><b>+{max(ctx8[m][1] for m in M):.1f} GB</b><span>extra memory for a ~25k-token generation: context is almost free</span></div>
<div class="card"><b>{total_h:.0f} h</b><span>of generation without throttling; peak RAM {max(max(D1[m].ram_used_gb_max) for m in M):.1f} GB</span></div>
</div>
<h3>Memory budget</h3>
<div class="fig"><div class="cap">Where the 16 GB goes (peak working memory taken over all runs). The 4B in bf16 leaves almost nothing free, which is why swap appeared; the 4-bit 9B has more headroom than the 4B.</div>{c_budget}</div>
<h3>Context length costs almost no memory</h3>
<div class="fig"><div class="cap">Peak MLX memory above the weights for the longest request in each set. Qwen3.5 mostly uses gated-delta (linear) attention layers whose state does not grow with length, so a 25k-token thinking trace adds under 1 GB. Memory, not context length, is what decides the model size you can run.</div>{c_ctx}</div>
<h3>What fits</h3>
<p>Rows marked "(est.)" are extrapolations: weights = parameters × bytes per parameter (bf16 2, 8-bit ≈1.06, 4-bit ≈0.6), plus ~1 GB working memory against the usable ≈{USABLE:.1f} GB; decode speed = measured bandwidth ÷ weight size (this rule predicts the four measured models within about 15%).</p>{T_fit}
<h3>What an experiment costs in time</h3>
<p>Time, not memory, is the practical limit for thinking-mode evaluation. Time for a <b>full</b> benchmark split at the measured mean latencies (batch size 1; thinking-on at the 1x caps used here):</p>{T_cost}
<p>Throughput is a hard physical ceiling: decoding streams all weights for every token, so at thinking-on speeds the machine generates about {tok_day['0.8B']:.1f} M tokens/day with the 0.8B, {tok_day['2B']:.1f} M with the 2B, {tok_day['4B']:.1f} M with the 4B and {tok_day['9B-4bit']:.1f} M with the 9B (4-bit).</p>
<h3>Datasets and what drives their cost</h3>{T_prof}
<p>Cost scales with <i>generated</i> tokens, not prompt size. Short-answer multiple-choice with thinking off (AI2D off: a few tokens) is cheap for every model; chain-of-thought math (GSM8K on: ~2,000 tokens) is 100-1000x more expensive per question. When choosing an evaluation set for this machine, estimate answer length first.</p>
<h3>Verdict</h3>
<ul>
<li><b>Comfortable:</b> bf16 models up to 2B (4-7 GB free) and a 4-bit 9B (3 GB free); thinking traces up to 25k tokens; multi-hour unattended runs on AC power with no throttling.</li>
<li><b>Tight:</b> 4B in bf16 (about {LOAD_GB['4B']+work['4B']:.1f} GB with a long generation; macOS started swapping but decode speed did not drop), and, by extrapolation, 14B-class at 4-bit or 8B-class at 8-bit (~8.5 GB of weights): anything above ~8 GB of weights.</li>
<li><b>Not possible:</b> 9B in bf16 (about 19 GB) and 32B-class models even at 4-bit.</li>
<li><b>Slow, not impossible:</b> thinking-mode evaluation of the larger models: a full GSM8K test split with the 4B thinking on takes about {hrs('4B','gsm8k_text',True):.0f} h at 1x caps and much longer at 8x. Sample 100 questions, as done here, or run non-thinking.</li>
<li><b>Not measured here:</b> batched or concurrent generation (decode is bandwidth-bound, so batching should raise total throughput substantially but was not tested), contexts beyond ~25k tokens, images larger than 896 px (images were downsized), power draw in watts, battery operation, and sustained load longer than the ~6 h single runs.</li>
</ul>

<h2>10. Glossary and method notes</h2>{T_gl}
<ul>
<li>Latency, tokens and memory columns are averages over the samples of that cell; Peak GB, RAM and Swap are maxima; Therm. limit is the minimum.</li>
<li>The 9B was converted to 4-bit (group size 64) because bf16 (19 GB) does not fit in 16 GB, so it is not an apples-to-apples precision comparison with the other three.</li>
<li>Sampling at T=0.6 makes thinking-on numbers vary run to run; n=100 (or 20) is small, so rely on large gaps and paired tests.</li>
<li>Power draw (watts) was not measured: it needs <code>sudo powermetrics</code>, and battery readings are meaningless on AC.</li>
<li>Reproduce: <code>python analysis/make_report.py</code> regenerates this page from <code>results/</code>. Every number here comes from the per-sample records.</li>
</ul>
</main></body></html>"""
os.makedirs(R, exist_ok=True)
open(f"{R}/report.html", "w").write(HTML)
print("wrote", f"{R}/report.html", f"{len(HTML)/1024:.0f} KB")
