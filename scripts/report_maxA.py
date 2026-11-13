"""Pre-registered tests on the Max-A analysis outputs (PREREG_MAX, D-099; reporting layer, D-116).

    python scripts/report_maxA.py --prefix maxA_interim --h3 flower mode xvla gr1 dda rf [--partial-h3 pi05 oft]

Reads results/analysis/<prefix>_q9_heldout.json, _q9_heldout_no3dda.json and _q9_all.json (compare_policies.py),
and recomputes from results/cases/<p>_maxA[_h3] what the frozen scripts print only as 95% intervals:
  H1  per-policy interaction p (one-sided, bootstrap), Holm over the policies present; pooled from the q9 file
  H2  pooled physical Mid − Pre, two-sided (sign reported, ±1 point = "no detectable effect")
  H3  per policy, perceptual core: Mid − mid_rewind > 0 (one-sided p, Holm) and the residual mid_rewind − Pre
      equivalent to 0 within ±3 points (TOST at α = 0.05, i.e. 90% interval inside [−3, +3]; Holm on the TOST p)
  H6  per policy, aftershock Mid < Base (one-sided p, Holm), the analyze_mechanism.py rows
Chain-cluster bootstrap, 5,000 draws, seed 0, as in the frozen scripts. Writes results/analysis/<prefix>_report.json.
--partial-h3 policies are reported with their current rows and kept out of the H3 verdicts and Holm family.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aggregate import load, parse_id  # noqa: E402
from compare_policies import CHANNEL  # noqa: E402
from max_core.stats import holm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NAME = {"flower": "FLOWER", "mode": "MoDE", "xvla": "X-VLA", "pi05": "pi0.5", "gr1": "GR-1", "seer": "Seer",
        "univla": "UniVLA", "dda": "3DDA", "rf": "RoboFlamingo", "oft": "OpenVLA-OFT"}
DIR = {v: k for k, v in NAME.items()}
MARGIN = 3.0


def draws(rows, fn, n=5000, seed=0):
    """Point and bootstrap draws of fn(sample) in points, resampling whole chains."""
    by = defaultdict(list)
    for r in rows:
        by[r["chain"]].append(r)
    chains = list(by)
    rng = np.random.default_rng(seed)
    out = np.empty(n)
    for b in range(n):
        out[b] = fn([r for j in rng.integers(0, len(chains), len(chains)) for r in by[chains[j]]])
    return fn(rows), out


def mean_diff(a, b):
    return lambda s: 100 * float(np.mean([r[a] - r[b] for r in s]))


def h3_rows(p):
    native = load([ROOT / f"results/cases/{p}_maxA/results.jsonl"])
    variants = defaultdict(dict)
    for line in (ROOT / f"results/cases/{p}_maxA_h3/results.jsonl").read_text().splitlines():
        r = json.loads(line) if line.strip() else None
        if r and r["status"] == "completed" and r["arm"] in ("mid_oracle", "mid_rewind"):
            variants[r["case_id"]][r["arm"]] = r
    rows = []
    for cid, c in native.items():  # the filter of analyze_decomposition.decomposition, perceptual core only
        v = variants.get(cid, {})
        d = parse_id(cid)
        if c.get("status") or not all(a in c for a in ("pre", "mid")) or "mid_rewind" not in v:
            continue
        if d["track"] != "core" or CHANNEL.get(d["change_type"]) != "perceptual":
            continue
        rows.append({"chain": d["chain"], "pre": c["pre"]["event_success"], "mid": c["mid"]["event_success"],
                     "mid_rewind": v["mid_rewind"]["event_success"]})
    return rows


def h3(p):
    rows = h3_rows(p)
    gain, gd = draws(rows, mean_diff("mid", "mid_rewind"))
    res, rd = draws(rows, mean_diff("mid_rewind", "pre"))
    tim, td = draws(rows, mean_diff("mid", "pre"))
    p_lo, p_hi = float(np.mean(rd <= -MARGIN)), float(np.mean(rd >= MARGIN))
    return {"n": len(rows), "chains": len({r["chain"] for r in rows}),
            "timing": {"mean": tim, "ci95": list(np.quantile(td, [0.025, 0.975]))},
            "mid_minus_rewind": {"mean": gain, "ci95": list(np.quantile(gd, [0.025, 0.975])),
                                 "p_one_sided_gt0": float(np.mean(gd <= 0))},
            "residual": {"mean": res, "ci90": list(np.quantile(rd, [0.05, 0.95])),
                         "ci95": list(np.quantile(rd, [0.025, 0.975])), "p_tost": max(p_lo, p_hi)}}


def h6(p):
    cases = load([ROOT / f"results/cases/{p}_maxA/results.jsonl"])
    rows = []
    for cid, c in cases.items():  # the aftershock filter of analyze_mechanism.py
        d = parse_id(cid)
        if d["track"] not in ("core", "new") or d["change_type"] not in CHANNEL or "mid" not in c or c.get("infra"):
            continue
        m, k, base_len = c["mid"], d["k"] - 1, c.get("len0")
        if base_len is None or k + 1 >= 5:
            continue
        if len(m["subtask_success"]) > k and m["subtask_success"][k] and base_len >= k + 1:
            rows.append({"chain": d["chain"], "base": int(base_len >= k + 2),
                         "mid": int(len(m["subtask_success"]) > k + 1 and m["subtask_success"][k + 1])})
    est, dd = draws(rows, mean_diff("mid", "base"))
    return {"n": len(rows), "mid": 100 * np.mean([r["mid"] for r in rows]), "base": 100 * np.mean([r["base"] for r in rows]),
            "difference": est, "ci95": list(np.quantile(dd, [0.025, 0.975])), "p_one_sided_lt0": float(np.mean(dd >= 0))}


def pct(e):
    return {"estimate": 100 * e["estimate"], "ci95": [100 * v for v in e["ci95"]],
            **{k: e[k] for k in e if k.startswith("p_")}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="maxA")
    ap.add_argument("--h3", nargs="*", default=list(NAME))
    ap.add_argument("--partial-h3", nargs="*", default=[])
    args = ap.parse_args()
    A = ROOT / "results/analysis"
    q = {k: json.loads((A / f"{args.prefix}_q9_{k}.json").read_text()) for k in ("heldout", "heldout_no3dda", "all")}
    pols = [k for k in q["heldout"] if k != "pooled"]
    rep = {"prefix": args.prefix, "policies": pols, "h3_policies": [NAME[p] for p in args.h3],
           "h3_partial": [NAME[p] for p in args.partial_h3]}

    # H1, H2
    per = {k: {ch: pct(q["heldout"][k]["channel_timing"][ch]) for ch in ("perceptual", "physical", "interaction")}
           for k in pols}
    adj = holm([per[k]["interaction"]["p_one_sided_gt0"] for k in pols])
    for k, a in zip(pols, adj):
        per[k]["interaction"]["p_holm"] = float(a)
    rep["H1"] = {"pooled": {f: pct(q[f]["pooled"]["interaction"]) for f in q}, "per_policy": per,
                 "n_holm_lt_0.05": int(sum(a < 0.05 for a in adj))}
    rep["H2"] = {f: pct(q[f]["pooled"]["physical"]) for f in q}
    rep["perceptual_pooled"] = {f: pct(q[f]["pooled"]["perceptual"]) for f in q}

    # H3
    h = {NAME[p]: h3(p) for p in args.h3 + args.partial_h3}
    full = [NAME[p] for p in args.h3]
    for key, pk in (("mid_minus_rewind", "p_one_sided_gt0"), ("residual", "p_tost")):
        for k, a in zip(full, holm([h[k][key][pk] for k in full])):
            h[k][key]["p_holm"] = float(a)
    for k in full:  # the registered criterion: gain Holm-significant, residual's 90% interval inside ±3 (p_holm on
        g, r = h[k]["mid_minus_rewind"], h[k]["residual"]  # the TOST p is reported as the stricter reading)
        r["equivalent"] = bool(-MARGIN < r["ci90"][0] and r["ci90"][1] < MARGIN)
        h[k]["verdict"] = ("supported" if g["p_holm"] < 0.05 and r["equivalent"] else
                           "gain only, residual not equivalent" if g["p_holm"] < 0.05 else
                           "equivalent residual, no gain" if r["equivalent"] else "not supported")
    rep["H3"] = h

    # H6
    a6 = {NAME[DIR[k]] if k in DIR else k: h6(DIR[k]) for k in pols}
    for k, a in zip(a6, holm([a6[k]["p_one_sided_lt0"] for k in a6])):
        a6[k]["p_holm"] = float(a)
    rep["H6"] = a6

    out = A / f"{args.prefix}_report.json"
    out.write_text(json.dumps(rep, indent=1, default=float) + "\n")

    f = lambda x: f"{x:+.1f}"  # noqa: E731
    c = lambda v: f"[{v[0]:+.1f}, {v[1]:+.1f}]"  # noqa: E731
    print(f"== {args.prefix}: {len(pols)} policies ({', '.join(pols)})")
    for fam, lab in (("heldout", "held-out"), ("heldout_no3dda", "held-out, no 3DDA"), ("all", "Lite + Max-A")):
        i, p2 = rep["H1"]["pooled"][fam], rep["H2"][fam]
        print(f"H1 pooled ({lab}): {f(i['estimate'])} {c(i['ci95'])} p={i['p_one_sided_gt0']:.4f}   "
              f"H2 physical: {f(p2['estimate'])} {c(p2['ci95'])}")
    print("H1 per policy: perc / phys / interaction [CI] p_holm")
    for k in pols:
        e = per[k]
        print(f"  {k:13s} {f(e['perceptual']['estimate'])} / {f(e['physical']['estimate'])} / "
              f"{f(e['interaction']['estimate'])} {c(e['interaction']['ci95'])} p_holm={e['interaction']['p_holm']:.4f}")
    print("H3 perceptual core: n, Mid-Pre, Mid-rewind [95%] p_holm, residual [90%] p_tost_holm, verdict")
    for k, e in h.items():
        g, r = e["mid_minus_rewind"], e["residual"]
        print(f"  {k:13s} n={e['n']:4d} {f(e['timing']['mean'])}  {f(g['mean'])} {c(g['ci95'])} "
              f"p={g.get('p_holm', g['p_one_sided_gt0']):.4f}  {f(r['mean'])} {c(r['ci90'])} "
              f"p={r.get('p_holm', r['p_tost']):.4f}  {e.get('verdict', 'PARTIAL')}")
    print("H6 aftershock: Mid vs Base next-subtask, diff [CI] p_holm")
    for k, e in a6.items():
        print(f"  {k:13s} n={e['n']:4d} {e['mid']:.1f} vs {e['base']:.1f}  {f(e['difference'])} {c(e['ci95'])} "
              f"p_holm={e['p_holm']:.4f}")


if __name__ == "__main__":
    main()
