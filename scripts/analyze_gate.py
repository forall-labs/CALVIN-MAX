"""Stale-commitment mechanism and disagreement gate (D-044): recovery of Mid variants over native Mid.

    python scripts/analyze_gate.py --native <flower native results.jsonl ...> --variants results/cases/flower_lite_gate/results.jsonl

For each realized case that has the native Mid arm and a variant arm (mid_oracle / mid_gate), computes
recovery = Y_variant − Y_mid_native on the event subtask, per event and per channel, with 95% chain-bootstrap
CIs. It also reports the share of the native loss (Base − Mid) each variant recovers, and how often the gate
probed and replaced the plan.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate import load, parse_id  # noqa: E402
from compare_policies import CHANNEL  # noqa: E402


def chain_boot(rows, n=5000, seed=0):
    """rows: (chain, y0, y_native, y_variant). Returns mean recovery and CI, resampling chains."""
    by = defaultdict(list)
    for r in rows:
        by[r[0]].append(r)
    chains = list(by)
    rng = np.random.default_rng(seed)
    point = float(np.mean([r[3] - r[2] for r in rows]))
    draws = []
    for _ in range(n):
        pick = rng.integers(0, len(chains), len(chains))
        s = [r for j in pick for r in by[chains[j]]]
        draws.append(np.mean([r[3] - r[2] for r in s]))
    return point, [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--native", nargs="+", type=Path, required=True)
    ap.add_argument("--variants", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    native = load(args.native)
    variant_cases = load(args.variants)
    var_rows = [c[arm] for c in variant_cases.values() if not c.get("infra")
                for arm in ("mid_oracle", "mid_gate") if arm in c]
    var = defaultdict(dict)
    gate_stats = defaultdict(lambda: [0, 0, 0])  # probes, replacements, rollouts
    for r in var_rows:
        if r.get("status") == "completed" and r["arm"] in ("mid_oracle", "mid_gate"):
            var[r["case_id"]][r["arm"]] = r
            if r["arm"] == "mid_gate":
                ct = parse_id(r["case_id"])["change_type"]
                gs = gate_stats[ct]
                gs[0] += r.get("gate_probes", 0)
                gs[1] += r.get("gate_replacements", 0)
                gs[2] += 1
    report = {}
    for arm in ("mid_oracle", "mid_gate"):
        groups = defaultdict(list)
        for cid, v in var.items():
            if arm not in v or cid not in native or "mid" not in native[cid] or "y0" not in native[cid]:
                continue
            d = parse_id(cid)
            ct = d["change_type"]
            row = (d["chain"], native[cid]["y0"], native[cid]["mid"]["event_success"], v[arm]["event_success"])
            groups[ct].append(row)
            if ct in CHANNEL:
                groups["channel:" + CHANNEL[ct]].append(row)
        out = {}
        for g, rows in sorted(groups.items()):
            rec, ci = chain_boot(rows)
            y0 = np.mean([r[1] for r in rows])
            ym = np.mean([r[2] for r in rows])
            yv = np.mean([r[3] for r in rows])
            loss = y0 - ym
            out[g] = {"n": len(rows), "base": float(y0), "mid_native": float(ym), "mid_variant": float(yv),
                      "recovery": rec, "ci95": ci, "share_of_loss_recovered": (rec / loss) if loss > 1e-9 else None}
        report[arm] = out
    report["gate_activity"] = {ct: {"probes_per_rollout": p / max(n, 1), "replacements_per_rollout": rpl / max(n, 1)}
                               for ct, (p, rpl, n) in gate_stats.items()}
    pct = lambda x: "" if x is None else f"{100 * x:+.1f}"  # noqa: E731
    for arm in ("mid_oracle", "mid_gate"):
        print(f"\n== {arm}: recovery = {arm} − native Mid (event subtask, %)")
        for g, v in report[arm].items():
            share = "" if v["share_of_loss_recovered"] is None else f" ({100 * v['share_of_loss_recovered']:.0f}% of loss)"
            print(f"  {g:32s} n={v['n']:3d} base {100*v['base']:5.1f} mid {100*v['mid_native']:5.1f} -> {100*v['mid_variant']:5.1f}"
                  f"  rec {pct(v['recovery'])} [{pct(v['ci95'][0])}, {pct(v['ci95'][1])}]{share}")
    print("\n== gate activity (per rollout)")
    for ct, v in sorted(report["gate_activity"].items()):
        print(f"  {ct:28s} probes {v['probes_per_rollout']:.1f}  replacements {v['replacements_per_rollout']:.2f}")
    if args.out:
        args.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
