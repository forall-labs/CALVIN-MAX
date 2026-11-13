"""Event-position profile along the chain (contribution #3; D-060). CALVIN-only: LIBERO-MAX has single tasks.

    python scripts/analyze_position.py --policy FLOWER <results.jsonl ...> --policy MoDE ... --out results/analysis/position.json

Per policy, channel and event position k (subtask 1-5 of the chain), on core+new cases with the full
denominator (aggregate.summarize): ΔTotal = Mid − Base, ΔConfig = Pre − Base, ΔTiming = Mid − Pre (event-subtask
success, 95% chain-cluster bootstrap) and the chain-length change Mid − Base. Positions differ in task mix and
Base success, so the profile is descriptive.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate import load, parse_id, summarize  # noqa: E402
from compare_policies import CHANNEL  # noqa: E402


def profile(cases):
    groups = defaultdict(list)
    for cid, c in cases.items():
        d = parse_id(cid)
        if d["track"] not in ("core", "new") or d["change_type"] not in CHANNEL:
            continue
        for ch in (CHANNEL[d["change_type"]], "all"):
            groups[(ch, d["k"])].append((cid, c))
    out = {}
    for (ch, k), g in sorted(groups.items()):
        s = summarize(g)
        row = {"n": s["n"], "realized": s["realized"], "base_sr": s["base_sr"], "delta_total": s["delta_total"],
               "delta_config": s.get("delta_config"), "delta_timing": s.get("delta_timing")}
        if "chain_len_base" in s:
            row["chain_len_delta"] = s["chain_len_mid"] - s["chain_len_base"]
        out[f"{ch}/k{k}"] = row
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", nargs="+", action="append", required=True, metavar=("NAME", "RESULTS"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    report = {spec[0]: profile(load([Path(f) for f in spec[1:]])) for spec in args.policy}
    f = lambda v: "" if v is None else f"{100 * v[0]:+5.1f} [{100 * v[1]:+5.1f},{100 * v[2]:+5.1f}]"  # noqa: E731
    for pol, rows in report.items():
        print(f"\n{pol}")
        print(f"{'group':16s} {'n':>4s} {'Base':>5s}  {'Mid-Base':>21s}  {'Pre-Base':>21s}  {'Mid-Pre':>21s}  dLen")
        for g, r in rows.items():
            print(f"{g:16s} {r['n']:4d} {100 * r['base_sr']:5.1f}  {f(r['delta_total']):>21s}  {f(r['delta_config']):>21s}"
                  f"  {f(r['delta_timing']):>21s}  {r.get('chain_len_delta', float('nan')):+.2f}")
    if args.out:
        args.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
