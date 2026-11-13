"""Dose-response × timing on the magnitude ladder (D-058).

    python scripts/analyze_magnitude.py --policy X-VLA --native <lite results.jsonl ...> \
        --magnitude results/cases/xvla_magnitude_v1/results.jsonl --policy ... --out results/analysis/magnitude.json

Per policy, event and level: Base/Pre/Mid event-subtask success on the full denominator (aggregate.summarize:
unreached or unrealizable cases count as Base), ΔTotal = Mid − Base, ΔConfig = Pre − Base, ΔTiming = Mid − Pre,
95% chain-cluster bootstrap. Level 3 on A slots is the core variant-A case, taken from the native files.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate import load, parse_id, summarize  # noqa: E402
from max_core.stats import chain_bootstrap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def ladder(native, magnitude):
    scales = {k: v for f in sorted((ROOT / "benchmark/manifests").glob("magnitude_v*.json"))
              for k, v in json.loads(f.read_text())["meta"]["scales"].items()}  # v1 camera/target, v2 noise (D-065)
    groups = defaultdict(list)
    for cid, c in magnitude.items():
        _, ct, lv = cid.split("-")[:3]
        groups[(ct, int(lv[1:]))].append((cid, c))
    events = {ct for ct, _ in groups}
    for cid, c in native.items():  # level 3, A slots, for the events on this ladder
        parts = cid.split("-")
        if parts[0] == "core" and parts[1] in events and parts[2] == "A":
            groups[(parts[1], 3)].append((cid, c))
    out = defaultdict(dict)
    for (ct, lv), g in sorted(groups.items()):
        s = summarize(g)
        out[ct][f"L{lv}"] = {"scale": scales[ct][lv - 1], "n": s["n"], "realized": s["realized"],
                             **{k: s.get(k) for k in ("base_sr", "pre_sr", "mid_sr", "delta_total", "delta_config",
                                                      "delta_timing")}}
        # Realization falls with magnitude for large target moves, which dilutes the full-denominator contrasts
        real = [(parse_id(cid)["chain"], c["pre"]["event_success"], c["mid"]["event_success"])
                for cid, c in g if not c.get("status") and not c.get("infra") and "pre" in c and "mid" in c]
        if real:
            ch, pre, mid = zip(*real)
            out[ct][f"L{lv}"]["delta_timing_realized"] = chain_bootstrap(lambda a, b: b.mean() - a.mean(), ch, pre, mid,
                                                                         n_boot=2000)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", action="append", required=True)
    ap.add_argument("--native", nargs="+", type=Path, action="append", required=True)
    ap.add_argument("--magnitude", nargs="+", type=Path, action="append", required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    report = {pol: ladder(load(args.native[i]), load(args.magnitude[i])) for i, pol in enumerate(args.policy)}
    f = lambda v: "" if v is None else f"{100 * v[0]:+5.1f} [{100 * v[1]:+5.1f},{100 * v[2]:+5.1f}]"  # noqa: E731
    for pol, evs in report.items():
        for ct, rows in evs.items():
            print(f"\n{pol} {ct}")
            print(f"{'level':6s} {'scale':>6s} {'n':>4s} {'real':>4s}  Base/Pre/Mid        {'Pre-Base':>21s}  {'Mid-Pre':>21s}"
                  f"  {'Mid-Pre realized':>21s}")
            for lv, r in rows.items():
                print(f"{lv:6s} {r['scale']:6.2f} {r['n']:4d} {r['realized']:4d}  "
                      f"{100 * r['base_sr']:4.0f}/{100 * (r['pre_sr'] or 0):4.0f}/{100 * r['mid_sr']:4.0f}"
                      f"      {f(r['delta_config']):>21s}  {f(r['delta_timing']):>21s}"
                      f"  {f(r.get('delta_timing_realized')):>21s}")
    if args.out:
        args.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
