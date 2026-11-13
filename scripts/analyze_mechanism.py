"""Free analyses on existing native Lite results (RESEARCH_PLAN Q3 aftershock, Q4 mechanism).

    python scripts/analyze_mechanism.py --policy FLOWER <results.jsonl ...> --policy MoDE ... --policy X-VLA ...

Q3 aftershock: among realized Mid cases where the event subtask succeeded in BOTH Base and Mid, compare the
success of the next subtask (k+1) in Mid vs Base. An event the policy "handled" may still damage what follows.
Q4 mechanism: within each channel, Mid event-subtask success by the number of stale queued actions at the onset
(0, 1-4, 5+). If physical losses come from executing the stale plan, success should fall as the queue grows.
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
from max_core.stats import chain_bootstrap


def boot_ci(x, y, chains, n=5000, seed=0):
    """Resample whole chains: multiple events on one Base trajectory are not independent."""
    point, lo, hi = chain_bootstrap(lambda a, b: a.mean() - b.mean(), chains, x, y, n_boot=n, seed=seed)
    return point, [lo, hi]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", nargs="+", action="append", required=True)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    report = {}
    for spec in args.policy:
        name, files = spec[0], [Path(f) for f in spec[1:]]
        cases = load(files)
        after_mid, after_base, after_chains = [], [], []
        by_stale = defaultdict(lambda: defaultdict(list))
        for cid, c in cases.items():
            d = parse_id(cid)
            if d["track"] not in ("core", "new") or d["change_type"] not in CHANNEL or "mid" not in c or c.get("infra"):
                continue
            m, k = c["mid"], d["k"] - 1
            ch = CHANNEL[d["change_type"]]
            bucket = "0" if m["pending_at_onset"] == 0 else ("1-4" if m["pending_at_onset"] <= 4 else "5+")
            by_stale[ch][bucket].append(m["event_success"])
            # aftershock: need the Base chain outcomes; reconstruct from base_chain_length
            base_len = c.get("len0")
            if base_len is None or k + 1 >= 5:
                continue
            base_next = int(base_len >= k + 2)
            mid_ok = len(m["subtask_success"]) > k and m["subtask_success"][k]
            base_ok = base_len >= k + 1
            if mid_ok and base_ok:
                mid_next = int(len(m["subtask_success"]) > k + 1 and m["subtask_success"][k + 1])
                after_mid.append(mid_next)
                after_base.append(base_next)
                after_chains.append(d["chain"])
        print(f"\n=== {name}")
        if after_mid:
            est, ci = boot_ci(after_mid, after_base, after_chains)
            report[name] = {"n": len(after_mid), "chains": len(set(after_chains)), "mid": float(np.mean(after_mid)),
                            "base": float(np.mean(after_base)), "difference": est, "ci95": ci}
            print(f"Q3 aftershock: next-subtask success | event subtask ok in both arms: Mid {100*np.mean(after_mid):.1f}% "
                  f"vs Base {100*np.mean(after_base):.1f}%  diff {100*est:+.1f} [{100*ci[0]:+.1f}, {100*ci[1]:+.1f}] (n={len(after_mid)})")
        for ch in ("perceptual", "physical"):
            row = "  ".join(f"{b}: {100*np.mean(v):.1f}% (n={len(v)})" for b, v in sorted(by_stale[ch].items()) if v)
            print(f"Q4 {ch:10s} Mid event success by stale actions at onset -> {row}")
    if args.out:
        args.out.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
