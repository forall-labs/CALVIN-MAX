"""Cross-policy summary of the behaviour metrics written by compute_behaviour.py (contribution #4; D-060).

    python scripts/summarize_behaviour.py --policy FLOWER results/cases/flower_lite_v0 results/cases/flower_lite_v1 \
        --policy MoDE results/cases/mode1_lite_v1 --policy X-VLA results/cases/xvla_lite_v1 --out results/analysis/behaviour.json

One event per run (--event, default target_relocation; block payloads), Mid arm. Later directories override
earlier ones per case. Latency is split by
whether the onset fell inside a queued chunk (pending_at_onset > 0), which is where open-loop commitment shows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from calvin_max.behaviour import RESPONSE_CLASSES


def summary(recs):
    lat = [r["latency"] for r in recs if r["latency"] is not None]
    med = lambda v: float(np.median(v)) if v else None  # noqa: E731
    return {"n": len(recs), "success": float(np.mean([r["success"] for r in recs])),
            "latency_median_steps": med(lat), "never_retargeted": sum(r["latency"] is None for r in recs),
            "ghost_reach_rate": float(np.mean([bool(r["ghost_reach"]) for r in recs])),
            "stale_m_median": med([r["stale_m"] for r in recs]),
            "latency_median_queued": med([r["latency"] for r in recs if r["latency"] is not None and r["pending_at_onset"]]),
            "latency_median_fresh": med([r["latency"] for r in recs if r["latency"] is not None and not r["pending_at_onset"]]),
            "classes": {c: sum(r["class"] == c for r in recs) for c in RESPONSE_CLASSES}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", nargs="+", action="append", required=True, metavar=("NAME", "CASE_DIRS"))
    ap.add_argument("--event", default="target_relocation")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    report = {}
    for spec in args.policy:
        recs = {}
        for d in spec[1:]:
            for line in (Path(d) / "behaviour.jsonl").read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    if r["case_id"].split("-")[1] == args.event:
                        recs[r["case_id"]] = r
        recs = list(recs.values())
        report[spec[0]] = {"all": summary(recs), "success": summary([r for r in recs if r["success"]]),
                           "failure": summary([r for r in recs if not r["success"]])}
    print(json.dumps(report, indent=1))
    if args.out:
        args.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
