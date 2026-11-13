"""Calibrate the pre_grasp_proximity radius on dev-chain Base traces (D-011: geometry only, no outcomes).

For every attempted subtask, onset(r) is the first step t >= MIN_STEPS where the TCP's distance to the
designated entity's AABB drops below r, provided it started outside r. The relative time is t divided
by the subtask's Base duration (360 if it timed out). Success and failure are pooled; outcomes are never used.

Picks the r whose median relative onset is closest to 0.45, within the 0.30-0.60 band, and reports
coverage (the share of subtasks where the trigger fires) by task kind.

    python scripts/calibrate_onsets.py --traces results/base/flower_dev/traces
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from calvin_max.tasks import aabb_distance, designated_entity, task_kind

RADII = np.round(np.arange(0.01, 0.16, 0.01), 3)
MIN_STEPS = 10


def subtask_windows(tr):
    starts = list(tr.subtask_starts) + [len(tr.actions)]
    for k in range(len(tr.subtask_success)):
        yield k, starts[k], starts[k + 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    rows = []
    for p in sorted(args.traces.glob("*.pkl")):
        tr = pickle.loads(p.read_bytes())
        tcp = np.asarray(tr.tcp)
        for k, s, e in subtask_windows(tr):
            task = tr.subtasks[k]
            positions = {b: tr.entity_pos[b][s] for b in ("block_red", "block_blue", "block_pink")}
            ent = designated_entity(task, positions, tcp[s])["target"]
            if ent is None or ent not in tr.entity_aabb:
                continue
            box = np.asarray(tr.entity_aabb[ent][s:e])
            d = aabb_distance(tcp[s:e], box[:, :3], box[:, 3:])
            rows.append({"task": task, "kind": task_kind(task), "dist": d, "len": e - s})
    report = {"subtasks": len(rows), "by_radius": {}}
    for r in RADII:
        rel, fired, starts_inside = [], 0, 0
        by_kind = {}
        for row in rows:
            d = row["dist"]
            if d[0] < r:
                starts_inside += 1
                continue
            hit = np.flatnonzero(d[MIN_STEPS:] < r)
            ok = len(hit) > 0
            by_kind.setdefault(row["kind"], [0, 0])
            by_kind[row["kind"]][0] += ok
            by_kind[row["kind"]][1] += 1
            if ok:
                fired += 1
                rel.append((hit[0] + MIN_STEPS) / row["len"])
        report["by_radius"][f"{r:.2f}"] = {
            "median_rel_onset": float(np.median(rel)) if rel else None,
            "coverage": fired / len(rows),
            "starts_inside": starts_inside / len(rows),
            "coverage_by_kind": {k: v[0] / max(v[1], 1) for k, v in by_kind.items()},
        }
    ok = {r: v for r, v in report["by_radius"].items()
          if v["median_rel_onset"] is not None and 0.30 <= v["median_rel_onset"] <= 0.60}
    report["chosen_radius"] = min(ok, key=lambda r: abs(ok[r]["median_rel_onset"] - 0.45)) if ok else None
    txt = json.dumps(report, indent=1)
    print(txt)
    if args.out:
        args.out.write_text(txt)


if __name__ == "__main__":
    main()
