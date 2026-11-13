"""Privileged-state response metrics for Mid branches of block-relocation events (RESEARCH_PLAN §4).

    python scripts/compute_behaviour.py --cases results/cases/flower_lite_v0 --base results/base/flower_abc

For each completed Mid arm of a block-relocation event (block payloads; --types, default target_relocation) it
computes from the branch trace: retargeting latency, stale commitment toward the old pose, ghost reach, side effects
and the response class. Receptacle relocation moves the stack-base block, whose onset is proximity to the receptacle,
so the same metrics measure retargeting of the placing motion. Writes behaviour.jsonl next to results.jsonl and prints
per-event summaries. Each record carries its event, so one file can hold several events.
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

from calvin_max import behaviour


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=Path, required=True)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--types", default="target_relocation", help="comma-separated block-relocation events")
    args = ap.parse_args()
    types = tuple(args.types.split(","))
    rows = [json.loads(l) for l in (args.cases / "results.jsonl").read_text().splitlines() if l.strip()]
    out, by = [], defaultdict(list)
    for r in rows:
        if r.get("arm") != "mid" or r["status"] != "completed" or "object" not in (r.get("payload") or {}):
            continue
        ct = r["case_id"].split("-")[1]
        if ct not in types:
            continue
        tr = pickle.loads((args.cases / "traces" / f"{r['case_id']}.mid.pkl").read_bytes())
        obj, onset = r["payload"]["object"], r["onset"]
        base = pickle.loads((args.base / "traces" / f"{r['case_id'].split('-')[-1]}.pkl").read_bytes())
        old = np.asarray(base.entity_pos[obj][onset - 1])
        new = np.asarray(r["payload"]["pos"])
        k = int(r["case_id"].split("-")[-2][1:]) - 1
        end = tr.subtask_starts[k + 1] if k + 1 < len(tr.subtask_starts) else len(tr.tcp)
        tcp = np.asarray(tr.tcp[onset:end])
        lat = behaviour.retargeting_latency(tcp, old, new)
        stale = behaviour.stale_commitment(tcp, old, until=lat)
        ghost = behaviour.ghost_reach(tcp, old, new)
        moved = float(np.sum(np.linalg.norm(np.diff(tcp, axis=0), axis=1))) if len(tcp) > 1 else 0.0
        cls = behaviour.classify_response(bool(r["event_success"]), lat, ghost, r.get("side_effects_event_subtask"), moved)
        rec = {"case_id": r["case_id"], "event": ct, "pending_at_onset": r.get("pending_at_onset"), "latency": lat,
               "stale_m": stale, "ghost_reach": ghost, "class": cls, "success": r["event_success"]}
        out.append(rec)
        by[ct].append(rec)
    (args.cases / "behaviour.jsonl").write_text("".join(json.dumps(x) + "\n" for x in out))
    for ct, recs in by.items():
        lats = [x["latency"] for x in recs if x["latency"] is not None]
        print(json.dumps({"event": ct, "n": len(recs),
                          "latency_median_steps": float(np.median(lats)) if lats else None,
                          "never_retargeted": sum(x["latency"] is None for x in recs),
                          "ghost_reach_rate": float(np.mean([bool(x["ghost_reach"]) for x in recs])),
                          "stale_m_median": float(np.median([x["stale_m"] for x in recs])),
                          "classes": {c: sum(x["class"] == c for x in recs) for c in behaviour.RESPONSE_CLASSES}}))


if __name__ == "__main__":
    main()
