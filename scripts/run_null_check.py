"""Null-event branches against saved Base traces, with a real policy server (Phase 1 gate, D-021).

For each Base trace it runs, each in a fresh process:
- a null Mid branch at a non-query step halfway through subtask 1 (restored queue, mid-chunk);
- a null Pre branch at the start of subtask 2, if Base reached it.

Both must reproduce Base's hashes and subtask outcomes exactly. With a stochastic GPU policy this
also tests seeded noise (D-023) and inference determinism.

    python scripts/run_null_check.py --base results/base/flower_smoke --chains benchmark/chains/eval_1000.json \
        --H 10 --Q 10 --workers 4
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import pickle
import traceback
from pathlib import Path

import numpy as np


def branch(job):
    chain, arm = job[0], job[2]
    try:
        _, trace_path, _, onset, port, H, Q, seed = job[:8]
        kind = job[8] if len(job) > 8 else "chunked"
        from calvin_max.adapter import make_adapter
        from calvin_max.envs import tracked_entities, load_annotations, make_env, make_task_oracle
        from calvin_max.rollout import ChainRunner

        base = pickle.loads(Path(trace_path).read_bytes())
        env = make_env(egl=True)
        runner = ChainRunner(env, make_task_oracle(), load_annotations(), tracked_entities(env))
        adapter = make_adapter(kind, port, H, Q, seed, chain["chain_id"], chain["chain_id"] + arm)
        init = (np.array(chain["robot_obs"]), np.array(chain["scene_obs"]))
        tr = runner.run_branch(adapter, base, init, arm, onset, None)
        adapter.close()
        same = [a == b for a, b in zip(tr.hashes, base.hashes)]
        return {
            "chain_id": chain["chain_id"], "arm": arm, "onset": onset, "pending": tr.pending_at_onset,
            "identical": all(same) and len(tr.hashes) == len(base.hashes),
            "first_divergence": next((i for i, s in enumerate(same) if not s), None),
            "steps": (len(tr.hashes), len(base.hashes)),
            "success": (tr.subtask_success, base.subtask_success),
        }
    except Exception:
        return {"chain_id": chain["chain_id"], "arm": arm, "error": traceback.format_exc()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--chains", type=Path, required=True)
    ap.add_argument("--adapter", default="chunked", choices=["chunked", "xvla", "history", "history:13", "dda", "dda:live"])
    ap.add_argument("--port", default="6100", help="port or comma-separated replica ports")
    ap.add_argument("--H", type=int, required=True)
    ap.add_argument("--Q", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    chains = {c["chain_id"]: c for c in json.loads(args.chains.read_text())}
    jobs = []
    for p in sorted((args.base / "traces").glob("*.pkl")):
        tr = pickle.loads(p.read_bytes())
        c = chains[p.stem]
        end1 = tr.subtask_starts[1] if len(tr.subtask_starts) > 1 else len(tr.actions)
        mid = end1 // 2
        while args.Q > 1 and (mid in tr.snapshots or (mid - 1) in tr.snapshots):  # land mid-chunk, not on a query step
            mid += 1  # (with Q = 1 every step is a query step)
        jobs.append((c, str(p), "mid", mid, args.port, args.H, args.Q, args.seed, args.adapter))
        if len(tr.subtask_starts) > 1:
            jobs.append((c, str(p), "pre", tr.subtask_starts[1], args.port, args.H, args.Q, args.seed, args.adapter))
    with mp.get_context("spawn").Pool(args.workers, maxtasksperchild=1) as pool:
        res = list(pool.imap_unordered(branch, jobs))
    for r in res:
        print(json.dumps(r if "error" not in r else {**r, "error": r["error"].splitlines()[-1]}))
    ok = [r for r in res if r.get("identical")]
    print(json.dumps({"branches": len(res), "identical": len(ok)}))


if __name__ == "__main__":
    main()
