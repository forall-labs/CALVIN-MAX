"""Run Base rollouts for a set of chains against a policy server, one fresh process per chain (D-018).

    python scripts/run_base.py --chains benchmark/chains/eval_1000.json --n 100 --port 6100 \
        --H 10 --Q 10 --out results/base/flower --workers 10

Writes traces/<chain_id>.pkl (used later for branching) and summary.json with the standard CALVIN
numbers: success rate at each chain position and average length.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import pickle
import time
import traceback
from pathlib import Path

import numpy as np


def run_chain(job):
    chain, out, port, H, Q, seed = job[:6]
    kind = job[6] if len(job) > 6 else "chunked"
    gate = job[7] if len(job) > 7 else None  # (every, tau) or None
    ep_len = job[8] if len(job) > 8 else None  # subtask step limit; None = EP_LEN (360, D-038)
    path = out / "traces" / f"{chain['chain_id']}.pkl"
    if path.exists():  # resume
        tr = pickle.loads(path.read_bytes())
        return {"chain_id": chain["chain_id"], "success": tr.subtask_success, "cached": True}
    try:
        from calvin_max.adapter import make_adapter
        from calvin_max.envs import tracked_entities, load_annotations, make_env, make_task_oracle
        from calvin_max.rollout import ChainRunner

        t0 = time.time()
        env = make_env(egl=True)
        runner = ChainRunner(env, make_task_oracle(), load_annotations(), tracked_entities(env),
                             **({"ep_len": ep_len} if ep_len else {}))
        adapter = make_adapter(kind, port, H, Q, seed, chain["chain_id"], chain["chain_id"])
        if gate:
            adapter.gate_every, adapter.gate_tau = gate
        init = (np.array(chain["robot_obs"]), np.array(chain["scene_obs"]))
        tr = runner.run_base(adapter, init, chain["subtasks"])
        adapter.close()
        path.write_bytes(pickle.dumps(tr))
        return {"chain_id": chain["chain_id"], "success": tr.subtask_success, "steps": len(tr.actions),
                "seconds": time.time() - t0}
    except Exception:
        return {"chain_id": chain["chain_id"], "error": traceback.format_exc()}


def summarize(results):
    ok = [r for r in results if "error" not in r]
    lengths = []
    for r in ok:
        n = 0
        for s in r["success"]:
            if not s:
                break
            n += 1
        lengths.append(n)
    lengths = np.array(lengths)
    return {
        "chains": len(ok),
        "errors": len(results) - len(ok),
        "avg_len": float(lengths.mean()) if len(lengths) else None,
        "sr_at": [float((lengths >= i).mean()) for i in range(1, 6)] if len(lengths) else None,
        "mean_seconds_per_chain": float(np.mean([r["seconds"] for r in ok if "seconds" in r] or [np.nan])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chains", type=Path, required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--adapter", default="chunked", choices=["chunked", "xvla", "history", "history:13", "dda", "dda:live"])
    ap.add_argument("--port", default="6100", help="port or comma-separated replica ports")
    ap.add_argument("--H", type=int, required=True)
    ap.add_argument("--Q", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--gate-every", type=int, default=None, help="probe the policy every N steps (D-044)")
    ap.add_argument("--ep-len", type=int, default=None, help="subtask step limit (default 360; diagnostics only)")
    ap.add_argument("--gate-tau", type=float, default=float("inf"), help="replace the queue if disagreement > tau; inf = log only")
    args = ap.parse_args()
    (args.out / "traces").mkdir(parents=True, exist_ok=True)
    chains = json.loads(args.chains.read_text())[: args.n]
    gate = (args.gate_every, args.gate_tau) if args.gate_every else None
    jobs = [(c, args.out, args.port, args.H, args.Q, args.seed, args.adapter, gate, args.ep_len) for c in chains]
    results = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, maxtasksperchild=1) as pool:
        for i, r in enumerate(pool.imap_unordered(run_chain, jobs), 1):
            results.append(r)
            if "error" in r:
                print("ERROR", r["chain_id"], r["error"].splitlines()[-1], flush=True)
            if i % 10 == 0 or i == len(jobs):
                print(json.dumps({"done": i, **summarize(results)}), flush=True)
    (args.out / "summary.json").write_text(json.dumps({"args": {k: str(v) for k, v in vars(args).items()},
                                                       **summarize(results), "results": results}, indent=1))
    errs = [r for r in results if "error" in r]
    if errs:
        print(errs[0]["error"])


if __name__ == "__main__":
    main()
