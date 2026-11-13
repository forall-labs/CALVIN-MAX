"""Re-derive per-step entity data for saved Base traces by exact physics-only replay (no policy needed).

Traces recorded before an entity was tracked are filled in from their own actions. Every step's state
hash must match the recorded one, so the added data is exactly what Base saw (D-018).

    python scripts/augment_traces.py --traces results/base/flower_abc/traces --chains benchmark/chains/eval_1000.json
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import pickle
import traceback
from pathlib import Path

import numpy as np


def augment(job):
    path, chain = job
    try:
        from calvin_max.envs import make_env, tracked_entities
        from calvin_max.rollout import physics_step, state_hash

        tr = pickle.loads(Path(path).read_bytes())
        if tr.entity_aabb and "drawer_link" in tr.entity_aabb and getattr(tr, "scene_obs", None):
            return {"path": path, "skipped": True}
        env = make_env(egl=True)
        env.reset(robot_obs=np.array(chain["robot_obs"]), scene_obs=np.array(chain["scene_obs"]))
        ents = tracked_entities(env)
        pos = {e.name: [] for e in ents}
        aabb = {e.name: [] for e in ents}
        contact = {e.name: [] for e in ents}
        scene = []
        for t, a in enumerate(tr.actions):
            physics_step(env, a)
            if state_hash(env) != tr.hashes[t]:
                return {"path": path, "error": f"hash mismatch at step {t}"}
            scene.append(env.scene.get_obs())
            for e in ents:
                pos[e.name].append(e.position(env))
                aabb[e.name].append(e.aabb(env))
                contact[e.name].append(e.touching_robot(env))
        tr.entity_pos, tr.entity_aabb, tr.contact, tr.scene_obs = pos, aabb, contact, scene
        Path(path).write_bytes(pickle.dumps(tr))
        return {"path": path, "steps": len(tr.actions)}
    except Exception:
        return {"path": path, "error": traceback.format_exc()[-500:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", type=Path, required=True)
    ap.add_argument("--chains", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    chains = {c["chain_id"]: c for c in json.loads(args.chains.read_text())}
    jobs = [(str(p), chains[p.stem]) for p in sorted(args.traces.glob("*.pkl"))]
    with mp.get_context("spawn").Pool(args.workers, maxtasksperchild=1) as pool:
        res = list(pool.imap_unordered(augment, jobs))
    errs = [r for r in res if "error" in r]
    print(json.dumps({"traces": len(res), "augmented": sum("steps" in r for r in res),
                      "skipped": sum("skipped" in r for r in res), "errors": len(errs)}))
    for r in errs[:5]:
        print(r)


if __name__ == "__main__":
    main()
