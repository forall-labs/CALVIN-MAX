"""Which reset makes rollouts history-independent? Also: render cost, CPU count.

A  fresh env object per rollout
B  one env; at every reset, restoreState(snapshot taken right after construction) + Python-side state
C  one env; p.resetSimulation() + env.load() + env.reset() at every reset
For each, rollouts of different lengths are run back to back and compared step-for-step with a
fresh-env env.step reference, so any history leak shows up. Physics-only replay is checked the same way.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from diag_determinism import full_state, python_side_state, set_python_side_state  # noqa: E402
from phase0_probe import make_env, physics_step, scripted_actions  # noqa: E402

N = 120


def roll(env, acts, physics_only):
    out = []
    for a in acts:
        physics_step(env, a) if physics_only else env.step(a)
        out.append(full_state(env))
    return np.array(out)


def maxdiff(a, b):
    return float(np.abs(a - b).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--egl", action="store_true")
    ap.add_argument("--nowarm", action="store_true")
    args = ap.parse_args()
    acts = scripted_actions(N, seed=3)
    noise = scripted_actions(400, seed=7)  # a different history to run in between
    rep = {"cpus": os.cpu_count()}

    # calvin_env supports one env per process (scene objects call pybullet on client 0), so one env,
    # and the reference is the first rollout after construction (fresh processes are reproducible).
    t0 = time.perf_counter()
    e = make_env(args.egl, "static_and_gripper")
    rep["construct_s"] = time.perf_counter() - t0
    if args.nowarm:
        e.p.setPhysicsEngineParameter(warmStartingFactor=0, articulatedWarmStartingFactor=0, physicsClientId=e.cid)
    rep["nowarm"] = args.nowarm
    e.reset()
    s0 = e.get_state_obs()
    init = (s0["robot_obs"].copy(), s0["scene_obs"].copy())

    def reset(env_):  # evaluator-style reset
        env_.reset(robot_obs=init[0], scene_obs=init[1])

    # Canonical reset: snapshot S0 once after construction; every rollout = restore S0 + Python-side
    # state + evaluator-style reset. Reference = first canonical rollout.
    sid = e.p.saveState(physicsClientId=e.cid)
    py0 = python_side_state(e)

    def canon():
        e.p.restoreState(stateId=sid, physicsClientId=e.cid)
        set_python_side_state(e, py0)
        reset(e)

    canon()
    ref = roll(e, acts, physics_only=False)
    res = {}
    canon()
    res["canon_again"] = maxdiff(roll(e, acts, False), ref)
    canon()
    roll(e, noise, False)
    canon()
    res["canon_after_other_history"] = maxdiff(roll(e, acts, False), ref)
    canon()
    res["canon_physics_only"] = maxdiff(roll(e, acts, True), ref)
    canon()  # mixed: physics-only prefix, then rendered steps
    mixed = np.vstack([roll(e, acts[:60], True), roll(e, acts[60:], False)])
    res["canon_physics_prefix_then_render"] = maxdiff(mixed, ref)
    # in-process branch snapshot at step 60 (with Python-side state) vs straight
    canon()
    roll(e, acts[:60], True)
    sid60 = e.p.saveState(physicsClientId=e.cid)
    py60 = python_side_state(e)
    roll(e, noise[:100], True)
    e.p.restoreState(stateId=sid60, physicsClientId=e.cid)
    set_python_side_state(e, py60)
    res["branch_snapshot_at_60"] = maxdiff(roll(e, acts[60:], True), ref[60:])
    rep["diffs"] = res

    # Render cost per camera
    reset(e)
    for cam in e.cameras:
        t0 = time.perf_counter()
        for _ in range(20):
            cam.render()
        rep[f"render_ms_{cam.name}"] = 1000 * (time.perf_counter() - t0) / 20
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
