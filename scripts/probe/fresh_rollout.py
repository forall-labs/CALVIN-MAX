"""One rollout in a fresh process; writes per-step full state to .npy. Compare several of these.

    python fresh_rollout.py --mode render|physics|mixed --out x.npy [--nowarm]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from diag_determinism import full_state  # noqa: E402
from phase0_probe import make_env, physics_step, scripted_actions  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["render", "physics", "mixed"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--egl", action="store_true")
    ap.add_argument("--nowarm", action="store_true")
    args = ap.parse_args()
    env = make_env(args.egl, "static_and_gripper")
    if args.nowarm:
        env.p.setPhysicsEngineParameter(warmStartingFactor=0, articulatedWarmStartingFactor=0, physicsClientId=env.cid)
    env.reset()
    s0 = env.get_state_obs()
    env.reset(robot_obs=s0["robot_obs"], scene_obs=s0["scene_obs"])  # evaluator-style
    acts = scripted_actions(args.steps, seed=3)
    out = []
    for t, a in enumerate(acts):
        render = args.mode == "render" or (args.mode == "mixed" and t >= args.steps // 2)
        env.step(a) if render else physics_step(env, a)
        out.append(full_state(env))
    np.save(args.out, np.array(out))


if __name__ == "__main__":
    main()
