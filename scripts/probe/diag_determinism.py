"""Find out why render-path and physics-only replay diverge, and why saveState/restoreState does.

Prints per-variant max |diff| of (robot_obs, scene_obs) against the env.step path after N steps,
plus the indices that differ. Run on Colab with the same flags as phase0_probe.py.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from phase0_probe import make_env, physics_step, scripted_actions  # noqa: E402


def full_state(env):
    s = env.get_state_obs()
    return np.concatenate([s["robot_obs"], s["scene_obs"]])


def run(env, actions, mode):
    env.reset()
    out = []
    for a in actions:
        if mode == "env_step":
            env.step(a)
        else:
            physics_step(env, a)
            if mode in ("phys+obs", "phys+obs+info"):
                env.get_obs()
            if mode in ("phys+info", "phys+obs+info"):
                env.get_info()
        out.append(full_state(env))
    return np.array(out)


def python_side_state(env):
    r = env.robot
    return {
        "target_pos": np.copy(r.target_pos),
        "target_orn": np.copy(r.target_orn),
        "gripper_action": copy.deepcopy(r.gripper_action),
        "buttons": [(b.state, b.prev_is_pressed) for b in env.scene.buttons],
        "switches": [(w.state, getattr(w, "prev_is_pressed", None)) for w in env.scene.switches],
        "lights": [l.get_state() for l in env.scene.lights],  # Python-side, plus visual colour
    }


def set_python_side_state(env, s):
    r = env.robot
    r.target_pos, r.target_orn, r.gripper_action = np.copy(s["target_pos"]), np.copy(s["target_orn"]), s["gripper_action"]
    for b, (st, prev) in zip(env.scene.buttons, s["buttons"]):
        b.state, b.prev_is_pressed = st, prev
    for light, v in zip(env.scene.lights, s.get("lights", [])):
        light.reset(v)  # sets state and colour
    for w, (st, prev) in zip(env.scene.switches, s["switches"]):
        w.state = st
        if prev is not None:
            w.prev_is_pressed = prev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--egl", action="store_true")
    ap.add_argument("--steps", type=int, default=50)
    args = ap.parse_args()
    env = make_env(args.egl, "static_and_gripper")
    acts = scripted_actions(args.steps)
    report = {}

    ref = run(env, acts, "env_step")
    for mode in ("phys", "phys+obs", "phys+info", "phys+obs+info", "env_step"):
        x = run(env, acts, mode)
        d = np.abs(x - ref)
        first = next((i for i in range(len(d)) if d[i].max() > 0), None)
        report[mode] = {
            "max_abs_diff": float(d.max()),
            "first_step": first,
            "indices_at_first": np.flatnonzero(d[first] > 0).tolist() if first is not None else [],
        }

    # save/restore, with and without the Python-side state
    for with_py in (False, True):
        env.reset()
        for a in acts[:25]:
            physics_step(env, a)
        sid = env.p.saveState(physicsClientId=env.cid)
        py = python_side_state(env)
        straight = []
        for a in acts[25:]:
            physics_step(env, a)
            straight.append(full_state(env))
        env.p.restoreState(stateId=sid, physicsClientId=env.cid)
        if with_py:
            set_python_side_state(env, py)
        again = []
        for a in acts[25:]:
            physics_step(env, a)
            again.append(full_state(env))
        d = np.abs(np.array(straight) - np.array(again))
        report[f"save_restore_python_state={with_py}"] = {"max_abs_diff": float(d.max())}

    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
