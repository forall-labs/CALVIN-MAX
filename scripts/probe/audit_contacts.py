"""Colab-only diagnostic: are post-event contacts stale after a teleport, and does refreshing them change physics?

PyBullet's getContactPoints reports the contacts of the last collision pass. An event that resets a body pose
and then reads env.get_info() before any step may therefore record contacts from before the event. For each
selected case, this replays Base to the Mid onset, applies the event, and records the moved objects' contacts
as read immediately and after p.performCollisionDetection(). It then replays 20 more Base actions with
and without the refresh and compares the final states, to check that the refresh leaves the dynamics unchanged.
"""
import argparse
import json
import multiprocessing as mp
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/content/CALVIN-MAX")
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]


def audit(job):
    case, chain, refresh = job
    from run_cases import _setup
    from calvin_max.cases import plan_case
    from calvin_max.events import make_event
    from calvin_max.rollout import physics_step
    base = pickle.loads((ROOT / "results/base/flower_abc/traces" / (chain["chain_id"] + ".pkl")).read_bytes())
    plan = plan_case(case, base)
    if plan["status"] != "planned":
        return {"case_id": case["case_id"], "refresh": refresh, "status": plan["status"]}
    env, runner, init = _setup(chain)
    onset = plan["mid_onset"]
    runner.replay_prefix(base, init, onset)
    effect = make_event(case["change_type"], plan["payload"])(env)
    names = list(effect.get("moved_objects", ())) + list(effect.get("shifted_objects", {}))

    def contacts():
        info = env.get_info()["scene_info"]["movable_objects"]
        return {n: sorted({(c[2], c[4]) for c in info[n]["contacts"]}) for n in names}

    stale = contacts()
    if refresh:
        env.p.performCollisionDetection(physicsClientId=env.cid)
    fresh = contacts()
    for action in base.actions[onset:onset + 20]:
        physics_step(env, action)
    obs = env.get_obs()
    return {"case_id": case["case_id"], "task": case["task"], "refresh": refresh, "status": "ok",
            "contacts_read": stale, "contacts_after_refresh": fresh,
            "final_state": np.concatenate([obs["robot_obs"], obs["scene_obs"]]).tolist()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--types", default="grasp_loss,target_relocation,arm_push")
    ap.add_argument("--n", type=int, default=4, help="cases per change_type")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    manifest = json.loads((ROOT / "benchmark/manifests/max_v1.json").read_text())["cases"]
    lite = set(json.loads((ROOT / "benchmark/manifests/lite_v1.json").read_text())["case_ids"])
    cases = []
    for ct in args.types.split(","):
        cases += sorted([c for c in manifest if c["case_id"] in lite and c["change_type"] == ct
                         and c["track"] != "phase"], key=lambda c: c["case_id"])[:args.n]
    chains = {c["chain_id"]: c for c in json.loads((ROOT / "benchmark/chains/eval_1000.json").read_text())}
    jobs = [(c, chains[c["chain_id"]], r) for c in cases for r in (False, True)]
    with mp.get_context("spawn").Pool(4, maxtasksperchild=1) as pool:
        rows = list(pool.imap(audit, jobs))
    report = []
    for a, b in zip(rows[0::2], rows[1::2]):
        if a["status"] != "ok":
            report.append(a)
            continue
        report.append({"case_id": a["case_id"], "task": a["task"], "contacts_read": b["contacts_read"],
                       "contacts_after_refresh": b["contacts_after_refresh"],
                       "stale": b["contacts_read"] != b["contacts_after_refresh"],
                       "dynamics_identical": a["final_state"] == b["final_state"],
                       "max_state_diff": float(np.max(np.abs(np.subtract(a["final_state"], b["final_state"]))))})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
