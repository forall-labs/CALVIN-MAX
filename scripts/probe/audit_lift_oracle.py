"""Colab-only diagnostic: can rebasing alone invalidate an otherwise unchanged successful lift?

Replays saved Base actions, without perturbing physics or querying a policy. Compares the original
oracle reference with the legacy reference replacement at a transport onset. Selects cases by ID.
"""
import argparse
import json
import multiprocessing as mp
import pickle
import sys
from pathlib import Path

ROOT = Path("/content/CALVIN-MAX")
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]


def audit(job):
    case, chain = job
    from run_cases import _setup
    from calvin_max.cases import plan_case
    from calvin_max.oracle import rebase_start_info
    from calvin_max.rollout import physics_step
    base = pickle.loads((ROOT / "results/base/flower_abc/traces" / (chain["chain_id"] + ".pkl")).read_bytes())
    plan = plan_case(case, base)
    if plan["status"] != "planned":
        return {"case_id": case["case_id"], "status": plan["status"]}
    env, runner, init = _setup(chain)
    onset = plan["mid_onset"]
    k, original = runner.replay_prefix(base, init, onset)
    at_onset = env.get_info()
    held = [o.name for o in env.scene.movable_objects
            if env.p.getContactPoints(bodyA=env.robot.robot_uid, bodyB=o.uid, physicsClientId=env.cid)]
    legacy = rebase_start_info(original, at_onset, held)
    task = base.subtasks[k]
    target = plan["info"]["entity"]["target"]
    before = original["scene_info"]["movable_objects"][target]
    after = legacy["scene_info"]["movable_objects"][target]
    stop = base.subtask_starts[k + 1] if k + 1 < len(base.subtask_starts) else len(base.actions)
    original_ok = legacy_ok = False
    for action in base.actions[onset:stop]:
        physics_step(env, action)
        info = env.get_info()
        original_ok |= bool(runner.oracle.get_task_info_for_set(original, info, {task}))
        legacy_ok |= bool(runner.oracle.get_task_info_for_set(legacy, info, {task}))
    return {"case_id": case["case_id"], "task": task, "held": held,
            "reference_z_original": before["current_pos"][2], "reference_z_legacy": after["current_pos"][2],
            "reference_contacts_original": sorted({(c[2], c[4]) for c in before["contacts"]}),
            "reference_contacts_legacy": sorted({(c[2], c[4]) for c in after["contacts"]}),
            "original_success": original_ok, "legacy_rebase_without_push_success": legacy_ok}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    manifest = json.loads((ROOT / "benchmark/manifests/max_v1.json").read_text())["cases"]
    cases = sorted([c for c in manifest if c["track"] == "phase" and c["change_type"] == "arm_push"
                    and c["onset"] == "transport"], key=lambda c: c["case_id"])[:args.n]
    chains = {c["chain_id"]: c for c in json.loads((ROOT / "benchmark/chains/eval_1000.json").read_text())}
    with mp.get_context("spawn").Pool(2, maxtasksperchild=1) as pool:
        reports = list(pool.imap(audit, [(c, chains[c["chain_id"]]) for c in cases]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(reports, indent=1))
    print(json.dumps(reports, indent=1))
