"""Colab-only: does the preflight's stale contact read change any validity verdict? (D-048 open item)

preflight() reads robot contacts right after the event, before any collision pass, so `new_robot_contacts` and
the held-object set (excluded from the drift check) describe the pre-event scene. For each preflighted case
(completed or rejected), replays the policy's Base to the Mid onset, applies the event and compares the
contact measurements without and with p.performCollisionDetection(). With --cases (e.g. the flagged ids of a
first pass), also runs the full preflight after the collision pass: the verdict the fixed check would give.
Unflagged cases need no rerun: their contact inputs match and the pass leaves dynamics bit-identical.

    python scripts/probe/audit_preflight.py --base results/base/flower_abc \
        --results results/cases/flower_lite_v1/results.jsonl ... --out results/analysis/preflight_audit_flower.json
"""
import argparse
import json
import multiprocessing as mp
import pickle
import sys
from pathlib import Path

ROOT = Path("/content/CALVIN-MAX")
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

PREFLIGHT_TYPES = {"target_relocation", "receptacle_relocation", "obstacle_insertion", "distractor_burst"}


def held(env):
    uid = {o.name: o.uid for o in env.scene.movable_objects}
    return sorted(n for n in uid
                  if env.p.getContactPoints(bodyA=env.robot.robot_uid, bodyB=uid[n], physicsClientId=env.cid))


def audit(job):
    case, status, chain, base_path, full = job  # one case per process, as in run_cases.preflight_job
    try:
        from run_cases import _setup
        from calvin_max.cases import plan_case
        from calvin_max.events import make_event
        from calvin_max.preflight import preflight, robot_contacts
        from calvin_max.tasks import BLOCKS
        base = pickle.loads(Path(base_path).read_bytes())
        plan = plan_case(case, base)
        if plan["status"] != "planned":
            return {"case_id": case["case_id"], "status": status, "error": "replan: " + plan["status"]}
        env, runner, init = _setup(chain)
        runner.replay_prefix(base, init, plan["mid_onset"])
        frames, before = env.get_obs()["rgb_obs"], robot_contacts(env)
        make_event(case["change_type"], plan["payload"])(env)
        stale_new, stale_held = sorted(robot_contacts(env) - before), held(env)
        env.p.performCollisionDetection(physicsClientId=env.cid)
        fresh_new, fresh_held = sorted(robot_contacts(env) - before), held(env)
        row = {"case_id": case["case_id"], "status": status, "change_type": case["change_type"],
               "stale_new": stale_new, "fresh_new": fresh_new, "stale_held": stale_held, "fresh_held": fresh_held,
               "contact_verdict_changed": bool(stale_new) != bool(fresh_new), "held_changed": stale_held != fresh_held}
        if full:
            tgt = (plan["info"].get("entity") or {}).get("target")
            rep = preflight(env, frames, before, check_visible=[tgt] if tgt in BLOCKS else [])
            row.update(fixed_valid=rep["valid"], fixed_report=rep,
                       verdict_changed=rep["valid"] != (status == "accepted"))
        return row
    except Exception:
        import traceback
        return {"case_id": case["case_id"], "status": status, "error": traceback.format_exc()[-400:]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--results", type=Path, nargs="+", required=True, help="later files override earlier ones per case")
    ap.add_argument("--cases", type=Path, default=None, help="JSON list of case ids: audit only these, with full preflight")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    manifest = {}
    for name in ("max_v0", "max_v1"):  # v1 wins for ids in both
        manifest.update({c["case_id"]: c for c in json.loads((ROOT / f"benchmark/manifests/{name}.json").read_text())["cases"]})
    status = {}
    for path in args.results:
        seen = {}
        for line in path.read_text().splitlines():
            r = json.loads(line) if line.strip() else None
            if r and manifest.get(r["case_id"], {}).get("change_type") in PREFLIGHT_TYPES:
                if r["status"] == "completed" or (r["status"] == "event_unrealizable" and "error" not in (r.get("detail") or {})):
                    seen.setdefault(r["case_id"], "rejected" if r["status"] == "event_unrealizable" else "accepted")
        status.update(seen)
    chains = {c["chain_id"]: c for c in json.loads((ROOT / "benchmark/chains/eval_1000.json").read_text())}
    only = set(json.loads(args.cases.read_text())) if args.cases else None
    jobs = [(manifest[cid], s, chains[manifest[cid]["chain_id"]],
             str(args.base / "traces" / f"{manifest[cid]['chain_id']}.pkl"), only is not None)
            for cid, s in sorted(status.items()) if only is None or cid in only]
    print(json.dumps({"cases": len(jobs)}), flush=True)
    rows = []
    with mp.get_context("spawn").Pool(args.workers, maxtasksperchild=1) as pool:
        for i, r in enumerate(pool.imap_unordered(audit, jobs), 1):
            rows.append(r)
            if i % 100 == 0:
                print(json.dumps({"done": i, "of": len(jobs)}), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".rows.json").write_text(json.dumps(rows))
    summary = {}
    for r in rows:
        s = summary.setdefault(f'{manifest[r["case_id"]]["change_type"]}:{r["status"]}',
                               {"cases": 0, "contact_verdict_changed": 0, "held_changed": 0, "verdict_changed": 0,
                                "errors": 0})
        s["cases"] += 1
        for k in ("contact_verdict_changed", "held_changed", "verdict_changed"):
            s[k] += int(bool(r.get(k)))
        s["errors"] += int("error" in r)
    flagged = sorted(r["case_id"] for r in rows if r.get("contact_verdict_changed") or r.get("held_changed"))
    args.out.write_text(json.dumps({"summary": summary, "flagged_case_ids": flagged,
                                    "rows": sorted(rows, key=lambda r: r["case_id"])}, indent=1))
    print(json.dumps(summary, indent=1))
