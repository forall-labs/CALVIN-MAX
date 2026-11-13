"""Colab-only: which completed cases does the D-048 oracle fix change?

For each case with a completed arm, replays the policy's Base to the Pre and Mid onsets and builds the subtask's
oracle reference (start_info) two ways:
  legacy: post-event info read without a collision pass; arm_push replaces carried objects' references;
  fixed:  contacts refreshed by p.performCollisionDetection(); carried objects' references are translated.
The refresh leaves dynamics bit-identical (audit_contacts.py), so a case whose references match under both
rules would rerun bit-identically and keeps its result. Only cases whose references differ need a rerun.

    python scripts/probe/audit_reference.py --base results/base/flower_abc \
        --results results/cases/flower_lite_v1/results.jsonl ... --out results/analysis/reference_audit_flower.json
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

PHYSICAL = {"target_relocation", "receptacle_relocation", "obstacle_insertion", "distractor_burst",
            "grasp_loss", "arm_push"}


def _canon(x):
    """JSON-comparable form of what the CALVIN oracle reads. tasks.py uses only a contact's body (c[2]) and link
    (c[4]); the force fields differ between a solver step and a bare collision pass and are dropped."""
    if isinstance(x, dict):
        return {str(k): (sorted({(int(c[2]), int(c[4])) for c in v}) if k == "contacts" else _canon(v))
                for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [_canon(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    return x


def references(runner, env, base, init, onset, case, payload, arm):
    from calvin_max.events import make_event
    from calvin_max.oracle import rebase_start_info
    _, start_info = runner.replay_prefix(base, init, onset)
    effect = make_event(case["change_type"], payload)(env)
    stale = env.get_info()
    env.p.performCollisionDetection(physicsClientId=env.cid)
    fresh = env.get_info()
    if arm == "pre":
        return _canon(stale), _canon(fresh)
    moved, doors, lights = effect.get("moved_objects", ()), effect.get("moved_doors", ()), effect.get("lights", ())
    shifted = effect.get("shifted_objects") or {}
    legacy = rebase_start_info(start_info, stale, list(moved) + list(shifted), doors, lights)
    fixed = rebase_start_info(start_info, fresh, moved, doors, lights, shifted)
    return _canon(legacy), _canon(fixed)


def audit(job):
    case, arm, chain, base_path = job  # one arm per process: an event can leave state env.reset() keeps
    try:
        from run_cases import _setup
        from calvin_max.cases import plan_case
        base = pickle.loads(Path(base_path).read_bytes())
        plan = plan_case(case, base)
        if plan["status"] != "planned":
            return {"case_id": case["case_id"], "arm": arm, "error": "replan: " + plan["status"]}
        env, runner, init = _setup(chain)
        onset = plan["pre_onset"] if arm == "pre" else plan["mid_onset"]
        legacy, fixed = references(runner, env, base, init, onset, case, plan["payload"], arm)
        return {"case_id": case["case_id"], "arm": arm, "change_type": case["change_type"], "task": case["task"],
                "changed": legacy != fixed}
    except Exception:
        import traceback
        return {"case_id": case["case_id"], "arm": arm, "error": traceback.format_exc()[-400:]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--results", type=Path, nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=11)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    manifest = {}
    for name in ("max_v0", "max_v1"):  # v1 wins for ids in both
        manifest.update({c["case_id"]: c for c in json.loads((ROOT / f"benchmark/manifests/{name}.json").read_text())["cases"]})
    arms = {}
    for path in args.results:
        for line in path.read_text().splitlines():
            r = json.loads(line) if line.strip() else None
            if r and r["status"] == "completed" and r["arm"] and manifest.get(r["case_id"], {}).get("change_type") in PHYSICAL:
                arms.setdefault(r["case_id"], set()).add(r["arm"])
    chains = {c["chain_id"]: c for c in json.loads((ROOT / "benchmark/chains/eval_1000.json").read_text())}
    jobs = [(manifest[cid], arm, chains[manifest[cid]["chain_id"]],
             str(args.base / "traces" / f"{manifest[cid]['chain_id']}.pkl"))
            for cid, a in sorted(arms.items()) for arm in sorted({"pre" if x == "pre" else "mid" for x in a})]
    print(json.dumps({"cases": len(arms), "arm_jobs": len(jobs)}), flush=True)
    rows = []
    with mp.get_context("spawn").Pool(args.workers, maxtasksperchild=1) as pool:
        for i, r in enumerate(pool.imap_unordered(audit, jobs), 1):
            rows.append(r)
            if i % 100 == 0:
                print(json.dumps({"done": i, "of": len(jobs)}), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".rows.json").write_text(json.dumps(rows))  # before summarising, so a summary bug loses nothing
    summary = {}
    for r in rows:
        s = summary.setdefault(f'{manifest[r["case_id"]]["change_type"]}:{r["arm"]}', {"arms": 0, "changed": 0, "errors": 0})
        s["arms"] += 1
        s["changed"] += int(bool(r.get("changed")))
        s["errors"] += int("error" in r)
    rerun = {r["case_id"] for r in rows if r.get("changed") or "error" in r}  # an unauditable case is rerun too
    report = {"summary": summary, "rerun_case_ids": sorted(rerun),
              "rows": sorted(rows, key=lambda r: r["case_id"])}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    print(json.dumps(summary, indent=1))
