"""Run manifest cases (Mid and Pre arms) for one policy against its saved Base traces.

    python scripts/run_cases.py --manifest benchmark/manifests/max_v0.json --subset benchmark/manifests/lite_v0.json \
        --base results/base/flower_abc --out results/cases/flower_lite_v0 --H 10 --Q 10 --workers 10

Stages:
1. plan every case offline from the Base trace (cases.plan_case);
2. preflight geometry/clutter events in a fresh process (replay to the onset, apply, check);
3. run each arm in a fresh process.
results.jsonl gets one line per (case, arm) and is resumable. Branch traces go to traces/.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import pickle
import traceback
from pathlib import Path

import numpy as np

PREFLIGHT_TYPES = {"target_relocation", "receptacle_relocation", "obstacle_insertion", "distractor_burst",
                   "reversible_relocation"}
TERMINAL_STATUSES = {"trigger_unreached", "event_unrealizable"}


def resume_state(rows):
    """A failed validity check remains terminal across restarts, even if an old runner bypassed it."""
    done, blocked = set(), set()
    for r in rows:
        if r["status"] in ("completed", "arm_unrealizable"):  # arm_unrealizable: a deterministic property (D-049)
            done.add((r["case_id"], r["arm"]))
        elif r["arm"] is None and r["status"] in TERMINAL_STATUSES:
            blocked.add(r["case_id"])
    return done, blocked


def _setup(chain):
    from calvin_max.envs import load_annotations, make_env, make_task_oracle, tracked_entities
    from calvin_max.rollout import ChainRunner

    env = make_env(egl=True)
    runner = ChainRunner(env, make_task_oracle(), load_annotations(), tracked_entities(env))
    init = (np.array(chain["robot_obs"]), np.array(chain["scene_obs"]))
    return env, runner, init


def preflight_job(job):
    case, plan, chain, base_path = job
    try:
        from calvin_max.events import make_event
        from calvin_max.preflight import preflight, robot_contacts
        from calvin_max.tasks import BLOCKS

        base = pickle.loads(Path(base_path).read_bytes())
        env, runner, init = _setup(chain)
        runner.replay_prefix(base, init, plan["mid_onset"])
        frames, contacts = env.get_obs()["rgb_obs"], robot_contacts(env)
        make_event(case["change_type"], plan["payload"])(env)
        tgt = (plan["info"].get("entity") or {}).get("target")
        rep = preflight(env, frames, contacts, check_visible=[tgt] if tgt in BLOCKS else [])
        return {"case_id": case["case_id"], "preflight": rep}
    except Exception:
        return {"case_id": case["case_id"], "preflight": {"valid": False, "error": traceback.format_exc()[-400:]}}


def arm_job(job):
    case, arm = job[0], job[4]
    try:
        _, plan, chain, base_path, _, port, H, Q, seed, out = job[:10]
        kind = job[10] if len(job) > 10 else "chunked"  # tolerate jobs from an older driver process
        gate = job[11] if len(job) > 11 else None
        from calvin_max.adapter import make_adapter
        from calvin_max.events import make_event

        base = pickle.loads(Path(base_path).read_bytes())
        env, runner, init = _setup(chain)
        adapter = make_adapter(kind, port, H, Q, seed, chain["chain_id"], case["case_id"] + arm)
        if arm == "mid_gate" and gate:
            adapter.gate_every, adapter.gate_tau = gate
        base_arm = "pre" if arm == "pre" else "mid"  # mid_oracle / mid_gate / mid_rewind are Mid variants (D-044, D-049)
        onset = plan["mid_onset"] if base_arm == "mid" else plan["pre_onset"]
        tr = runner.run_branch(adapter, base, init, base_arm, onset, make_event(case["change_type"], plan["payload"]),
                               oracle_flush=(arm == "mid_oracle"), rewind=(arm == "mid_rewind"))
        adapter.close()
        tr.snapshots = {}
        (Path(out) / "traces" / f"{case['case_id']}.{arm}.pkl").write_bytes(pickle.dumps(tr))
        k = case["position"] - 1
        return {"case_id": case["case_id"], "arm": arm, "status": "completed", "onset": onset, "payload": plan["payload"],
                "pending_at_onset": tr.pending_at_onset,
                "event_success": int(len(tr.subtask_success) > k and tr.subtask_success[k]),
                "subtask_success": tr.subtask_success, "chain_length": tr.chain_length,
                "side_effects_event_subtask": tr.side_effects.get(k, []),
                "gate_replacements": sum(1 for g in tr.gate_log if g[2]), "gate_probes": len(tr.gate_log),
                "tick_log": getattr(tr, "tick_log", [])}
    except Exception as e:
        if type(e).__name__ == "RewindUnrealizable":  # a property of the case, not an infra failure
            return {"case_id": case["case_id"], "arm": arm, "status": "arm_unrealizable", "detail": str(e)}
        return {"case_id": case["case_id"], "arm": arm, "status": "infra_error", "error": traceback.format_exc()[-800:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--subset", type=Path, default=None)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--chains", type=Path, default=Path("benchmark/chains/eval_1000.json"))
    ap.add_argument("--onset", type=Path, default=None, help="calibration JSON with chosen_radius")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--adapter", default="chunked", choices=["chunked", "xvla", "history", "history:13", "dda", "dda:live"])
    ap.add_argument("--port", default="6100", help="port or comma-separated replica ports")
    ap.add_argument("--H", type=int, required=True)
    ap.add_argument("--Q", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--types", default=None, help="comma-separated change_types to run")
    ap.add_argument("--arms", default=None, help="override the manifest arms, e.g. mid_oracle,mid_gate (D-044)")
    ap.add_argument("--tracks", default=None, help="comma-separated tracks to run (core,new,phase,null,intent)")
    ap.add_argument("--gate-every", type=int, default=2)
    ap.add_argument("--gate-tau", type=float, default=None)
    args = ap.parse_args()

    from calvin_max.cases import ONSET, plan_case

    onset = dict(ONSET)
    if args.onset:
        onset["radius"] = float(json.loads(args.onset.read_text())["chosen_radius"])
    man = json.loads(args.manifest.read_text())
    cases = man["cases"]
    if args.subset:
        keep = set(json.loads(args.subset.read_text())["case_ids"])
        cases = [c for c in cases if c["case_id"] in keep]
    if args.types:
        cases = [c for c in cases if c["change_type"] in args.types.split(",")]
    if args.tracks:
        cases = [c for c in cases if c["track"] in args.tracks.split(",")]
    if args.arms:
        cases = [{**c, "arms": args.arms.split(",")} for c in cases]
    gate = (args.gate_every, args.gate_tau) if args.gate_tau is not None else None
    chains = {c["chain_id"]: c for c in json.loads(args.chains.read_text())}
    (args.out / "traces").mkdir(parents=True, exist_ok=True)
    res_path = args.out / "results.jsonl"
    rows = [json.loads(line) for line in res_path.read_text().splitlines() if line.strip()] if res_path.exists() else []
    done, blocked = resume_state(rows)

    def write(r):
        with res_path.open("a") as f:
            f.write(json.dumps(r) + "\n")

    # 1. plan
    plans = {}
    for c in cases:
        if c["case_id"] in blocked:
            continue
        base_path = args.base / "traces" / f"{c['chain_id']}.pkl"
        if not base_path.exists():  # Base rollout failed (infra); rerun run_base.py, then this resumes
            write({"case_id": c["case_id"], "arm": None, "status": "infra_error", "error": "missing base trace"})
            continue
        base = pickle.loads(base_path.read_bytes())
        plans[c["case_id"]] = (c, plan_case(c, base, onset), str(base_path))
        p = plans[c["case_id"]][1]
        if p["status"] != "planned":
            k = c["position"] - 1
            write({"case_id": c["case_id"], "arm": None, "status": p["status"], "detail": p.get("reason") or p.get("info"),
                   "base_event_success": int(len(base.subtask_success) > k and base.subtask_success[k]),
                   "base_chain_length": base.chain_length})
    print(json.dumps({"cases": len(cases), "planned": sum(v[1]["status"] == "planned" for v in plans.values())}), flush=True)

    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, maxtasksperchild=1) as pool:
        # 2. preflight
        pf_jobs = [(c, p, chains[c["chain_id"]], bp) for c, p, bp in plans.values()
                   if p["status"] == "planned" and c["change_type"] in PREFLIGHT_TYPES
                   and not all((c["case_id"], arm) in done for arm in c["arms"])]
        invalid = set()
        for r in pool.imap_unordered(preflight_job, pf_jobs):
            if not r["preflight"].get("valid"):
                invalid.add(r["case_id"])
                p = plans[r["case_id"]][1]
                write({"case_id": r["case_id"], "arm": None,
                       "status": "infra_error" if "error" in r["preflight"] else "event_unrealizable",
                       "detail": r["preflight"], "base_event_success": p["base_event_success"],
                       "base_chain_length": p["base_chain_length"]})
        print(json.dumps({"preflight": len(pf_jobs), "invalid": len(invalid)}), flush=True)
        # 3. arms
        jobs = []
        for c, p, bp in plans.values():  # cases without a Base trace are not in plans
            if p["status"] != "planned" or c["case_id"] in invalid:
                continue
            for arm in c["arms"]:
                if (c["case_id"], arm) not in done:
                    jobs.append((c, p, chains[c["chain_id"]], bp, arm, args.port, args.H, args.Q, args.seed, str(args.out), args.adapter, gate))
        n = 0
        for r in pool.imap_unordered(arm_job, jobs):
            r["base_event_success"] = plans[r["case_id"]][1].get("base_event_success")
            r["base_chain_length"] = plans[r["case_id"]][1].get("base_chain_length")
            write(r)
            n += 1
            if n % 25 == 0 or n == len(jobs):
                print(json.dumps({"arms_done": n, "of": len(jobs)}), flush=True)


if __name__ == "__main__":
    main()
