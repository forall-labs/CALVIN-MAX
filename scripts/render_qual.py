"""Rendered frames for the paper's qualitative figures (teaser, supplement Figs. qual_perc / qual_phys).

Two steps. Locally, pick one case per event from a policy's Max-A results and pack the traces it needs:

    python scripts/render_qual.py select --policy flower --base results/base/flower_abc --out results/qual

On a GPU runtime (EGL, the renderer the policies saw), replay the saved actions and render the static camera:

    python scripts/render_qual.py render --sel results/qual/selection.json --out results/qual/png

No policy is queried: every frame comes from replaying actions stored in the Base, Pre and Mid traces, so the
frames are exactly the states those rollouts visited. For each case and arm, in a fresh process:
  base  Base replayed to the Mid onset (the frame just before the event), then onward;
  mid   the same prefix, the event applied at the onset, then the Mid trace's own actions;
  pre   Base replayed to the start of subtask k, the event applied there, then the Pre trace's actions.
Frames are written as <event>-<arm>.png (the onset frame) and <event>-<arm>-tNN.png (NN steps after the onset).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import pickle
import tarfile
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CHAINS = ROOT / "benchmark/chains/eval_1000.json"
PERCEPTUAL = ("camera_shift", "sensor_noise_onset", "illumination_switch", "visual_theme_switch", "distractor_burst")
PHYSICAL = ("target_relocation", "receptacle_relocation", "obstacle_insertion", "arm_push", "grasp_loss")
AFTER = (15, 30, 60, 100)  # steps after the onset for the short sequences


def select(args):
    """One case per event: variant A, Base solved subtask k, both arms ran. Perceptual: prefer Mid solved and Pre
    failed (the timing benefit the paper reports); physical: prefer Mid failed. Ties: lowest chain id."""
    rows = defaultdict(dict)
    for line in (ROOT / f"results/cases/{args.policy}_maxA/results.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r.get("arm") in ("pre", "mid") and r["status"] == "completed":
            rows[r["case_id"]][r["arm"]] = r
    sel = []
    for ev in PERCEPTUAL + PHYSICAL:
        cands = []
        for cid, a in rows.items():
            track, ct, var, k, chain = cid.split("-")
            if ct != ev or var != "A" or set(a) != {"pre", "mid"} or not a["mid"]["base_event_success"]:
                continue
            want = (a["mid"]["event_success"] == 1 and a["pre"]["event_success"] == 0) if ev in PERCEPTUAL else \
                (a["mid"]["event_success"] == 0)
            cands.append((not want, chain, cid))
        if not cands:
            print("no case for", ev)
            continue
        _, chain, cid = min(cands)
        a = rows[cid]
        sel.append({"event": ev, "case_id": cid, "chain_id": chain, "k": int(cid.split("-")[3][1:]),
                    "mid_onset": a["mid"]["onset"], "pre_onset": a["pre"]["onset"], "payload": a["mid"]["payload"],
                    "pre_payload": a["pre"]["payload"],
                    "outcome": {"mid": a["mid"]["event_success"], "pre": a["pre"]["event_success"], "base": 1},
                    "base_trace": f"{args.base}/traces/{chain}.pkl",
                    "mid_trace": f"results/cases/{args.policy}_maxA/traces/{cid}.mid.pkl",
                    "pre_trace": f"results/cases/{args.policy}_maxA/traces/{cid}.pre.pkl"})
        print(f"{ev:22s} {cid}  Pre {a['pre']['event_success']} Mid {a['mid']['event_success']}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "selection.json").write_text(json.dumps(sel, indent=1) + "\n")
    with tarfile.open(out / "traces.tar.gz", "w:gz") as tf:
        for s in sel:
            for key in ("base_trace", "mid_trace", "pre_trace"):
                tf.add(ROOT / s[key], arcname=s[key])
    print(f"wrote {out}/selection.json and traces.tar.gz ({len(sel)} cases)")


def render_one(job):
    s, arm, out, size = job
    try:
        from PIL import Image

        from calvin_max.envs import load_annotations, make_env, make_task_oracle, tracked_entities
        from calvin_max.events import make_event
        from calvin_max.rollout import ChainRunner, physics_step

        chain = next(c for c in json.loads(CHAINS.read_text()) if c["chain_id"] == s["chain_id"])
        env = make_env(egl=True)
        runner = ChainRunner(env, make_task_oracle(), load_annotations(), tracked_entities(env))
        init = (np.array(chain["robot_obs"]), np.array(chain["scene_obs"]))
        base = pickle.loads((ROOT / s["base_trace"]).read_bytes())
        onset = s["pre_onset"] if arm == "pre" else s["mid_onset"]
        runner.replay_prefix(base, init, onset)
        cam = next(c for c in env.cameras if c.name == "static")

        def grab():
            # --size: the same static camera (and any event's view, lighting or noise) at a higher resolution
            if not size:
                return env.get_obs()["rgb_obs"]["rgb_static"]
            w, h = cam.width, cam.height
            cam.width = cam.height = size
            try:
                return env.get_obs()["rgb_obs"]["rgb_static"]
            finally:
                cam.width, cam.height = w, h

        frames = {}
        if arm != "base":
            payload = s["pre_payload"] if arm == "pre" else s["payload"]
            make_event(s["event"], payload)(env)
            env.p.performCollisionDetection(physicsClientId=env.cid)
            tr = pickle.loads((ROOT / s[f"{arm}_trace"]).read_bytes())
        else:
            tr = base
        frames[0] = grab()
        t = onset
        for n in range(1, max(AFTER) + 1):
            if t >= len(tr.actions):
                break
            physics_step(env, tr.actions[t])
            t += 1
            if n in AFTER:
                frames[n] = grab()
        for n, img in frames.items():
            name = f"{s['event']}-{arm}" + (f"-t{n:03d}" if n else "")
            Image.fromarray(img).save(Path(out) / f"{name}.png")
        return {"event": s["event"], "arm": arm, "frames": sorted(frames)}
    except Exception:
        return {"event": s["event"], "arm": arm, "error": traceback.format_exc()}


def render(args):
    sel = json.loads(Path(args.sel).read_text())
    Path(args.out).mkdir(parents=True, exist_ok=True)
    jobs = [(s, arm, args.out, args.size) for s in sel for arm in ("base", "pre", "mid")]
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, maxtasksperchild=1) as pool:
        for r in pool.imap_unordered(render_one, jobs):
            print(json.dumps(r)[:400], flush=True)
    print("RENDER_DONE")


# ---------------------------------------------------------------- family gallery (Figure 1, D-123)
GALLERY_PHYS_PERC = ("target_relocation", "receptacle_relocation", "obstacle_insertion", "camera_shift",
                     "sensor_noise_onset", "illumination_switch", "visual_theme_switch", "distractor_burst")


def gallery_select(args):
    """Candidate scenes for the Figure 1 gallery: one frozen Base state per scene, every event of a family applied
    to that same state (variant B, the stronger benchmark setting, so each change is visible at figure size).

    Physical and perceptual: a push_into_drawer subtask at its Mid onset (the only task with a block target and a
    joint receptacle, so all three physical events realize). Embodiment: over all chains, the lift/place steps where
    the held block is highest, so a grasp loss is visible. Payloads come from cases.realize(), exactly as in evaluation."""
    from calvin_max import cases as C

    man = json.loads((ROOT / "benchmark/manifests/max_v1.json").read_text())["cases"]
    params = {}
    for c in man:
        params.setdefault((c["change_type"], c["variant"]), c["params"])
    chains = {c["chain_id"]: c for c in json.loads(CHAINS.read_text())}
    jobs, traces = [], set()
    for cid, ch in sorted(chains.items()):
        st = ch["subtasks"]
        if "push_into_drawer" not in st:
            continue
        tr = f"{args.base}/traces/{cid}.pkl"
        base = pickle.loads((ROOT / tr).read_bytes())
        k = st.index("push_into_drawer") + 1
        if len(base.subtask_success) < k or not base.subtask_success[k - 1]:
            continue
        case = {"position": k, "task": "push_into_drawer", "onset": "pre_grasp_proximity", "seed": 20261108,
                "case_id": "gallery", "change_type": "", "params": {}}
        t, _ = C.onset_step(case, base)
        if t is None:
            continue
        evs = [(ev, C.realize({**case, "change_type": ev, "params": params[(ev, "B")]}, base, t))
               for ev in GALLERY_PHYS_PERC]
        if any(p is None for _, p in evs):
            continue
        scene = f"{cid}-k{k}"
        jobs += [{"scene": scene, "chain_id": cid, "base_trace": tr, "t": t, "event": None, "payload": None}]
        jobs += [{"scene": scene, "chain_id": cid, "base_trace": tr, "t": t, "event": ev, "payload": p} for ev, p in evs]
        traces.add(tr)
    # embodiment: over all chains, the steps of lift/place subtasks Base solved where the held block is highest
    emb = []
    for cid, ch in sorted(chains.items()):
        tr = f"{args.base}/traces/{cid}.pkl"
        base = pickle.loads((ROOT / tr).read_bytes())
        best = None
        for i, task in enumerate(ch["subtasks"]):
            if not task.startswith(("lift_", "place_")) or len(base.subtask_success) <= i or not base.subtask_success[i]:
                continue
            s = base.subtask_starts[i]
            e = base.subtask_starts[i + 1] if i + 1 < len(base.subtask_starts) else len(base.actions)
            c2 = {"position": i + 1, "task": task, "onset": "transport", "seed": 20261108, "case_id": "gallery",
                  "change_type": "grasp_loss", "params": {}}
            for tt in range(s + 5, e - 3):
                gl = C.realize(c2, base, tt)
                if gl is not None:
                    z = base.entity_pos[gl["object"]][tt - 1][2]
                    if best is None or z > best[0]:
                        best = (z, i + 1, tt, gl, c2, tr, cid)
        if best:
            emb.append(best)
    for z, kk, tt, gl, c2, tr, cid in sorted(emb, key=lambda b: -b[0])[: args.n_emb]:
        base = pickle.loads((ROOT / tr).read_bytes())
        push = C.realize({**c2, "change_type": "arm_push", "params": params[("arm_push", "B")]}, base, tt)
        scene = f"{cid}-k{kk}"
        jobs += [{"scene": scene, "chain_id": cid, "base_trace": tr, "t": tt, "event": ev, "payload": p,
                  "held_z": float(z)} for ev, p in ((None, None), ("grasp_loss", gl), ("arm_push", push))]
        traces.add(tr)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "gallery.json").write_text(json.dumps(jobs, indent=1) + "\n")
    with tarfile.open(out / "gallery_traces.tar.gz", "w:gz") as tf:
        for tr in sorted(traces):
            tf.add(ROOT / tr, arcname=tr)
    print(f"{len(jobs)} gallery frames over {len({j['scene'] for j in jobs})} scenes, {len(traces)} traces")


def _gallery_state(env):
    """World positions the figure annotates (blocks, drawer/slider links, TCP) and the static camera matrices."""
    p, cid = env.p, env.cid
    st = {o.name: list(p.getBasePositionAndOrientation(o.uid, physicsClientId=cid)[0]) for o in env.scene.movable_objects}
    for d in env.scene.doors:
        st[d.name] = list(p.getLinkState(d.uid, d.joint_index, physicsClientId=cid)[0])
    st["tcp"] = list(p.getLinkState(env.robot.robot_uid, env.robot.tcp_link_id, physicsClientId=cid)[0])
    cam = next(c for c in env.cameras if c.name == "static")
    return {"pos": st, "view": list(cam.viewMatrix), "proj": list(cam.projectionMatrix)}


def gallery_one(job):
    j, out, size = job
    name = f"{j['scene']}-{j['event'] or 'original'}"
    try:
        from PIL import Image

        from calvin_max.envs import load_annotations, make_env, make_task_oracle, tracked_entities
        from calvin_max.events import make_event
        from calvin_max.rollout import ChainRunner

        chain = next(c for c in json.loads(CHAINS.read_text()) if c["chain_id"] == j["chain_id"])
        env = make_env(egl=True)
        runner = ChainRunner(env, make_task_oracle(), load_annotations(), tracked_entities(env))
        base = pickle.loads((ROOT / j["base_trace"]).read_bytes())
        runner.replay_prefix(base, (np.array(chain["robot_obs"]), np.array(chain["scene_obs"])), j["t"])
        meta = {"before": _gallery_state(env)}
        if j["event"]:
            meta["effect"] = str(make_event(j["event"], j["payload"])(env))[:300]
            env.p.performCollisionDetection(physicsClientId=env.cid)
        meta["after"] = _gallery_state(env)
        cam = next(c for c in env.cameras if c.name == "static")
        w, h = cam.width, cam.height
        cam.width = cam.height = size
        img = env.get_obs()["rgb_obs"]["rgb_static"]
        cam.width, cam.height = w, h
        Image.fromarray(img).save(Path(out) / f"{name}.png")
        meta.update(size=size, job=j)
        (Path(out) / f"{name}.json").write_text(json.dumps(meta) + "\n")
        return {"frame": name}
    except Exception:
        return {"frame": name, "error": traceback.format_exc()}


def gallery_render(args):
    jobs = json.loads(Path(args.sel).read_text())
    Path(args.out).mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, maxtasksperchild=1) as pool:
        for r in pool.imap_unordered(gallery_one, [(j, args.out, args.size) for j in jobs]):
            print(json.dumps(r)[:400], flush=True)
    print("GALLERY_DONE")


# ---------------------------------------------------------------- human feasibility audit (D-120)
AUDIT_QUOTA = {**{e: 80 for e in ("target_relocation", "receptacle_relocation", "obstacle_insertion")},
               **{e: 40 for e in ("arm_push", "grasp_loss")},
               **{e: 16 for e in PERCEPTUAL}}  # 400; weighted toward events that can make a task infeasible


def audit_select(args):
    """400 realized held-out FLOWER cases, a seeded random draw per event (quota above), spread over variants."""
    rng = np.random.default_rng(20261108)
    rows = {}
    for line in (ROOT / f"results/cases/{args.policy}_maxA/results.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r.get("arm") == "mid" and r["status"] == "completed":
            rows[r["case_id"]] = r
    sel = []
    quota = dict(AUDIT_QUOTA)
    short = sum(max(0, q - sum(c.split("-")[1] == ev for c in rows)) for ev, q in quota.items())
    quota["arm_push"] += short  # grasp loss is realized only with a block in hand; arm push takes its shortfall
    for ev, q in quota.items():
        ids = sorted(c for c in rows if c.split("-")[1] == ev)
        pick = sorted(rng.choice(len(ids), size=min(q, len(ids)), replace=False))
        for i in pick:
            cid = ids[i]
            track, ct, var, k, chain = cid.split("-")
            sel.append({"event": ev, "case_id": cid, "chain_id": chain, "k": int(k[1:]), "variant": var,
                        "mid_onset": rows[cid]["onset"], "payload": rows[cid]["payload"],
                        "base_trace": f"{args.base}/traces/{chain}.pkl"})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "audit_selection.json").write_text(json.dumps(sel, indent=1) + "\n")
    with tarfile.open(out / "audit_traces.tar.gz", "w:gz") as tf:
        for b in sorted({s["base_trace"] for s in sel}):
            tf.add(ROOT / b, arcname=b)
    print(f"{len(sel)} audit cases, {len({s['base_trace'] for s in sel})} Base traces")


def audit_one(job):
    s, out = job
    try:
        from PIL import Image

        from calvin_max.envs import load_annotations, make_env, make_task_oracle, tracked_entities
        from calvin_max.events import make_event
        from calvin_max.rollout import ChainRunner

        chain = next(c for c in json.loads(CHAINS.read_text()) if c["chain_id"] == s["chain_id"])
        env = make_env(egl=True)
        runner = ChainRunner(env, make_task_oracle(), load_annotations(), tracked_entities(env))
        base = pickle.loads((ROOT / s["base_trace"]).read_bytes())
        runner.replay_prefix(base, (np.array(chain["robot_obs"]), np.array(chain["scene_obs"])), s["mid_onset"])
        before = env.get_obs()["rgb_obs"]
        make_event(s["event"], s["payload"])(env)
        env.p.performCollisionDetection(physicsClientId=env.cid)
        after = env.get_obs()["rgb_obs"]
        g = lambda im: np.asarray(Image.fromarray(im).resize((200, 200)))  # noqa: E731  gripper camera is 84 px
        grid = np.concatenate([np.concatenate([before["rgb_static"], after["rgb_static"]], axis=1),
                               np.concatenate([g(before["rgb_gripper"]), g(after["rgb_gripper"])], axis=1)], axis=0)
        Image.fromarray(grid).save(Path(out) / f"{s['case_id']}.jpg", quality=88)
        return {"case_id": s["case_id"], "task": base.subtasks[s["k"] - 1]}
    except Exception:
        return {"case_id": s["case_id"], "error": traceback.format_exc()}


def audit_render(args):
    sel = json.loads(Path(args.sel).read_text())
    Path(args.out).mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("spawn")
    res = []
    with ctx.Pool(args.workers, maxtasksperchild=4) as pool:
        for r in pool.imap_unordered(audit_one, [(s, args.out) for s in sel]):
            res.append(r)
            if "error" in r:
                print(json.dumps(r)[:400], flush=True)
    (Path(args.out) / "tasks.json").write_text(json.dumps(res, indent=1) + "\n")
    print(f"AUDIT_RENDER_DONE {sum('error' not in r for r in res)}/{len(res)}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("select")
    a.add_argument("--policy", default="flower")
    a.add_argument("--base", default="results/base/flower_abc")
    a.add_argument("--out", default="results/qual")
    b = sub.add_parser("render")
    b.add_argument("--sel", default="results/qual/selection.json")
    b.add_argument("--out", default="results/qual/png")
    b.add_argument("--workers", type=int, default=6)
    b.add_argument("--size", type=int, default=0, help="static-camera resolution for the figure (default: 200, as run)")
    c = sub.add_parser("audit-select")
    c.add_argument("--policy", default="flower")
    c.add_argument("--base", default="results/base/flower_abc")
    c.add_argument("--out", default="results/audit")
    d = sub.add_parser("audit-render")
    d.add_argument("--sel", default="results/audit/audit_selection.json")
    d.add_argument("--out", default="results/audit/frames")
    d.add_argument("--workers", type=int, default=8)
    e = sub.add_parser("gallery-select")
    e.add_argument("--base", default="results/base/flower_abc")
    e.add_argument("--out", default="results/qual/gallery")
    e.add_argument("--n-emb", type=int, default=6, help="embodiment candidate scenes (highest held block)")
    f = sub.add_parser("gallery-render")
    f.add_argument("--sel", default="results/qual/gallery/gallery.json")
    f.add_argument("--out", default="results/qual/gallery/png")
    f.add_argument("--workers", type=int, default=8)
    f.add_argument("--size", type=int, default=1024)
    args = ap.parse_args()
    {"select": select, "render": render, "audit-select": audit_select, "audit-render": audit_render,
     "gallery-select": gallery_select, "gallery-render": gallery_render}[args.cmd](args)


if __name__ == "__main__":
    main()
