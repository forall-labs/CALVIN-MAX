"""Phase 0 probe: state layout, replay determinism, branch restore, throughput.

Runs against calvin_env alone (no policy, no dataset). It composes the env config from
calvin_env/conf instead of a dataset's merged_config.yaml. Before any policy run, we still
check that this matches the official D validation config.

    python scripts/probe/phase0_probe.py --out results/phase0 [--egl] [--compare other.json]

Run it twice in separate processes, passing the first run's JSON to --compare, to test
cross-process determinism (Phase 1 gate).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np

STEPS = 600
BRANCH_AT = 300
BRANCH_LEN = 100


def make_env(egl: bool, cameras: str):
    import calvin_env
    import hydra
    from hydra import compose, initialize_config_dir

    conf_dir = str(Path(calvin_env.__file__).resolve().parents[1] / "conf")
    with initialize_config_dir(config_dir=conf_dir, version_base=None):
        cfg = compose(
            config_name="config_data_collection",
            overrides=[
                "use_vr=false",
                f"cameras={cameras}",
                "scene=calvin_scene_D_eval",
                "robot=panda_longer_finger",
                f"env.use_egl={str(egl).lower()}",
                f"data_path={Path(calvin_env.__file__).resolve().parents[1] / 'data'}",
            ],
        )
    return hydra.utils.instantiate(cfg.env, show_gui=False, use_vr=False, use_scene_info=True)


def scripted_actions(n: int, seed: int = 0) -> np.ndarray:
    """Smooth relative actions with gripper toggles; float64, the gripper exactly +-1."""
    rng = np.random.default_rng(seed)
    a = np.cumsum(rng.normal(0, 0.15, size=(n, 6)), axis=0)
    a = np.clip(a / (1 + np.abs(a)), -1, 1)  # bounded random walk
    grip = np.where((np.arange(n) // 60) % 2 == 0, 1.0, -1.0)
    return np.concatenate([a, grip[:, None]], axis=1)


def state_hash(env) -> str:
    s = env.get_state_obs()
    buf = np.concatenate([s["robot_obs"], s["scene_obs"]]).astype(np.float64).tobytes()
    return hashlib.sha256(buf).hexdigest()[:16]


def physics_step(env, action):
    """env.step without rendering: same physics path, no camera calls (prefix replay)."""
    env.robot.apply_action(action)
    for _ in range(env.action_repeat):
        env.p.stepSimulation(physicsClientId=env.cid)
    env.scene.step()


def rollout(env, actions, render: bool):
    env.reset()
    hashes = []
    for a in actions:
        if render:
            env.step(a)
        else:
            physics_step(env, a)
        hashes.append(state_hash(env))
    return hashes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results/phase0"))
    ap.add_argument("--egl", action="store_true")
    ap.add_argument("--cameras", default="static_and_gripper")
    ap.add_argument("--compare", type=Path, help="JSON from another process to compare hashes with")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    env = make_env(args.egl, args.cameras)
    actions = scripted_actions(STEPS)
    report: dict = {"platform": platform.platform(), "egl": args.egl, "cameras": args.cameras}

    # 1. Layout.
    env.reset()
    s = env.get_state_obs()
    report["robot_obs_len"] = int(len(s["robot_obs"]))
    report["robot_obs_labels"] = env.robot.get_observation_labels()
    report["scene_obs_len"] = int(len(s["scene_obs"]))
    report["scene_obs_initial"] = s["scene_obs"].tolist()
    report["doors"] = [d.name for d in env.scene.doors]
    report["buttons"] = [b.name for b in env.scene.buttons]
    report["switches"] = [w.name for w in env.scene.switches]
    report["lights"] = [l.name for l in env.scene.lights]
    report["movable_objects"] = {o.name: o.uid for o in env.scene.movable_objects}
    report["action_repeat"] = env.action_repeat
    info = env.get_info()
    report["info_keys"] = {k: sorted(v.keys()) for k, v in info["scene_info"].items()}

    # 2. Same-process determinism: rendered vs rendered, and rendered vs physics-only.
    t0 = time.perf_counter()
    h_render_a = rollout(env, actions, render=True)
    t_render = time.perf_counter() - t0
    h_render_b = rollout(env, actions, render=True)
    t0 = time.perf_counter()
    h_phys = rollout(env, actions, render=False)
    t_phys = time.perf_counter() - t0
    report["same_process_render_match"] = h_render_a == h_render_b
    report["render_vs_physics_only_match"] = h_render_a == h_phys
    report["first_mismatch_render_vs_phys"] = next(
        (i for i, (x, y) in enumerate(zip(h_render_a, h_phys)) if x != y), None
    )
    report["steps_per_s_render"] = STEPS / t_render
    report["steps_per_s_physics_only"] = STEPS / t_phys

    # 3. Branch via saveState/restoreState vs straight continuation (D-013: speedup only).
    env.reset()
    for a in actions[:BRANCH_AT]:
        physics_step(env, a)
    sid = env.p.saveState(physicsClientId=env.cid)
    straight = []
    for a in actions[BRANCH_AT : BRANCH_AT + BRANCH_LEN]:
        physics_step(env, a)
        straight.append(state_hash(env))
    env.p.restoreState(stateId=sid, physicsClientId=env.cid)
    restored = []
    for a in actions[BRANCH_AT : BRANCH_AT + BRANCH_LEN]:
        physics_step(env, a)
        restored.append(state_hash(env))
    report["save_restore_match"] = straight == restored
    report["first_mismatch_save_restore"] = next(
        (i for i, (x, y) in enumerate(zip(straight, restored)) if x != y), None
    )

    # 4. Cross-process comparison.
    report["hashes"] = h_render_a
    if args.compare:
        other = json.loads(args.compare.read_text())["hashes"]
        report["cross_process_match"] = other == h_render_a
        report["first_mismatch_cross_process"] = next(
            (i for i, (x, y) in enumerate(zip(other, h_render_a)) if x != y), None
        )

    out = args.out / f"probe_{int(time.time())}.json"
    out.write_text(json.dumps(report, indent=1))
    summary = {k: v for k, v in report.items() if k not in ("hashes", "scene_obs_initial", "robot_obs_labels")}
    print(json.dumps(summary, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
