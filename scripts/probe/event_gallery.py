"""Before/after frames and first preflight numbers for each event (one fresh process per event).

    python scripts/probe/event_gallery.py --out results/gallery

For each event, on chain E0000's initial state, it records:
- a PNG strip (static + gripper, before | after);
- the mean absolute pixel change;
- drift of every block over 20 hold steps after the event (stability);
- whether the robot touches anything new at onset.

It also checks whether getCameraImage's light parameters change the EGL render (illumination support).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import traceback
from pathlib import Path

import numpy as np

CHAINS = Path(__file__).resolve().parents[2] / "benchmark/chains/eval_1000.json"


def payloads(env):
    red = env.p.getBasePositionAndOrientation(next(o.uid for o in env.scene.movable_objects if o.name == "block_red"),
                                              physicsClientId=env.cid)[0]
    tcp = np.array(env.robot.get_observation()[1]["tcp_pos"])
    rng = np.random.default_rng(0)
    table = np.array([[0.0, -0.15], [0.35, -0.03]])
    distract = [{"pos": [float(rng.uniform(*table[:, 0])), float(rng.uniform(*table[:, 1])), 0.48],
                 "half_extents": [0.015, 0.015, 0.015]} for _ in range(5)]
    mid = (tcp + np.array(red)) / 2
    return {
        "block_relocation_8cm": ("target_relocation", {"object": "block_red", "pos": [red[0] + 0.08, red[1], red[2]]}),
        "joint_relocation_slider": ("target_relocation", {"joint": "base__slide", "value": 0.15}),
        "camera_shift_B": ("camera_shift", {"translate_m": [0, 0, 0.0142], "yaw_deg": 1.42, "pitch_deg": 0.71, "fov_delta_deg": 1.0}),
        "sensor_noise_static": ("sensor_noise_onset", {"cameras": ["static"], "sigma": 0.08, "occlusion_frac": 0.1, "seed": 1}),
        "distractor_burst_5": ("distractor_burst", {"objects": distract}),
        "obstacle_box": ("obstacle_insertion", {"pos": [float(mid[0]), float(mid[1]), 0.50], "half_extents": [0.02, 0.02, 0.04]}),
    }


def block_poses(env):
    return {o.name: np.array(env.p.getBasePositionAndOrientation(o.uid, physicsClientId=env.cid)[0])
            for o in env.scene.movable_objects}


def run_one(name):
    try:
        from calvin_max.envs import make_env
        from calvin_max.events import make_event

        chain = json.loads(CHAINS.read_text())[0]
        env = make_env(egl=True)
        env.reset(robot_obs=np.array(chain["robot_obs"]), scene_obs=np.array(chain["scene_obs"]))
        for _ in range(5):
            env.step(np.r_[np.zeros(6), 1.0])
        before = env.get_obs()["rgb_obs"]
        contacts_before = len(env.p.getContactPoints(bodyA=env.robot.robot_uid, physicsClientId=env.cid))
        change_type, payload = payloads(env)[name]
        effect = make_event(change_type, payload)(env)
        after = env.get_obs()["rgb_obs"]
        contacts_after = len(env.p.getContactPoints(bodyA=env.robot.robot_uid, physicsClientId=env.cid))
        p0 = block_poses(env)
        for _ in range(20):
            env.step(np.r_[np.zeros(6), 1.0])
        p1 = block_poses(env)
        rep = {
            "event": name, "change_type": change_type, "effect": effect,
            "pixel_change": {k: float(np.abs(after[k].astype(float) - before[k].astype(float)).mean()) for k in before},
            "max_block_drift_20_steps_m": float(max(np.linalg.norm(p1[k] - p0[k]) for k in p0)),
            "new_robot_contacts": contacts_after - contacts_before,
        }
        out = Path("results/gallery")
        out.mkdir(parents=True, exist_ok=True)
        strip = np.concatenate([before["rgb_static"], after["rgb_static"]], axis=1)
        grip = np.concatenate([before["rgb_gripper"], after["rgb_gripper"]], axis=1)
        grip = np.kron(grip, np.ones((2, 2, 1), dtype=np.uint8))[: strip.shape[0]]  # 84 -> 168 px tall
        pad = np.zeros((strip.shape[0] - grip.shape[0], grip.shape[1], 3), np.uint8)
        img = np.concatenate([strip, np.concatenate([grip, pad], axis=0)], axis=1)
        import cv2

        cv2.imwrite(str(out / f"{name}.png"), img[:, :, ::-1])
        return rep
    except Exception:
        return {"event": name, "error": traceback.format_exc()}


def light_test(_):
    try:
        from calvin_max.envs import make_env

        chain = json.loads(CHAINS.read_text())[0]
        env = make_env(egl=True)
        env.reset(robot_obs=np.array(chain["robot_obs"]), scene_obs=np.array(chain["scene_obs"]))
        cam = next(c for c in env.cameras if c.name == "static")
        base = env.p.getCameraImage(cam.width, cam.height, cam.viewMatrix, cam.projectionMatrix,
                                    physicsClientId=env.cid)[2]
        lit = env.p.getCameraImage(cam.width, cam.height, cam.viewMatrix, cam.projectionMatrix,
                                   lightDirection=[-1, 1, 0.3], lightColor=[1.0, 0.6, 0.4],
                                   lightAmbientCoeff=0.2, lightDiffuseCoeff=0.9, physicsClientId=env.cid)[2]
        a, b = np.asarray(base, float).reshape(cam.height, cam.width, 4), np.asarray(lit, float).reshape(cam.height, cam.width, 4)
        return {"event": "light_param_test", "pixel_change_from_light_params": float(np.abs(a - b)[..., :3].mean())}
    except Exception:
        return {"event": "light_param_test", "error": traceback.format_exc()}


def main():
    names = ["block_relocation_8cm", "joint_relocation_slider", "camera_shift_B",
             "sensor_noise_static", "distractor_burst_5", "obstacle_box"]
    with mp.get_context("spawn").Pool(4, maxtasksperchild=1) as pool:
        res = pool.map(run_one, names) + pool.map(light_test, [0])
    for r in res:
        print(json.dumps(r if "error" not in r else {**r, "error": r["error"][-600:]}))


if __name__ == "__main__":
    main()
