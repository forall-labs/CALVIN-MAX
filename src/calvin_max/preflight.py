"""Preflight validity checks run in the simulator at the onset state (RESEARCH_PLAN §3.3, checks 1-5).

`preflight(env, event, effect_entities)` applies nothing itself: call it right after the event, in the
same fresh process that replayed the prefix. It returns a dict of measurements and `valid`.
"""

from __future__ import annotations

import numpy as np

STATIC = "static"
MIN_VISIBLE_PX = 6  # target pixels in the 200x200 static view
MAX_DRIFT_M = 0.005  # over 20 hold steps
MIN_PIXEL_CHANGE = 0.05  # mean |delta| (0-255) on at least one camera, for observation-changing events


def visible_pixels(env, body_uid, cam_name=STATIC):
    cam = next(c for c in env.cameras if c.name == cam_name)
    seg = env.p.getCameraImage(cam.width, cam.height, cam.viewMatrix, cam.projectionMatrix,
                               physicsClientId=env.cid)[4]
    seg = np.asarray(seg).reshape(cam.height, cam.width)
    return int(np.sum((seg & ((1 << 24) - 1)) == body_uid))


def hold(env, n=20):
    grip = 1.0 if env.robot.gripper_action > 0 else -1.0
    for _ in range(n):
        env.step(np.r_[np.zeros(6), grip])


def robot_contacts(env):
    return {(c[2], c[4]) for c in env.p.getContactPoints(bodyA=env.robot.robot_uid, physicsClientId=env.cid)}


def preflight(env, frames_before, contacts_before, check_visible=(), require_pixel_change=True, hold_steps=20):
    """frames_before = env.get_obs()['rgb_obs'] and contacts_before = robot_contacts(env), both captured
    just before the event."""
    rep = {}
    # Contacts are stale until a collision pass: after a teleport they still describe the pre-event scene (D-053).
    env.p.performCollisionDetection(physicsClientId=env.cid)
    rep["new_robot_contacts"] = sorted(robot_contacts(env) - set(contacts_before))
    after = env.get_obs()["rgb_obs"]
    rep["pixel_change"] = {k: float(np.abs(after[k].astype(float) - frames_before[k].astype(float)).mean())
                           for k in frames_before}
    uid = {o.name: o.uid for o in env.scene.movable_objects}
    rep["visible_px"] = {name: visible_pixels(env, uid[name]) for name in check_visible}
    # A block in the gripper sags a few mm while the arm holds still; that is not scene instability (D-033).
    held = {n for n in uid if env.p.getContactPoints(bodyA=env.robot.robot_uid, bodyB=uid[n], physicsClientId=env.cid)}
    free = [n for n in uid if n not in held]
    p0 = {n: np.array(env.p.getBasePositionAndOrientation(uid[n], physicsClientId=env.cid)[0]) for n in free}
    hold(env, hold_steps)
    p1 = {n: np.array(env.p.getBasePositionAndOrientation(uid[n], physicsClientId=env.cid)[0]) for n in free}
    rep["max_drift_m"] = float(max((np.linalg.norm(p1[n] - p0[n]) for n in free), default=0.0))
    rep["held_excluded"] = sorted(held)
    rep["valid"] = bool(
        rep["max_drift_m"] <= MAX_DRIFT_M
        and not rep["new_robot_contacts"]
        and all(v >= MIN_VISIBLE_PX for v in rep["visible_px"].values())
        and (not require_pixel_change or max(rep["pixel_change"].values()) >= MIN_PIXEL_CHANGE)
    )
    return rep
