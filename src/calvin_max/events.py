"""Event implementations (RESEARCH_PLAN §3.3). Each event is `fn(env, payload) -> effect`.

`payload` is fully resolved when the manifest is built (D-009): poses, magnitudes, noise seeds.
Evaluation never samples. `effect` tells the runner what changed:
  moved_objects / moved_doors / lights  -> rebased in start_info for mid-subtask onsets (oracle.py)
  shifted_objects                       -> carried objects: reference translated by the displacement (D-048)
  tick                                  -> transient events: tick(env, i) before step i of the event subtask;
                                           may return restore_objects (reference restored to pre-event, D-050)
  goal                                  -> instruction_revision: the subtask now checked and instructed
  lang                                  -> paraphrase_swap: new instruction text, same task
Observation events (camera, sensor noise) persist by modifying the env instance; each rollout runs in
a fresh process (D-018), so nothing leaks between rollouts.
"""

from __future__ import annotations

import zlib

import numpy as np

# --- helpers ----------------------------------------------------------------------------------------


def _body(env, name):
    for o in env.scene.movable_objects:
        if o.name == name:
            return o.uid
    raise KeyError(name)


def _door(env, joint_name):
    for d in env.scene.doors:
        if d.name == joint_name:
            return d
    raise KeyError(joint_name)


def _set_block_pose(env, name, pos, yaw_delta=0.0):
    uid = _body(env, name)
    _, orn = env.p.getBasePositionAndOrientation(uid, physicsClientId=env.cid)
    if yaw_delta:
        e = list(env.p.getEulerFromQuaternion(orn))
        e[2] += yaw_delta
        orn = env.p.getQuaternionFromEuler(e)
    env.p.resetBasePositionAndOrientation(uid, pos, orn, physicsClientId=env.cid)
    env.p.resetBaseVelocity(uid, [0, 0, 0], [0, 0, 0], physicsClientId=env.cid)


# --- geometry ---------------------------------------------------------------------------------------


def block_relocation(env, payload):
    """target_relocation / receptacle_relocation for a block: payload {object, pos, yaw_delta} (resolve.py)."""
    _set_block_pose(env, payload["object"], payload["pos"], payload.get("yaw_delta", 0.0))
    return {"moved_objects": [payload["object"]]}


def reversible_relocation(env, payload):
    """A block relocation that is undone: payload as block_relocation plus return_after (steps, D-050).

    The block is teleported back to its exact pre-event pose `return_after` steps later, unless the robot is
    touching it then (the policy has already engaged the moved block; the return is skipped and logged).
    """
    uid = _body(env, payload["object"])
    pos0, orn0 = env.p.getBasePositionAndOrientation(uid, physicsClientId=env.cid)
    _set_block_pose(env, payload["object"], payload["pos"], payload.get("yaw_delta", 0.0))

    def tick(env, i):
        if i != payload["return_after"]:
            return None
        env.p.performCollisionDetection(physicsClientId=env.cid)
        if env.p.getContactPoints(bodyA=env.robot.robot_uid, bodyB=uid, physicsClientId=env.cid):
            return {"return_blocked": True}
        env.p.resetBasePositionAndOrientation(uid, pos0, orn0, physicsClientId=env.cid)
        env.p.resetBaseVelocity(uid, [0, 0, 0], [0, 0, 0], physicsClientId=env.cid)
        return {"restore_objects": [payload["object"]]}

    return {"moved_objects": [payload["object"]], "tick": tick}


def joint_relocation(env, payload):
    """target/receptacle relocation for the slider or drawer: payload {joint, value} (absolute joint state).

    Blocks resting in/on the moving part are carried along by the same displacement. Without this, the
    drawer teleports out from under them and they fall: Lite v0 preflight drift failures (D-032).
    """
    door = _door(env, payload["joint"])
    link = door.joint_index
    before = np.array(env.p.getLinkState(door.uid, link, physicsClientId=env.cid)[0])
    lo, hi = env.p.getAABB(door.uid, link, physicsClientId=env.cid)
    lo, hi = np.array(lo) - 0.01, np.array(hi) + 0.03
    riders = []
    for o in env.scene.movable_objects:
        pos = np.array(env.p.getBasePositionAndOrientation(o.uid, physicsClientId=env.cid)[0])
        if np.all(pos >= lo) and np.all(pos <= hi):
            riders.append(o)
    env.p.resetJointState(door.uid, link, payload["value"], 0.0, physicsClientId=env.cid)
    delta = np.array(env.p.getLinkState(door.uid, link, physicsClientId=env.cid)[0]) - before
    moved = []
    for o in riders:
        pos, orn = env.p.getBasePositionAndOrientation(o.uid, physicsClientId=env.cid)
        env.p.resetBasePositionAndOrientation(o.uid, (np.array(pos) + delta).tolist(), orn, physicsClientId=env.cid)
        env.p.resetBaseVelocity(o.uid, [0, 0, 0], [0, 0, 0], physicsClientId=env.cid)
        moved.append(o.name)
    return {"moved_doors": [payload["joint"]], "moved_objects": moved}


# --- observation ------------------------------------------------------------------------------------


def shifted_static_view(look_from, look_at, up, payload):
    """A camera bump: translate the camera, then rotate it in place (yaw about world z, pitch about its
    right axis). Returns (eye, target). The static camera is ~4 m away with a 10 deg FOV, so a 1 deg
    rotation moves the image by ~20 px and 1 cm of translation by ~3 px (scripts/calibrate_camera.py)."""
    eye, tgt, up = (np.asarray(v, float) for v in (look_from, look_at, up))
    t = np.asarray(payload.get("translate_m", [0, 0, 0]), float)
    eye, tgt = eye + t, tgt + t
    d = tgt - eye
    yaw = np.deg2rad(payload.get("yaw_deg", 0.0))
    rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    d = rz @ d
    right = np.cross(d, up)
    right /= np.linalg.norm(right)
    pitch = np.deg2rad(payload.get("pitch_deg", 0.0))
    k = right
    d = d * np.cos(pitch) + np.cross(k, d) * np.sin(pitch) + k * np.dot(k, d) * (1 - np.cos(pitch))
    return eye, eye + d


def camera_shift(env, payload):
    """payload {translate_m: [dx,dy,dz], yaw_deg, pitch_deg, fov_delta_deg}; static camera only."""
    cam = next(c for c in env.cameras if c.name == "static")
    eye, tgt = shifted_static_view(cam.look_from, cam.look_at, cam.up_vector, payload)
    cam.fov = cam.fov + payload.get("fov_delta_deg", 0.0)
    cam.viewMatrix = env.p.computeViewMatrix(eye.tolist(), tgt.tolist(), cam.up_vector)
    cam.projectionMatrix = env.p.computeProjectionMatrixFOV(cam.fov, cam.aspect, cam.nearval, cam.farval)
    return {}


def sensor_noise_onset(env, payload):
    """Gaussian pixel noise plus an occluded square, from a frozen seed. payload {cameras, sigma, occlusion_frac, seed}.

    Applied to every rendered observation after onset. The rest of the noise is drawn from a generator
    seeded per render index, so a rollout is reproducible.
    """
    rng_seed, sigma, frac = payload["seed"], payload["sigma"], payload["occlusion_frac"]
    cams = set(payload["cameras"])
    orig = env.get_obs
    counter = {"i": 0}

    def corrupt(img, key):
        g = np.random.default_rng([rng_seed, counter["i"], zlib.crc32(key.encode())])
        x = img.astype(np.float32) / 255.0 + g.normal(0, sigma, img.shape).astype(np.float32)
        h, w = img.shape[:2]
        side = int(round(np.sqrt(frac) * min(h, w)))
        if side > 0:
            y0 = int(np.random.default_rng([rng_seed, 7, zlib.crc32(key.encode())]).integers(0, h - side + 1))
            x0 = int(np.random.default_rng([rng_seed, 8, zlib.crc32(key.encode())]).integers(0, w - side + 1))
            x[y0 : y0 + side, x0 : x0 + side] = 0.0  # fixed occluder position for the whole rollout
        return (np.clip(x, 0, 1) * 255).astype(np.uint8)

    def get_obs():
        obs = orig()
        for k in list(obs["rgb_obs"]):
            if k.replace("rgb_", "") in cams:
                obs["rgb_obs"][k] = corrupt(obs["rgb_obs"][k], k)
        counter["i"] += 1
        return obs

    env.get_obs = get_obs
    return {}


def illumination_switch(env, payload):
    """Re-light every camera render. payload {light_direction, light_color, ambient, diffuse, specular}.

    getCameraImage's light parameters do change the EGL render (event_gallery light test: 30 levels mean).
    """
    kw = {
        "lightDirection": payload["light_direction"],
        "lightColor": payload["light_color"],
        "lightAmbientCoeff": payload["ambient"],
        "lightDiffuseCoeff": payload["diffuse"],
        "lightSpecularCoeff": payload.get("specular", 0.05),
    }
    for cam in env.cameras:
        orig_get = env.p.getCameraImage

        def render(cam=cam, orig_get=orig_get):
            if cam.name == "gripper":
                ls = env.p.getLinkState(cam.robot_uid, cam.gripper_cam_link, physicsClientId=env.cid)
                rot = np.array(env.p.getMatrixFromQuaternion(ls[1])).reshape(3, 3)
                cam.view_matrix = env.p.computeViewMatrix(ls[0], np.array(ls[0]) + rot[:, 1], -rot[:, 2])
                cam.projection_matrix = env.p.computeProjectionMatrixFOV(cam.fov, cam.aspect, cam.nearval, cam.farval)
                view, proj = cam.view_matrix, cam.projection_matrix
            else:
                view, proj = cam.viewMatrix, cam.projectionMatrix
            img = orig_get(width=cam.width, height=cam.height, viewMatrix=view, projectionMatrix=proj,
                           physicsClientId=env.cid, **kw)
            return cam.process_rgbd(img, cam.nearval, cam.farval)

        cam.render = render
    return {}


def visual_theme_switch(env, payload):
    """Recolour the table/furniture links (never the blocks). payload {rgba} or {rgba_per_link: {link: rgba}}."""
    table = env.scene.fixed_objects[0]
    n = env.p.getNumJoints(table.uid, physicsClientId=env.cid)
    for link in range(-1, n):
        rgba = payload.get("rgba_per_link", {}).get(str(link), payload.get("rgba"))
        if rgba is not None:
            env.p.changeVisualShape(table.uid, link, rgbaColor=rgba, physicsClientId=env.cid)
    return {"lights_preserved": True}


# --- embodiment -------------------------------------------------------------------------------------


def grasp_loss(env, payload):
    """The held block slips out and lands on the surface below. payload {object, pos} (resolved: below TCP)."""
    _set_block_pose(env, payload["object"], payload["pos"])
    return {"moved_objects": [payload["object"]]}


def arm_push(env, payload):
    """An external shove that physically displaces the arm: payload {displacement_m: [dx, dy, dz]} (world frame).

    The arm's joints are reset (zero velocity) to an IK solution for TCP + d at the current orientation.
    The relative-action controller's hold point (robot.target_pos) moves with it, and a block held in the
    gripper is carried by the same displacement. Relative-action policies (FLOWER, MoDE) and absolute-action
    policies (X-VLA) therefore all face the same displaced arm (D-039; the v1 hold-point-only shift is
    invisible to absolute control).
    """
    r, p, cid = env.robot, env.p, env.cid
    tcp, orn = p.getLinkState(r.robot_uid, r.tcp_link_id, physicsClientId=cid)[:2]
    d = np.asarray(payload["displacement_m"], float)
    held = [o for o in env.scene.movable_objects
            if p.getContactPoints(bodyA=r.robot_uid, bodyB=o.uid, physicsClientId=cid)]
    q = r.mixed_ik.get_ik(np.asarray(tcp) + d, orn)
    for i, j in enumerate(r.arm_joint_ids):
        p.resetJointState(r.robot_uid, j, float(q[i]), 0.0, physicsClientId=cid)
    r.control_motors(q)
    new_tcp = np.asarray(p.getLinkState(r.robot_uid, r.tcp_link_id, computeForwardKinematics=True, physicsClientId=cid)[0])
    moved = new_tcp - np.asarray(tcp)
    r.target_pos = np.asarray(r.target_pos, float) + moved
    for o in held:
        pos, o_orn = p.getBasePositionAndOrientation(o.uid, physicsClientId=cid)
        p.resetBasePositionAndOrientation(o.uid, (np.asarray(pos) + moved).tolist(), o_orn, physicsClientId=cid)
        p.resetBaseVelocity(o.uid, [0, 0, 0], [0, 0, 0], physicsClientId=cid)
    return {"shifted_objects": {o.name: moved.tolist() for o in held}, "achieved_displacement_m": moved.tolist()}


# --- clutter and path -------------------------------------------------------------------------------


def _spawn_box(env, pos, half_extents, rgba, mass=0.0):
    col = env.p.createCollisionShape(env.p.GEOM_BOX, halfExtents=half_extents, physicsClientId=env.cid)
    vis = env.p.createVisualShape(env.p.GEOM_BOX, halfExtents=half_extents, rgbaColor=rgba, physicsClientId=env.cid)
    return env.p.createMultiBody(mass, col, vis, basePosition=pos, physicsClientId=env.cid)


def distractor_burst(env, payload):
    """Insert neutral (grey) boxes on the table. payload {objects: [{pos, half_extents}]}. Never red/blue/pink."""
    for o in payload["objects"]:
        _spawn_box(env, o["pos"], o["half_extents"], [0.5, 0.5, 0.5, 1.0], mass=0.05)
    return {}


def obstacle_insertion(env, payload):
    """A static box in the approach corridor. payload {pos, half_extents}."""
    _spawn_box(env, payload["pos"], payload["half_extents"], [0.35, 0.3, 0.25, 1.0], mass=0.0)
    return {}


# --- intent -----------------------------------------------------------------------------------------


def instruction_revision(env, payload):
    """payload {goal}: a different feasible task in the same scene; success is judged on it."""
    return {"goal": payload["goal"]}


def paraphrase_swap(env, payload):
    """payload {lang}: same task, different wording (a language null event)."""
    return {"lang": payload["lang"]}


def null_event(env, payload):
    return {}


REGISTRY = {
    "target_relocation": None,  # dispatched on payload kind (block vs joint)
    "receptacle_relocation": None,
    "camera_shift": camera_shift,
    "illumination_switch": illumination_switch,
    "visual_theme_switch": visual_theme_switch,
    "grasp_loss": grasp_loss,
    "arm_push": arm_push,
    "sensor_noise_onset": sensor_noise_onset,
    "distractor_burst": distractor_burst,
    "obstacle_insertion": obstacle_insertion,
    "instruction_revision": instruction_revision,
    "paraphrase_swap": paraphrase_swap,
    "null_event": null_event,
    "reversible_relocation": reversible_relocation,
}


def make_event(change_type: str, payload: dict):
    """Return `event(env) -> effect` for the runner."""
    if change_type in ("target_relocation", "receptacle_relocation"):
        fn = joint_relocation if "joint" in payload else block_relocation
    else:
        fn = REGISTRY[change_type]
        if fn is None:
            raise NotImplementedError(change_type)
    return lambda env: fn(env, payload)
