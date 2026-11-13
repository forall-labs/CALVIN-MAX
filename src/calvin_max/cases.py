"""From a manifest case plus one policy's Base trace to a concrete branch plan (D-025).

plan_case() runs offline from the Base trace (no simulator):
- status `trigger_unreached` if Base never attempted subtask k, or the onset rule never fires in it;
- the Mid onset step and the Pre onset step (subtask k start);
- the absolute payload, realized from the state at the Mid onset, which both arms apply. Pre and Mid
  therefore end in the same post-event world state.
- status `event_unrealizable` if no valid payload exists (e.g. the target block is held).
Both statuses keep the case in the denominator with Dynamic = Base.
"""

from __future__ import annotations

import numpy as np

from calvin_max import resolve
from calvin_max.tasks import BLOCKS, DOOR_JOINT, aabb_distance, designated_entity

ONSET = {"radius": 0.06, "min_steps": 10, "receptacle_proximity": True}  # D-031 calibration
SCENE_IDX = {"base__slide": 0, "base__drawer": 1}


def _positions(base, t):
    return {b: list(base.entity_pos[b][t]) for b in BLOCKS}


def onset_step(case, base, onset=ONSET):
    """(onset, info) or (None, reason).

    onset = number of chain actions executed before the event. Trace index j holds the state after action j,
    so a condition first true at index j gives onset j + 1. A condition first true on the subtask's final
    (successful) action would put the event at the next subtask's start, with the event subtask already
    scored as Base; such cases are trigger_unreached (D-051).
    """
    t, info = _onset_step(case, base, onset)
    if t is not None:
        k = case["position"] - 1
        e = base.subtask_starts[k + 1] if k + 1 < len(base.subtask_starts) else len(base.actions)
        if t >= e and case["onset"] not in ("subtask_start", "subtask_boundary"):
            return None, "fired_at_subtask_end"
    return t, info


def _onset_step(case, base, onset):
    k = case["position"] - 1
    if len(base.subtask_success) <= k:
        return None, "subtask_not_reached"
    s = base.subtask_starts[k]
    e = base.subtask_starts[k + 1] if k + 1 < len(base.subtask_starts) else len(base.actions)
    if case["onset"] in ("subtask_start", "subtask_boundary"):
        return s, {"rule": "subtask_start"}
    tcp = np.asarray(base.tcp[s:e])
    ent = designated_entity(case["task"], _positions(base, s), base.tcp[s])
    tgt = ent["target"]
    if tgt is None:
        return None, "no_designated_entity"
    # For place/stack the target is already in the gripper, so approach is measured to the receptacle (D-031).
    prox = ent["receptacle"] if ent.get("receptacle") and onset.get("receptacle_proximity", True) else tgt
    box = np.asarray(base.entity_aabb[prox][s:e])
    d = aabb_distance(tcp, box[:, :3], box[:, 3:])
    m, rule = onset["min_steps"], case["onset"]
    info = {"rule": rule, "entity": ent}
    if rule == "pre_grasp_proximity":
        if d[0] < onset["radius"]:  # started inside: fire as soon as the minimum delay allows
            return (s + m if s + m < e else None), {**info, "started_inside": True}
        hit = np.flatnonzero(d[m:] < onset["radius"])
        return (s + m + int(hit[0]) + 1, {**info, "started_inside": False}) if len(hit) else (None, "rule_never_fired")
    if rule == "approach_early":  # TCP has closed approach_fraction of its initial distance
        if d[0] <= 1e-6:
            return None, "rule_never_fired"
        hit = np.flatnonzero((d[0] - d[m:]) / d[0] >= onset.get("approach_fraction", 0.25))
        return (s + m + int(hit[0]) + 1, info) if len(hit) else (None, "rule_never_fired")
    contact = np.asarray(base.contact[tgt][s:e], dtype=bool)
    if rule == "contact":
        hit = np.flatnonzero(contact)
        return (s + int(hit[0]) + 1, info) if len(hit) else (None, "rule_never_fired")
    if rule in ("transport", "place"):
        if tgt not in BLOCKS:
            return None, "rule_not_applicable"
        z = np.asarray(base.entity_pos[tgt][s:e])[:, 2]
        if rule == "transport":
            hit = np.flatnonzero(contact & (z - z[0] > onset.get("lift_height", 0.03)))
            return (s + int(hit[0]) + 1, info) if len(hit) else (None, "rule_never_fired")
        # place: the block is usually already held at subtask start (place/stack follow a lift)
        hit = np.flatnonzero(contact)
        if not len(hit):
            return None, "rule_never_fired"
        t0 = int(hit[0])
        rec = ent["receptacle"]
        if rec is None:
            return None, "rule_not_applicable"
        rbox = np.asarray(base.entity_aabb[rec][s:e])
        dr = aabb_distance(np.asarray(base.entity_pos[tgt][s:e]), rbox[:, :3], rbox[:, 3:])
        hit = np.flatnonzero(dr[t0:] < onset.get("place_radius", 0.06))
        return (s + t0 + int(hit[0]) + 1, info) if len(hit) else (None, "rule_never_fired")
    raise ValueError(f"unknown onset rule {rule}")


def realize(case, base, t):
    """Absolute payload from the Base state after `t` actions (state index t-1), or None if unrealizable."""
    ct, p, seed = case["change_type"], case["params"], case["seed"]
    i = max(t - 1, 0)
    pos = _positions(base, i)
    tcp = base.tcp[i]
    ent = designated_entity(case["task"], _positions(base, base.subtask_starts[case["position"] - 1]),
                            base.tcp[base.subtask_starts[case["position"] - 1]])
    scene = np.asarray(base.scene_obs[i])
    if ct == "reversible_relocation":  # same realization as the matched persistent case (D-050)
        role = p["role"]
        e = ent[role]
        if e not in BLOCKS:
            return None
        r = resolve.resolve_block_relocation(pos, e, p["block_distance"], seed, p.get("yaw_delta", 0.0))
        return None if r is None else {**r, "return_after": p["return_after"]}
    if ct in ("target_relocation", "receptacle_relocation"):
        role = "target" if ct == "target_relocation" else "receptacle"
        e = ent[role]
        if e in BLOCKS:
            return resolve.resolve_block_relocation(pos, e, p["block_distance"], seed, p.get("yaw_delta", 0.0))
        joint = DOOR_JOINT.get(e)
        if joint is None:
            return None
        return resolve.resolve_joint_relocation(joint, float(scene[SCENE_IDX[joint]]), p["joint_frac"], seed,
                                                task=case["task"], role=role)
    if ct == "obstacle_insertion":
        tgt = ent["target"]
        return resolve.resolve_obstacle(tcp, pos[tgt], seed, half_extents=tuple(p["half_extents"])) if tgt in BLOCKS else None
    if ct == "distractor_burst":
        return resolve.resolve_distractors(pos, tcp, seed, n=p["n"], half=p["half"])
    if ct == "sensor_noise_onset":
        return {**p, "seed": seed}
    if ct == "grasp_loss":
        held = ent["target"]
        if held not in BLOCKS or pos[held][2] < 0.46 + 0.02:
            return None
        below = [pos[held][0], pos[held][1], 0.46]
        if resolve.surface_of(below) is None:
            return None
        return {"object": held, "pos": below}
    if ct == "arm_push":
        a = np.random.default_rng(seed).uniform(0, 2 * np.pi)
        d = p["distance_m"]
        return {"displacement_m": [float(d * np.cos(a)), float(d * np.sin(a)), 0.0], "heading_rad": float(a)}
    return dict(p)  # camera, illumination, theme, instruction_revision, null


def plan_case(case, base, onset=ONSET):
    t_mid, info = onset_step(case, base, onset)
    if t_mid is None:
        return {"case_id": case["case_id"], "status": "trigger_unreached", "reason": info}
    payload = realize(case, base, t_mid)
    if payload is None:
        return {"case_id": case["case_id"], "status": "event_unrealizable", "onset": t_mid, "info": info}
    k = case["position"] - 1
    return {"case_id": case["case_id"], "status": "planned", "mid_onset": t_mid,
            "pre_onset": base.subtask_starts[k], "payload": payload, "info": info,
            "base_event_success": int(base.subtask_success[k]), "base_chain_length": base.chain_length}
