"""Resolve event payloads from a Base trace at the onset step (outcome-blind, D-009/D-011).

Resolution uses only the privileged state recorded in the Base trace (block positions, TCP), the
scene's support surfaces, and a frozen per-case seed. It never looks at outcomes. A resolver returns
None when no valid payload exists; the manifest builder then replaces the case inside its cell.
"""

from __future__ import annotations

import numpy as np

# Support surfaces of scene D (calvin_env/conf/scene/calvin_scene_D.yaml), x/y boxes at z = 0.46.
SURFACES = {
    "table": np.array([[0.0, -0.15], [0.35, -0.03]]),
    "slider_left": np.array([[-0.32, 0.05], [-0.16, 0.12]]),
    "slider_right": np.array([[-0.05, 0.05], [0.13, 0.12]]),
}
EDGE_MARGIN = 0.02  # keep the block's centre this far inside its surface
MIN_SEPARATION = 0.06  # from other blocks' centres after the move
BLOCK_NAMES = ("block_red", "block_blue", "block_pink")


def surface_of(pos, tol=0.03):
    """The support surface a block rests on, from its x/y and height; None if held or in the drawer."""
    x, y, z = pos
    if abs(z - 0.46) > tol:
        return None
    for name, box in SURFACES.items():
        if box[0, 0] - tol <= x <= box[1, 0] + tol and box[0, 1] - tol <= y <= box[1, 1] + tol:
            return name
    return None


def resolve_block_relocation(positions: dict, obj: str, distance: float, seed: int, yaw_delta: float = 0.0,
                             n_directions: int = 72):
    """Move `obj` by `distance` (m) on its own support surface, in the first valid direction of a
    seeded permutation of n_directions headings. Returns {object, pos, yaw_delta, surface} or None."""
    pos = np.asarray(positions[obj], dtype=float)
    surf = surface_of(pos)
    if surf is None:
        return None
    lo, hi = SURFACES[surf][0] + EDGE_MARGIN, SURFACES[surf][1] - EDGE_MARGIN
    others = [np.asarray(positions[o], float) for o in positions if o != obj]
    angles = np.random.default_rng(seed).permutation(n_directions) * (2 * np.pi / n_directions)
    for a in angles:
        new = pos[:2] + distance * np.array([np.cos(a), np.sin(a)])
        if np.any(new < lo) or np.any(new > hi):
            continue
        if any(np.linalg.norm(new - o[:2]) < MIN_SEPARATION for o in others):
            continue
        return {"object": obj, "pos": [float(new[0]), float(new[1]), float(pos[2])],
                "yaw_delta": float(yaw_delta), "surface": surf, "heading_rad": float(a)}
    return None


def resolve_obstacle(tcp, target, seed: int, half_extents=(0.02, 0.02, 0.04), offset=0.07, table_z=0.46):
    """A static box on the table `offset` m in front of the target, along the TCP->target direction in x/y.

    The v0 midpoint rule often fell off the table when the TCP was far away (D-033).
    """
    tcp, target = np.asarray(tcp, float), np.asarray(target, float)
    v = tcp[:2] - target[:2]
    n = np.linalg.norm(v)
    if n < 1e-6:
        return None
    p = target[:2] + offset * v / n
    if surface_of([p[0], p[1], table_z]) != "table":
        return None
    return {"pos": [float(p[0]), float(p[1]), table_z + half_extents[2]], "half_extents": list(half_extents)}


def resolve_distractors(positions: dict, tcp, seed: int, n: int = 5, half=0.015, min_sep=None):
    """n grey boxes on the table, away from blocks, the TCP and each other."""
    rng = np.random.default_rng(seed)
    min_sep = half + 0.035 if min_sep is None else min_sep  # clear of the largest block (D-032/D-033)
    box = SURFACES["table"]
    taken = [np.asarray(p, float)[:2] for p in positions.values()] + [np.asarray(tcp, float)[:2]]
    out = []
    for _ in range(3000):
        if len(out) == n:
            break
        c = rng.uniform(box[0] + EDGE_MARGIN, box[1] - EDGE_MARGIN)
        if all(np.linalg.norm(c - t) >= min_sep for t in taken):
            taken.append(c)
            out.append({"pos": [float(c[0]), float(c[1]), 0.46 + half + 0.002], "half_extents": [half] * 3})
    return {"objects": out} if len(out) == n else None


# Joint ranges (calvin_env/conf/tasks/new_playtable_tasks.yaml comments) and task thresholds.
JOINT_RANGE = {"base__slide": 0.56, "base__drawer": 0.24}
TASK_JOINT_DELTA = {"move_slider_left": ("base__slide", 0.15), "move_slider_right": ("base__slide", -0.15),
                    "open_drawer": ("base__drawer", 0.12), "close_drawer": ("base__drawer", -0.12)}
MIN_DRAWER_OPEN_FOR_PLACE = 0.10


def resolve_joint_relocation(joint: str, current: float, frac: float, seed: int, task: str | None = None,
                             role: str = "target"):
    """Shift a drawer/slider joint by frac x range, in a seeded direction that keeps the task completable.

    target role: after the rebase the task still needs its full threshold, so leave that much headroom.
    receptacle role (place_in_drawer): the drawer stays open at least MIN_DRAWER_OPEN_FOR_PLACE.
    """
    R = JOINT_RANGE[joint]
    delta = frac * R
    signs = [1, -1] if np.random.default_rng(seed).random() < 0.5 else [-1, 1]
    for s in signs:
        new = current + s * delta
        if not 0.0 <= new <= R:
            continue
        if role == "target" and task in TASK_JOINT_DELTA:
            thr = TASK_JOINT_DELTA[task][1]
            if not 0.0 <= new + thr <= R:
                continue
        if role == "receptacle" and joint == "base__drawer" and new < MIN_DRAWER_OPEN_FOR_PLACE:
            continue
        return {"joint": joint, "value": float(new), "from": float(current)}
    return None
