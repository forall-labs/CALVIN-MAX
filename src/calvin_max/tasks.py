"""Which entity each of CALVIN's 34 tasks is about, and which kind it is (RESEARCH_PLAN §3.3).

The designated entity is what the onset trigger measures distance to (TCP to the entity's AABB) and what
geometry events move. For most tasks it follows from the task name. For stack/unstack/place/push_into it
follows from the state at subtask start (privileged state only, never outcomes).
"""

from __future__ import annotations

import numpy as np

TABLE_LINKS = ("slide_link", "drawer_link", "button_link", "switch_link", "plank_link")
BLOCKS = ("block_red", "block_blue", "block_pink")

ARTICULATED = {
    "move_slider_left": "slide_link",
    "move_slider_right": "slide_link",
    "open_drawer": "drawer_link",
    "close_drawer": "drawer_link",
    "turn_on_lightbulb": "switch_link",
    "turn_off_lightbulb": "switch_link",
    "turn_on_led": "button_link",
    "turn_off_led": "button_link",
}
TOGGLE_TASKS = {"turn_on_led", "turn_off_led"}  # button toggles: blind repetition undoes the goal
RECEPTACLE_TASKS = {"place_in_slider": "plank_link", "place_in_drawer": "drawer_link", "stack_block": None}
DOOR_JOINT = {"slide_link": "base__slide", "drawer_link": "base__drawer"}


def task_kind(task: str) -> str:
    if task in ARTICULATED:
        return "articulated"
    if task in RECEPTACLE_TASKS or task == "push_into_drawer":
        return "receptacle"
    return "block"


def _block_in_name(task):
    for c in ("red", "blue", "pink"):
        if f"_{c}_" in task:
            return f"block_{c}"
    return None


def held_block(positions: dict, tcp, radius=0.06, lift=0.03):
    """A block above the table near the TCP, if any."""
    best = None
    for b in BLOCKS:
        p = np.asarray(positions[b], float)
        d = np.linalg.norm(p - np.asarray(tcp, float))
        if p[2] > 0.46 + lift and d < radius and (best is None or d < best[1]):
            best = (b, d)
    return best[0] if best else None


def designated_entity(task: str, positions: dict, tcp) -> dict:
    """{'target': entity, 'receptacle': entity or None} at subtask start.

    positions: block name -> xyz at subtask start; tcp: xyz at subtask start.
    """
    if task in ARTICULATED:
        return {"target": ARTICULATED[task], "receptacle": None}
    b = _block_in_name(task)
    if b is not None:
        return {"target": b, "receptacle": None}
    held = held_block(positions, tcp)
    tcp = np.asarray(tcp, float)
    if task in ("place_in_slider", "place_in_drawer"):
        return {"target": held, "receptacle": RECEPTACLE_TASKS[task]}
    if task == "stack_block":
        others = [x for x in BLOCKS if x != held]
        base = min(others, key=lambda x: np.linalg.norm(np.asarray(positions[x], float)[:2] - tcp[:2]))
        return {"target": held, "receptacle": base}
    if task == "unstack_block":
        top = max(BLOCKS, key=lambda x: positions[x][2])
        return {"target": top, "receptacle": None}
    if task == "push_into_drawer":
        on_table = [x for x in BLOCKS if abs(positions[x][2] - 0.46) < 0.03]
        tgt = min(on_table or BLOCKS, key=lambda x: np.linalg.norm(np.asarray(positions[x], float) - tcp))
        return {"target": tgt, "receptacle": "drawer_link"}
    raise KeyError(task)


def aabb_distance(points, aabb_min, aabb_max):
    """Euclidean distance from each point to an axis-aligned box (0 inside). Shapes (T,3)."""
    p, lo, hi = (np.asarray(a, float) for a in (points, aabb_min, aabb_max))
    return np.linalg.norm(np.maximum(0.0, np.maximum(lo - p, p - hi)), axis=-1)
