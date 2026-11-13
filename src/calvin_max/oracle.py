"""CALVIN task-oracle rules for mid-subtask events (RESEARCH_PLAN §3.4).

CALVIN judges subtask success by comparing the current scene info with `start_info`, captured
at the subtask's start. A teleported block still touching the table satisfies `push_object`'s
contact check, so a 10 cm move registers a false push success. After a mid-subtask event,
the moved entities in `start_info` are therefore overwritten with their post-event state.

An object carried by the gripper during an arm push is different: the policy has already made progress on
it (lifted it off its support). Replacing its reference erases that progress and the original support
contact that `lift_object` requires, so an unperturbed successful lift fails the oracle (D-048). Its
reference is instead translated by the event's displacement, which removes exactly the event's
contribution and keeps the policy's.
"""

from __future__ import annotations

import copy

import numpy as np


def rebase_start_info(start_info: dict, post_event_info: dict, moved_objects=(), moved_doors=(), lights=(),
                      shifted_objects=None):
    """Return a copy of start_info whose moved entities carry their post-event state.

    moved_objects:   movable object names (block_red, ...); copies pose and contacts.
    moved_doors:     door joint names (base__slide, base__drawer); copies joint state.
    lights:          light names (led, lightbulb); copies logical state.
    shifted_objects: {name: [dx, dy, dz]}; translates the reference position by the event's displacement,
                     keeping its orientation and contacts (carried objects, D-048).
    """
    rebased = copy.deepcopy(start_info)
    src = post_event_info["scene_info"]
    dst = rebased["scene_info"]
    for name in moved_objects:
        dst["movable_objects"][name] = copy.deepcopy(src["movable_objects"][name])
    for name, d in (shifted_objects or {}).items():
        ref = dst["movable_objects"][name]
        ref["current_pos"] = (np.asarray(ref["current_pos"], float) + np.asarray(d, float)).tolist()
    for name in moved_doors:
        dst["doors"][name] = copy.deepcopy(src["doors"][name])
    for name in lights:
        dst["lights"][name] = copy.deepcopy(src["lights"][name])
    return rebased


def side_effects(task_oracle, start_info: dict, end_info: dict, instructed: str) -> set[str]:
    """All tasks the full oracle sees completed, other than the instructed one."""
    return set(task_oracle.get_task_info(start_info, end_info)) - {instructed}
