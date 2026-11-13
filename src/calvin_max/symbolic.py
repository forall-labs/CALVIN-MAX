"""CALVIN's symbolic scene state along a chain (from the vendored official task conditions/effects).

Base reaches subtask k only if subtasks 1..k-1 succeeded, so the symbolic state at k is policy-independent.
"""

from __future__ import annotations

from calvin_max.vendor.calvin_multistep_sequences import tasks, valid_task

COLORS = ("red", "blue", "pink")


def state_before(initial_condition: dict, subtasks, k: int) -> dict:
    """Symbolic state at the start of subtask k (0-based), applying the effects of subtasks 0..k-1."""
    state = dict(initial_condition)
    for t in subtasks[:k]:
        nxt = valid_task(state, tasks[t])
        if not nxt:
            raise ValueError(f"{t} invalid in {state}")
        state = nxt[0]
    return state


def revision_candidates(state: dict, original: str) -> list[str]:
    """Feasible alternatives to `original` in `state`, best first.

    First the same verb on a different block (lift red -> lift blue), then any other valid task that acts
    on a different entity.
    """
    valid = [t for t in tasks if t != original and valid_task(state, tasks[t])]
    same_family = []
    for c in COLORS:
        if f"_{c}_" in original:
            for c2 in COLORS:
                alt = original.replace(f"_{c}_", f"_{c2}_")
                if c2 != c and alt in valid:
                    same_family.append(alt)
    rest = [t for t in valid if t not in same_family and not _same_entity(t, original)]
    return same_family + rest


def _same_entity(a: str, b: str) -> bool:
    for key in ("slider", "drawer", "led", "lightbulb") + tuple(f"_{c}_" for c in COLORS):
        if key in a and key in b:
            return True
    return False
