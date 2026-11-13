"""Privileged-state response metrics for geometry and embodiment events (RESEARCH_PLAN §4).

These measure *how* a policy responds, not only whether it succeeds. LIBERO-MAX leaves this
out: its spec says the benchmark "does not measure internal detection or replanning".
All inputs are the Mid trace from the onset step onward.
"""

from __future__ import annotations

import numpy as np


def _unit(v, eps=1e-9):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)


def retargeting_latency(tcp_pos, old_target, new_target, sustain=5, min_speed=1e-3):
    """Steps from onset until the TCP heads toward the new target rather than the old one.

    At each step, the TCP velocity is projected onto the unit directions to each target.
    Retargeting happens at the first step that starts a run of `sustain` consecutive moving
    steps where the projection onto the new target exceeds the one onto the old target.
    Returns None if this never happens within the trace.
    """
    tcp = np.asarray(tcp_pos, dtype=float)
    if len(tcp) < 2:
        return None
    vel = np.diff(tcp, axis=0)
    speed = np.linalg.norm(vel, axis=1)
    to_new = _unit(np.asarray(new_target) - tcp[:-1])
    to_old = _unit(np.asarray(old_target) - tcp[:-1])
    toward_new = (np.sum(vel * to_new, axis=1) > np.sum(vel * to_old, axis=1)) & (speed > min_speed)
    run = 0
    for t, ok in enumerate(toward_new):
        run = run + 1 if ok else 0
        if run == sustain:
            return t - sustain + 1
    return None


def stale_commitment(tcp_pos, old_target, until=None):
    """Path length (m) covered *toward the old target* after onset, up to `until` steps."""
    tcp = np.asarray(tcp_pos, dtype=float)[: None if until is None else until + 1]
    d_old = np.linalg.norm(tcp - np.asarray(old_target), axis=1)
    return float(np.sum(np.clip(d_old[:-1] - d_old[1:], 0.0, None)))


def ghost_reach(tcp_pos, old_target, new_target, radius=0.03):
    """True if the TCP reaches where the entity used to be.

    Only defined when the old and new poses are clearly apart; returns None otherwise.
    """
    old_target, new_target = np.asarray(old_target), np.asarray(new_target)
    if np.linalg.norm(old_target - new_target) < 2 * radius:
        return None
    d_old = np.linalg.norm(np.asarray(tcp_pos) - old_target, axis=1)
    return bool(np.any(d_old < radius))


RESPONSE_CLASSES = (
    "adapted",
    "late_recovered",
    "stale_then_recovered",
    "stale_failed",
    "frozen",
    "wrong_task",
)


def classify_response(success, latency, ghost, side_effect_tasks, moved_path, late_after=30, frozen_path=0.02):
    """Rule-based response class for one Mid rollout (thresholds frozen with the other params)."""
    if side_effect_tasks and not success:
        return "wrong_task"
    if moved_path < frozen_path:
        return "frozen"
    if success:
        if ghost:
            return "stale_then_recovered"
        if latency is not None and latency > late_after:
            return "late_recovered"
        return "adapted"
    return "stale_failed"
