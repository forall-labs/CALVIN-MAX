"""Onset resolution from a recorded Base trace (RESEARCH_PLAN §3.2).

Every onset is a pure function of the Base trace's privileged state. None of them looks at
outcomes, so manifests stay outcome-blind. A return of None means Base never reached that
onset, which the evaluator records as `trigger_unreached` (the case stays in the denominator).

Trace arrays cover one subtask window, indexed from that subtask's first step:
  tcp_pos       (T, 3)  TCP position
  entity_pos    (T, 3)  designated entity position (block centre or handle/button link)
  contact       (T,)    bool, gripper-entity contact
  receptacle_pos(T, 3)  optional, for `place`
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OnsetParams:
    # Starting values; frozen per D-011 after dev-chain calibration. r is not LIBERO-MAX's 18 cm:
    # CALVIN's workspace is smaller and calibration targets onset at 30-60% of Base subtask time.
    radius: float = 0.10
    min_steps: int = 10
    approach_fraction: float = 0.25
    lift_height: float = 0.03
    place_radius: float = 0.06


def _dist(a, b):
    return np.linalg.norm(np.asarray(a) - np.asarray(b), axis=-1)


def pre_grasp_proximity(tcp_pos, entity_pos, p: OnsetParams = OnsetParams()):
    """First step >= min_steps where the TCP enters radius r, given that it started outside r.

    Returns (step | None, starts_inside). Cases that start inside r are reassigned to the
    `subtask_boundary` onset at manifest build time, using geometry only.
    """
    d = _dist(tcp_pos, entity_pos)
    starts_inside = bool(d[0] < p.radius)
    if starts_inside:
        return None, True
    hits = np.flatnonzero(d[p.min_steps :] < p.radius)
    return (int(hits[0]) + p.min_steps if len(hits) else None), False


def approach_early(tcp_pos, entity_pos, p: OnsetParams = OnsetParams()):
    """First step where the TCP has closed `approach_fraction` of its initial distance."""
    d = _dist(tcp_pos, entity_pos)
    if d[0] <= 1e-9:
        return None
    progress = (d[0] - d) / d[0]
    hits = np.flatnonzero(progress[p.min_steps :] >= p.approach_fraction)
    return int(hits[0]) + p.min_steps if len(hits) else None


def contact(contact_flags, p: OnsetParams = OnsetParams()):
    hits = np.flatnonzero(np.asarray(contact_flags, dtype=bool))
    return int(hits[0]) if len(hits) else None


def transport(entity_pos, contact_flags, p: OnsetParams = OnsetParams()):
    """First step where the entity is lifted above its start height while held."""
    z = np.asarray(entity_pos)[:, 2]
    held = np.asarray(contact_flags, dtype=bool)
    hits = np.flatnonzero(held & (z - z[0] > p.lift_height))
    return int(hits[0]) if len(hits) else None


def place(entity_pos, receptacle_pos, contact_flags, p: OnsetParams = OnsetParams()):
    """First step after transport where the held entity is within place_radius of its receptacle."""
    t0 = transport(entity_pos, contact_flags, p)
    if t0 is None:
        return None
    d = _dist(entity_pos, receptacle_pos)
    hits = np.flatnonzero(d[t0:] < p.place_radius)
    return int(hits[0]) + t0 if len(hits) else None
