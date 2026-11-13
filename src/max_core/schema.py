"""Simulator-agnostic case and result records (LIBERO-MAX-compatible, D-009)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

# The eight LIBERO-MAX change types, kept verbatim so results can sit side by side.
COMPAT_CHANGE_TYPES = (
    "target_relocation",
    "receptacle_relocation",
    "camera_shift",
    "sensor_noise_onset",
    "illumination_switch",
    "visual_theme_switch",
    "distractor_burst",
    "obstacle_insertion",
)

NEW_CHANGE_TYPES = (
    "grasp_loss",
    "arm_push",
    "instruction_revision",
    "transient_occlusion",
    "reversible_relocation",
)
EXPLORATORY_CHANGE_TYPES = ("goal_preemption", "precondition_regression")
NULL_CHANGE_TYPES = ("null_event", "paraphrase_swap")

ONSETS = (
    "approach_early",
    "pre_grasp_proximity",
    "contact",
    "transport",
    "place",
    "subtask_boundary",
)


class Arm(str, Enum):
    BASE = "control"  # LIBERO-MAX name
    MID = "intervention"  # LIBERO-MAX name
    PRE = "pre_change"  # new: same post-event world, applied at subtask start


class Delivery(str, Enum):
    TELEPORT = "teleport"
    PUSHED = "pushed"  # physically mediated, visible motion
    INSTANT = "instant"  # non-geometric changes (camera, light, language)
    TRANSIENT = "transient"


class Status(str, Enum):
    COMPLETED = "completed"
    TRIGGER_UNREACHED = "trigger_unreached"  # valid outcome, stays in denominator
    NO_POST_EVENT_QUERY = "no_post_event_query"  # valid outcome, flagged
    INFRA_ERROR = "infra_error"  # must be rerun, never dropped


@dataclass(frozen=True)
class Case:
    case_id: str
    chain_id: int
    subtasks: tuple[str, ...]  # the five chain subtasks
    position: int  # event subtask k, 1-based
    change_type: str
    variant: str
    onset: str
    delivery: Delivery
    policy_seed: int
    designated_entity: str
    change_payload: dict[str, Any] = field(default_factory=dict)  # fully resolved, never resampled
    track: str = "core"

    def __post_init__(self):
        known = COMPAT_CHANGE_TYPES + NEW_CHANGE_TYPES + EXPLORATORY_CHANGE_TYPES + NULL_CHANGE_TYPES
        if self.change_type not in known:
            raise ValueError(f"unknown change_type {self.change_type!r}")
        if self.onset not in ONSETS:
            raise ValueError(f"unknown onset {self.onset!r}")
        if not 1 <= self.position <= len(self.subtasks):
            raise ValueError(f"position {self.position} outside chain of {len(self.subtasks)}")
        if self.onset == "subtask_boundary" and self.position == len(self.subtasks):
            raise ValueError("subtask_boundary needs a following subtask")

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["delivery"] = self.delivery.value
        d["subtasks"] = list(self.subtasks)
        return d


@dataclass
class ArmResult:
    case_id: str
    arm: Arm
    status: Status
    event_subtask_success: int | None  # Y in {0,1}; None only for infra errors
    chain_length: int  # completed subtasks 0..5
    onset_step: int | None = None
    stale_actions: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)  # behaviour metrics, side effects
