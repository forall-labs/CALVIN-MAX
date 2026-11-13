"""Build the policy-independent case manifest from the frozen chains (D-009, D-025).

    python scripts/build_manifest.py --chains benchmark/chains/eval_1000.json --out benchmark/manifests --version v0

A case freezes: chain, event position k, change_type, variant parameters, onset rule, arms and a realization
seed. Concrete poses are realized at run time from each policy's Base trace (resolve.py). Cases that cannot
be realized are scored `event_unrealizable` and stay in the denominator, like `trigger_unreached`.
Cells are change_type x variant x k. Chains are drawn per cell with a fixed seed from those whose k-th
task fits the event (task names only, so nothing depends on any policy). Lite takes LITE_PER_CELL per cell
from metadata with a second seed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from calvin_max.symbolic import revision_candidates, state_before
from calvin_max.tasks import ARTICULATED, task_kind

MANIFEST_SEED = 20261028
LITE_SEED = 20261029
PER_CELL = 60  # x 2 variants x 5 positions = 600 per core event
LITE_PER_CELL = 6

VARIANTS = {
    "target_relocation": {"A": {"block_distance": 0.06, "joint_frac": 0.30},
                          "B": {"block_distance": 0.12, "yaw_delta": 0.785, "joint_frac": 0.60}},
    "receptacle_relocation": {"A": {"block_distance": 0.06, "joint_frac": 0.20},
                              "B": {"block_distance": 0.12, "joint_frac": 0.40}},
    # scripts/calibrate_camera.py: mean workspace displacement A = 10 px, B = 20 px (5% / 10% of 200 px)
    "camera_shift": {"A": {"translate_m": [0.0, 0.0, 0.0], "yaw_deg": 0.69, "pitch_deg": 0.345, "fov_delta_deg": 0.0},
                     "B": {"translate_m": [0.0, 0.0, 0.0142], "yaw_deg": 1.42, "pitch_deg": 0.71, "fov_delta_deg": 1.0}},
    "sensor_noise_onset": {"A": {"cameras": ["static"], "sigma": 0.05, "occlusion_frac": 0.10},
                           "B": {"cameras": ["static", "gripper"], "sigma": 0.05, "occlusion_frac": 0.10}},
    "illumination_switch": {"A": {"light_direction": [1, 1, 1], "light_color": [1, 1, 1], "ambient": 0.3, "diffuse": 0.35},
                            "B": {"light_direction": [-1, 1, 0.3], "light_color": [1.0, 0.75, 0.55], "ambient": 0.45, "diffuse": 0.6}},
    "visual_theme_switch": {"A": {"rgba": [0.85, 0.75, 0.6, 1.0]}, "B": {"rgba": [0.35, 0.45, 0.6, 1.0]}},
    "distractor_burst": {"A": {"n": 5, "half": 0.015}, "B": {"n": 5, "half": 0.025}},
    "obstacle_insertion": {"A": {"half_extents": [0.02, 0.02, 0.02]}, "B": {"half_extents": [0.015, 0.015, 0.05]}},
}
CORE = list(VARIANTS)
# At the calibrated 6 cm pre-grasp radius the TCP-target corridor is too short for an obstacle (Lite v0: 59/60
# unrealizable), so obstacles are inserted at the early-approach onset (D-032).
CORE_ONSET = {"obstacle_insertion": "approach_early"}
NEW_VARIANTS = {
    "grasp_loss": {"A": {}},
    "arm_push": {"A": {"distance_m": 0.05}, "B": {"distance_m": 0.10}},
}
ONSETS = ("approach_early", "pre_grasp_proximity", "contact", "transport", "place", "subtask_boundary")


def onset_applicable(onset: str, task: str) -> bool:
    if onset == "transport":
        return task.startswith("lift_")
    if onset == "place":
        return task in ("place_in_slider", "place_in_drawer", "stack_block")
    return True


def applicable(change_type: str, task: str) -> bool:
    kind = task_kind(task)
    if change_type == "target_relocation":
        return (kind == "block") or task in ("move_slider_left", "move_slider_right", "open_drawer", "close_drawer")
    if change_type == "receptacle_relocation":
        return task in ("stack_block", "place_in_drawer")
    if change_type == "obstacle_insertion":
        return kind == "block" and "_slider" not in task and "_drawer" not in task
    if change_type == "grasp_loss":
        return task.startswith("lift_")
    return True


def case_seed(case_id: str) -> int:
    return int.from_bytes(hashlib.sha256(case_id.encode()).digest()[:4], "little")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chains", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("benchmark/manifests"))
    ap.add_argument("--version", default="v0")
    args = ap.parse_args()
    chains = json.loads(args.chains.read_text())
    rng = np.random.default_rng(MANIFEST_SEED)
    lite_rng = np.random.default_rng(LITE_SEED)
    cases, lite, shortfall = [], [], {}

    def add(cell_cases):
        cases.extend(cell_cases)
        if cell_cases:
            pick = lite_rng.choice(len(cell_cases), size=min(LITE_PER_CELL, len(cell_cases)), replace=False)
            lite.extend(cell_cases[i]["case_id"] for i in sorted(pick))

    for ct in CORE:
        for var, params in VARIANTS[ct].items():
            for k in range(1, 6):
                pool = [c for c in chains if applicable(ct, c["subtasks"][k - 1])]
                n = min(PER_CELL, len(pool))
                if n < PER_CELL:
                    shortfall[f"{ct}/{var}/k{k}"] = n
                chosen = sorted(rng.choice(len(pool), size=n, replace=False))
                cell = []
                for i in chosen:
                    c = pool[i]
                    cid = f"core-{ct}-{var}-k{k}-{c['chain_id']}"
                    cell.append({"case_id": cid, "track": "core", "chain_id": c["chain_id"], "position": k,
                                 "task": c["subtasks"][k - 1], "change_type": ct, "variant": var,
                                 "params": params, "onset": CORE_ONSET.get(ct, "pre_grasp_proximity"),
                                 "arms": ["mid", "pre"], "seed": case_seed(cid)})
                add(cell)

    # Phase track (RESEARCH_PLAN 3.2/3.5): same event at different onsets, variant A, 100 per (event, onset).
    PHASE_EVENTS = {"target_relocation": VARIANTS["target_relocation"]["A"],
                    "receptacle_relocation": VARIANTS["receptacle_relocation"]["A"],
                    "camera_shift": VARIANTS["camera_shift"]["A"], "arm_push": NEW_VARIANTS["arm_push"]["A"]}
    for ct, params in PHASE_EVENTS.items():
        for onset in ONSETS:
            if ct == "target_relocation" and onset in ("transport", "place"):
                continue  # the target is in the gripper then; moving it is grasp_loss, not relocation
            pool = [(c, k) for c in chains for k in range(1, 6)
                    if applicable(ct, c["subtasks"][k - 1]) and onset_applicable(onset, c["subtasks"][k - 1])]
            if not pool:
                continue
            chosen = sorted(rng.choice(len(pool), size=min(100, len(pool)), replace=False))
            cell = []
            for i in chosen:
                c, k = pool[i]
                cid = f"phase-{ct}-{onset}-k{k}-{c['chain_id']}"
                cell.append({"case_id": cid, "track": "phase", "chain_id": c["chain_id"], "position": k,
                             "task": c["subtasks"][k - 1], "change_type": ct, "variant": "A", "params": params,
                             "onset": onset, "arms": ["mid"], "seed": case_seed(cid)})
            add(cell)

    # New-channel track: embodiment events.
    for ct, vs in NEW_VARIANTS.items():
        onset = "transport" if ct == "grasp_loss" else "pre_grasp_proximity"
        for var, params in vs.items():
            pool = [(c, k) for c in chains for k in range(1, 6)
                    if applicable(ct, c["subtasks"][k - 1]) and onset_applicable(onset, c["subtasks"][k - 1])]
            chosen = sorted(rng.choice(len(pool), size=min(150, len(pool)), replace=False))
            cell = []
            for i in chosen:
                c, k = pool[i]
                cid = f"new-{ct}-{var}-k{k}-{c['chain_id']}"
                cell.append({"case_id": cid, "track": "new", "chain_id": c["chain_id"], "position": k,
                             "task": c["subtasks"][k - 1], "change_type": ct, "variant": var, "params": params,
                             "onset": onset, "arms": ["mid", "pre"], "seed": case_seed(cid)})
            add(cell)

    # Null track: the exact-replay control (D-021); 60 per position.
    for k in range(1, 6):
        chosen = sorted(rng.choice(len(chains), size=60, replace=False))
        cell = []
        for i in chosen:
            c = chains[i]
            cid = f"null-null_event-k{k}-{c['chain_id']}"
            cell.append({"case_id": cid, "track": "null", "chain_id": c["chain_id"], "position": k,
                         "task": c["subtasks"][k - 1], "change_type": "null_event", "variant": "-",
                         "params": {}, "onset": "pre_grasp_proximity", "arms": ["mid", "pre"], "seed": case_seed(cid)})
        add(cell)

    # Intent track: instruction_revision with a frozen, symbolically feasible alternative task.
    for k in range(1, 6):
        chosen = sorted(rng.choice(len(chains), size=60, replace=False))
        cell = []
        for i in chosen:
            c = chains[i]
            cands = revision_candidates(state_before(c["initial_condition"], c["subtasks"], k - 1), c["subtasks"][k - 1])
            if not cands:
                continue
            cid = f"intent-instruction_revision-k{k}-{c['chain_id']}"
            cell.append({"case_id": cid, "track": "intent", "chain_id": c["chain_id"], "position": k,
                         "task": c["subtasks"][k - 1], "change_type": "instruction_revision", "variant": "same_family_first",
                         "params": {"goal": cands[0]}, "onset": "pre_grasp_proximity", "arms": ["mid", "pre"],
                         "chain_scoring": "event_subtask_only", "seed": case_seed(cid)})
        add(cell)

    args.out.mkdir(parents=True, exist_ok=True)
    meta = {"version": args.version, "manifest_seed": MANIFEST_SEED, "lite_seed": LITE_SEED,
            "chains_file": args.chains.name, "per_cell": PER_CELL, "lite_per_cell": LITE_PER_CELL,
            "onset_params": "pre_grasp radius 0.06 m, min_steps 10, receptacle proximity (D-031)",
            "counts": Counter(c["change_type"] for c in cases), "lite_count": len(lite), "shortfall": shortfall}
    (args.out / f"max_{args.version}.json").write_text(json.dumps({"meta": meta, "cases": cases}, indent=0))
    (args.out / f"lite_{args.version}.json").write_text(json.dumps({"meta": meta, "case_ids": lite}, indent=0))
    print(json.dumps({k: v for k, v in meta.items()}, indent=1, default=dict))


if __name__ == "__main__":
    main()
