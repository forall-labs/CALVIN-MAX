"""Magnitude ladder on the Lite slots, with Pre and Mid at every level (D-058).

    python scripts/build_magnitude_manifest.py                                            # magnitude_v1 (D-058)
    python scripts/build_magnitude_manifest.py --types sensor_noise_onset --version magnitude_v2  # D-065

Every Lite core case of the chosen events (v1: camera_shift, target_relocation; v2: sensor_noise_onset; A and B) is a slot: chain, position, onset
and realization seed are kept, and the event is re-parameterized on a 5-level ladder scaled from variant A.
Level 3 is variant A itself, so A slots reuse their core case at level 3 (no new case); B slots get one. Writing
a separate file keeps max_v1 frozen. LIBERO-MAX has two frozen magnitudes; this gives a dose-response curve per
arm, so ΔTiming can be read against event size.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

SCALES = {  # level -> scale relative to variant A
    "camera_shift": [0.25, 0.5, 1.0, 2.0, 4.0],  # A: 0.69 deg yaw, 0.345 deg pitch
    "target_relocation": [1 / 3, 2 / 3, 1.0, 1.5, 2.0],  # A: 6 cm block move / 0.3 joint range
    "sensor_noise_onset": [0.25, 0.5, 1.0, 2.0, 4.0],  # A: sigma 0.05, occluder 10% of the static view
}
A_PARAMS = {
    "camera_shift": {"translate_m": [0.0, 0.0, 0.0], "yaw_deg": 0.69, "pitch_deg": 0.345, "fov_delta_deg": 0.0},
    "target_relocation": {"block_distance": 0.06, "joint_frac": 0.3},
    "sensor_noise_onset": {"cameras": ["static"], "sigma": 0.05, "occlusion_frac": 0.1},
}
SCALED = {"camera_shift": ("yaw_deg", "pitch_deg"), "target_relocation": ("block_distance", "joint_frac"),
          "sensor_noise_onset": ("sigma", "occlusion_frac")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--types", default="camera_shift,target_relocation")
    ap.add_argument("--version", default="magnitude_v1")
    args = ap.parse_args()
    types = args.types.split(",")
    man = json.loads(Path("benchmark/manifests/max_v1.json").read_text())["cases"]
    lite = set(json.loads(Path("benchmark/manifests/lite_v1.json").read_text())["case_ids"])
    cases = []
    for c in man:
        ct = c["change_type"]
        if c["case_id"] not in lite or c["track"] != "core" or ct not in types:
            continue
        assert c["params"] == A_PARAMS[ct] or c["variant"] == "B", c["case_id"]
        for level, s in enumerate(SCALES[ct], 1):
            if s == 1.0 and c["variant"] == "A":
                continue  # identical to the core case
            params = {**A_PARAMS[ct], **{k: round(A_PARAMS[ct][k] * s, 4) for k in SCALED[ct]},
                      "level": level, "scale": round(s, 4)}
            cid = f"magnitude-{ct}-L{level}-{c['variant']}-k{c['position']}-{c['chain_id']}"
            cases.append({**c, "case_id": cid, "track": "magnitude", "params": params, "slot_case": c["case_id"]})
    meta = {"version": args.version, "source": "max_v1 + lite_v1 slots", "scales": {t: SCALES[t] for t in types},
            "level3_A_is_core_case": True,
            "counts": Counter(f"{c['change_type']}/L{c['params']['level']}" for c in cases)}
    Path(f"benchmark/manifests/{args.version}.json").write_text(json.dumps({"meta": meta, "cases": cases}, indent=0))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
