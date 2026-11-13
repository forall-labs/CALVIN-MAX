"""Sensor-noise onset-phase sweep on the camera_shift phase slots (D-066).

    python scripts/build_phase_noise_manifest.py

Every phase-track camera_shift case (6 onsets x 100 slots) gets a sensor_noise_onset twin with the same chain,
position, onset and seed, and variant-A noise. Camera shift degrades only X-VLA (D-062); noise degrades all three
policies (D-065), so this tests on each whether the loss shrinks as the onset moves later into the subtask, the
commitment prediction. Matched slots also give a within-slot camera vs noise comparison. Separate file; max_v1 frozen.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

NOISE_A = {"cameras": ["static"], "sigma": 0.05, "occlusion_frac": 0.1}


def main():
    man = json.loads(Path("benchmark/manifests/max_v1.json").read_text())["cases"]
    cases = []
    for c in man:
        if c["track"] != "phase" or c["change_type"] != "camera_shift":
            continue
        cid = c["case_id"].replace("phase-camera_shift-", "phase-sensor_noise_onset-", 1)
        cases.append({**c, "case_id": cid, "change_type": "sensor_noise_onset", "variant": "A", "params": dict(NOISE_A),
                      "matched_case": c["case_id"]})
    meta = {"version": "phase_noise_v1", "source": "max_v1 phase camera_shift slots", "params": NOISE_A,
            "counts": Counter(c["onset"] for c in cases)}
    Path("benchmark/manifests/phase_noise_v1.json").write_text(json.dumps({"meta": meta, "cases": cases}, indent=0))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
