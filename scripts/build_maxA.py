"""Max-A subset of max_v1 (D-099): the confirmatory run, a third of Max.

    python scripts/build_maxA.py

For each core and new stratum (track, change_type, variant, position), keep the cases on the first third of its
chains in manifest order, which is ascending chain id (20 of 60 for core). Stratifying by position keeps the chain
positions balanced (the manifest lists each cell by position, so a cut per cell would take mostly k=1 and k=2).
eval_1000's chain ids are a random draw, so the cut depends only on the frozen manifest (D-047), not on outcomes, and
the remaining chains extend Max-A to full Max without rerunning anything. Lite cases are left out: they shaped the
hypotheses and are reported separately (D-070), so every Max-A case is held out.

Writes, under benchmark/manifests/:
- maxA_v1.json       Max-A cases outside Lite, core+new, arms mid and pre (the H1, H2 and H6 tests);
- maxA_v1_perc.json  their perceptual core cases, which also get mid_rewind and, for policies that queue actions,
                     mid_oracle (H3 and the queue/state split).
"""

import json
from collections import OrderedDict
from pathlib import Path

M = Path("benchmark/manifests")
PERCEPTUAL = {"camera_shift", "distractor_burst", "illumination_switch", "sensor_noise_onset", "visual_theme_switch"}
FRACTION = 3  # keep 1/FRACTION of each cell's chains


def main():
    cases = json.loads((M / "max_v1.json").read_text())["cases"]
    lite = set(json.loads((M / "lite_v1.json").read_text())["case_ids"])
    cells = OrderedDict()
    for c in cases:
        if c["track"] in ("core", "new"):
            cells.setdefault((c["track"], c["change_type"], c["variant"], c["position"]), []).append(c)
    keep, sizes = [], {}
    for key, cc in cells.items():
        chains = list(OrderedDict.fromkeys(c["chain_id"] for c in cc))
        first = set(chains[: -(-len(chains) // FRACTION)])
        sel = [c for c in cc if c["chain_id"] in first]
        sizes["/".join(map(str, key))] = {"chains": len(first), "cases": len(sel),
                                          "lite": sum(c["case_id"] in lite for c in sel)}
        keep += sel
    held = [c for c in keep if c["case_id"] not in lite]
    perc = [c for c in held if c["track"] == "core" and c["change_type"] in PERCEPTUAL]
    rule = (f"first 1/{FRACTION} of the chains of each core/new (event, variant, position) stratum, manifest order, "
            "minus lite_v1")
    meta = {"source": "max_v1", "rule": rule,
            "decision": "D-099", "cells": sizes, "n_maxA": len(keep), "n_lite_in_maxA": len(keep) - len(held)}
    (M / "maxA_v1.json").write_text(json.dumps({"meta": {**meta, "n": len(held)},
                                                "case_ids": [c["case_id"] for c in held]}, indent=1))
    (M / "maxA_v1_perc.json").write_text(json.dumps({"meta": {**meta, "n": len(perc), "events": sorted(PERCEPTUAL)},
                                                     "case_ids": [c["case_id"] for c in perc]}, indent=1))
    print(f"Max-A {len(keep)} cases ({len(keep) - len(held)} in Lite), held out {len(held)}, "
          f"perceptual core {len(perc)}")
    print(f"distinct chains {len({c['chain_id'] for c in held})}")


if __name__ == "__main__":
    main()
