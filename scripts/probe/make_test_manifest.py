"""A small manifest over given chains for pipeline tests: every implemented event at positions 1-2.

    python scripts/probe/make_test_manifest.py --chains E0000,E0001,E0002,E0003 --out /tmp/test_manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_manifest import VARIANTS, applicable, case_seed  # noqa: E402

from calvin_max.symbolic import revision_candidates, state_before  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chains", required=True)
    ap.add_argument("--chains-file", type=Path, default=Path("benchmark/chains/eval_1000.json"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    want = args.chains.split(",")
    chains = [c for c in json.loads(args.chains_file.read_text()) if c["chain_id"] in want]
    cases = []
    for c in chains:
        for k in (1, 2):
            task = c["subtasks"][k - 1]
            for ct, vs in VARIANTS.items():
                var = "B" if ct in ("camera_shift", "illumination_switch") else "A"
                if applicable(ct, task):
                    cid = f"test-{ct}-{var}-k{k}-{c['chain_id']}"
                    cases.append({"case_id": cid, "track": "test", "chain_id": c["chain_id"], "position": k, "task": task,
                                  "change_type": ct, "variant": var, "params": vs[var], "onset": "pre_grasp_proximity",
                                  "arms": ["mid", "pre"], "seed": case_seed(cid)})
            cid = f"test-null_event-k{k}-{c['chain_id']}"
            cases.append({"case_id": cid, "track": "test", "chain_id": c["chain_id"], "position": k, "task": task,
                          "change_type": "null_event", "variant": "-", "params": {}, "onset": "pre_grasp_proximity",
                          "arms": ["mid", "pre"], "seed": case_seed(cid)})
            cands = revision_candidates(state_before(c["initial_condition"], c["subtasks"], k - 1), task)
            if cands:
                cid = f"test-instruction_revision-k{k}-{c['chain_id']}"
                cases.append({"case_id": cid, "track": "test", "chain_id": c["chain_id"], "position": k, "task": task,
                              "change_type": "instruction_revision", "variant": "same_family_first",
                              "params": {"goal": cands[0]}, "onset": "pre_grasp_proximity", "arms": ["mid", "pre"],
                              "seed": case_seed(cid)})
    args.out.write_text(json.dumps({"meta": {"version": "test"}, "cases": cases}))
    print(len(cases), "cases")


if __name__ == "__main__":
    main()
