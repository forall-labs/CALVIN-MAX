"""Transient-event manifest, matched case-for-case to the Lite persistent relocations (D-050).

    python scripts/build_transient_manifest.py                                   # transient_v1 (return after 10)
    python scripts/build_transient_manifest.py --return-after 2 5 20 --arms mid  # transient_v2: sweep (D-059)

Every Lite core target_relocation (block tasks) and receptacle_relocation (stack_block) case gets a
reversible twin: same chain, position, onset, variant parameters and realization seed, so the block lands
at the identical pose. The twin returns it to its pre-event pose RETURN_AFTER steps later. Persistent minus
reversible, paired per case, is the cost of the change *staying*; reversible minus Base is the cost of
the transient disturbance alone. Written as a separate file so max_v1 stays frozen.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from calvin_max.tasks import task_kind

RETURN_AFTER = 10  # steps: one FLOWER chunk, half an X-VLA chunk. At 20 the robot is usually already touching the block (smoke test)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--return-after", type=int, nargs="+", default=None, help="sweep -> transient_v2.json, ids tagged r<N>")
    ap.add_argument("--arms", nargs="+", default=None, help="override the matched case's arms")
    args = ap.parse_args()
    sweep = args.return_after is not None
    man = json.loads(Path("benchmark/manifests/max_v1.json").read_text())["cases"]
    lite = set(json.loads(Path("benchmark/manifests/lite_v1.json").read_text())["case_ids"])
    cases = []
    for c in man:
        if c["case_id"] not in lite or c["track"] != "core":
            continue
        if c["change_type"] == "target_relocation" and task_kind(c["task"]) == "block":
            role = "target"
        elif c["change_type"] == "receptacle_relocation" and c["task"] == "stack_block":
            role = "receptacle"
        else:
            continue
        for ra in (args.return_after if sweep else [RETURN_AFTER]):
            tag = f"{role}-r{ra}" if sweep else role
            cid = f"transient-reversible_relocation-{tag}-{c['variant']}-k{c['position']}-{c['chain_id']}"
            cases.append({**c, "case_id": cid, "track": "transient", "change_type": "reversible_relocation",
                          "params": {**c["params"], "role": role, "return_after": ra}, "arms": args.arms or c["arms"],
                          "matched_case": c["case_id"]})  # seed, chain, position, onset unchanged
    version = "transient_v2" if sweep else "transient_v1"
    meta = {"version": version, "source": "max_v1 + lite_v1 (matched)",
            "return_after": args.return_after if sweep else RETURN_AFTER,
            "counts": Counter(f"{c['params']['role']}/{c['variant']}/r{c['params']['return_after']}" if sweep
                              else c["params"]["role"] + "/" + c["variant"] for c in cases)}
    Path(f"benchmark/manifests/{version}.json").write_text(json.dumps({"meta": meta, "cases": cases}, indent=0))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
