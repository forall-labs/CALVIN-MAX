"""Recover missing legacy preflight Base outcomes without rewriting raw experiment records.

python scripts/backfill_base_metadata.py --base results/base/flower_abc/summary.json --results results/cases/flower_lite_v0/results.jsonl
"""
import argparse
import hashlib
import json
from pathlib import Path


def backfill(base, results):
    raw = Path(base).read_bytes()
    summary = json.loads(raw)
    outcomes = {r["chain_id"]: r["success"] for r in summary["results"] if "success" in r}
    payload = {"schema_version": 1, "source_summary_sha256": hashlib.sha256(raw).hexdigest(), "outcomes": outcomes}
    for result in results:
        target = Path(result).parent / "base_outcomes.json"
        target.write_text(json.dumps(payload, indent=1) + "\n")
        print(target)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--results", type=Path, nargs="+", required=True)
    args = ap.parse_args()
    backfill(args.base, args.results)
