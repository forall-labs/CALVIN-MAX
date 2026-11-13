#!/usr/bin/env bash
# Merge a policy's two Max-A halves (D-113: PART=a on one runtime, PART=b on the other) into the folders the frozen
# analysis reads, <policy>_maxA and <policy>_maxA_h3. The halves are disjoint by case, so rows are concatenated
# and traces copied; a case found in both halves is an error. Run locally on the mirror, then recompute behaviour.
#   bash scripts/merge_maxA_parts.sh <policy> <base_dir>     e.g. univla univla_abc
set -euo pipefail
cd "$(dirname "$0")/.."
p=$1; B=$2; R=results/cases
for kind in maxA maxA_h3; do
  a=$R/${p}_${kind}_a; b=$R/${p}_${kind}_b; out=$R/${p}_${kind}
  [ -s $a/results.jsonl ] && [ -s $b/results.jsonl ] || { echo "missing $a or $b"; exit 1; }
  [ -e $out ] && { echo "$out exists; not overwriting"; exit 1; }
  python3 - "$a/results.jsonl" "$b/results.jsonl" <<'EOF'
import json, sys
seen = {}
for f in sys.argv[1:]:
    for l in open(f):
        c = json.loads(l).get("case_id")
        if c in seen and seen[c] != f:
            sys.exit(f"case {c} in both halves")
        seen[c] = f
print(f"{len(seen)} cases, disjoint")
EOF
  mkdir -p $out/traces
  cat $a/results.jsonl $b/results.jsonl > $out/results.jsonl
  cp -n $a/traces/* $b/traces/* $out/traces/
  echo "$out: $(wc -l < $out/results.jsonl) rows, $(ls $out/traces | wc -l) traces"
done
echo "now: PYTHONPATH=src python3 scripts/compute_behaviour.py --cases $R/${p}_maxA --base results/base/$B"
