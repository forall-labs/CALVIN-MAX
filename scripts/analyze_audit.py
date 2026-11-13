"""Human feasibility audit results (D-120): feasibility rate per event with 95% Wilson intervals.

    python scripts/analyze_audit.py results/audit/audit_answers.json     # -> results/analysis/audit.json

Reads the file exported from results/audit/index.html. A case counts as feasible, infeasible or unclear; rates are over
the cases answered feasible or infeasible, with unclear reported separately. Notes are kept for the infeasible ones.
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [100 * (c - h), 100 * (c + h)]


def main():
    ans = json.loads(Path(sys.argv[1]).read_text())["answers"]
    sel = {s["case_id"]: s for s in json.loads((ROOT / "results/audit/audit_selection.json").read_text())}
    by = defaultdict(lambda: {"feasible": 0, "infeasible": 0, "unclear": 0, "unanswered": 0, "notes": []})
    for cid, s in sel.items():
        a = ans.get(cid, {})
        v = a.get("v", "unanswered")
        for key in (s["event"], "all"):
            by[key][v] += 1
            if v in ("infeasible", "unclear") and a.get("note"):
                by[key]["notes"].append({"case_id": cid, "answer": v, "note": a["note"]})
    out = {}
    for ev, c in by.items():
        n = c["feasible"] + c["infeasible"]
        out[ev] = {**c, "n_judged": n, "feasible_rate": 100 * c["feasible"] / n if n else None,
                   "ci95": wilson(c["feasible"], n)}
        e = out[ev]
        rate = "--" if e["feasible_rate"] is None else f"{e['feasible_rate']:.1f} [{e['ci95'][0]:.1f}, {e['ci95'][1]:.1f}]"
        print(f"{ev:22s} feasible {c['feasible']:3d}  infeasible {c['infeasible']:3d}  unclear {c['unclear']:3d}  "
              f"unanswered {c['unanswered']:3d}  rate {rate}")
    (ROOT / "results/analysis/audit.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
