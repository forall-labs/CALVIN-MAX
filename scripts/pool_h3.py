"""Pooled H3 rows for Figures 1 and 6: equal-weight mean over policies of the perceptual-core residual
mid_rewind - Pre, Mid - Pre and Mid - mid_rewind on the same cases, and the budget-matched residual
mid_rewind - Pre_matched (explore_rewind_budget.py, D-116), with 95% and 90% chain-cluster bootstrap CIs that
resample a chain jointly across policies, as compare_policies.bootstrap_pooled does for the pooled timing effects.
Descriptive only (not a pre-registered test).

    python scripts/pool_h3.py [--prefix maxA]

Reads results/analysis/<prefix>_report.json for the H3 policy list and results/cases/<p>_maxA[_h3] for the rows
(report_maxA.h3_rows; explore_rewind_budget.budget_rows reads the traces). Writes
results/analysis/<prefix>_h3_pooled.json.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_maxA import DIR, ROOT, h3_rows  # noqa: E402
from explore_rewind_budget import budget_rows  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="maxA")
    ap.add_argument("--n-boot", type=int, default=5000)
    args = ap.parse_args()
    a = ROOT / "results" / "analysis"
    pols = json.loads((a / f"{args.prefix}_report.json").read_text())["h3_policies"]
    rows = {k: h3_rows(DIR[k]) for k in pols}
    brows = {k: budget_rows(DIR[k]) for k in pols}
    chains = sorted({r["chain"] for rs in rows.values() for r in rs})
    extra = {r["chain"] for rs in brows.values() for r in rs} - set(chains)
    assert not extra, f"budget rows on chains outside the H3 rows: {sorted(extra)[:5]}"
    by = {k: defaultdict(list) for k in pols}
    bby = {k: defaultdict(list) for k in pols}
    for src, dst in ((rows, by), (brows, bby)):
        for k, rs in src.items():
            for r in rs:
                dst[k][r["chain"]].append(r)

    def stats(pick):
        out = defaultdict(list)
        for k in pols:
            s = [r for j in pick for r in by[k].get(chains[j], [])]
            out["residual"].append(100 * np.mean([r["mid_rewind"] - r["pre"] for r in s]))
            out["timing"].append(100 * np.mean([r["mid"] - r["pre"] for r in s]))
            out["mid_minus_rewind"].append(100 * np.mean([r["mid"] - r["mid_rewind"] for r in s]))
            b = [r for j in pick for r in bby[k].get(chains[j], [])]
            out["residual_budget_matched"].append(100 * np.mean([r["mid_rewind"] - r["pre_matched"] for r in b]))
        return {m: float(np.mean(v)) for m, v in out.items()}

    point = stats(np.arange(len(chains)))
    rng = np.random.default_rng(0)
    draws = defaultdict(list)
    for _ in range(args.n_boot):
        for m, v in stats(rng.integers(0, len(chains), len(chains))).items():
            draws[m].append(v)
    res = {m: {"estimate": v, "ci95": [float(x) for x in np.quantile(draws[m], [0.025, 0.975])],
               "ci90": [float(x) for x in np.quantile(draws[m], [0.05, 0.95])]}
           for m, v in point.items()}
    res["policies"] = pols
    (a / f"{args.prefix}_h3_pooled.json").write_text(json.dumps(res, indent=1))
    print({m: (round(v["estimate"], 2), [round(x, 2) for x in v["ci95"]], [round(x, 2) for x in v["ci90"]])
           for m, v in res.items() if m != "policies"})


if __name__ == "__main__":
    main()
