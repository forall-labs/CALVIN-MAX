"""Next-instruction success in all three arms (exploratory; D-131). The registered H6 compares Mid with Base among
cases whose affected instruction succeeded in both. Every core change persists to the end of the chain, so that
difference mixes the changed world with any trace of meeting the change mid-motion. Here the affected instruction
must succeed in Base, Pre and Mid, and success on instruction k+1 is compared across the three arms: Pre - Base is
the cost of the changed world on the next instruction, Mid - Pre what the timing adds.

    PYTHONPATH=src python scripts/analyze_next_instruction.py [--prefix maxA]

Equal-weight mean over policies, chains resampled jointly (as pool_core.py). Also prints the two-arm H6 selection
per policy as a check against <prefix>_report.json. Writes results/analysis/<prefix>_next_instruction.json.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate import load, parse_id  # noqa: E402
from compare_policies import CHANNEL  # noqa: E402
from report_maxA import DIR, ROOT  # noqa: E402


def rows(p):
    """(chain, channel, base_next, pre_next, mid_next, pre_ok) for realized cases with the affected instruction
    k < 5 solved in Base and Mid; pre_ok says whether Pre solved it too."""
    out = []
    for cid, c in load([ROOT / f"results/cases/{p}_maxA/results.jsonl"]).items():
        d = parse_id(cid)
        if d["track"] not in ("core", "new") or d["change_type"] not in CHANNEL or "mid" not in c or c.get("infra"):
            continue
        k, L = d["k"], c.get("len0")
        if L is None or k >= 5:
            continue
        m, pr = c["mid"]["subtask_success"], c.get("pre", {}).get("subtask_success")
        ok = lambda s: len(s) >= k and bool(s[k - 1])  # noqa: E731
        nxt = lambda s: int(len(s) > k and bool(s[k]))  # noqa: E731
        if not (L >= k and ok(m)):
            continue
        out.append((d["chain"], CHANNEL[d["change_type"]], int(L >= k + 1), nxt(pr) if pr else 0, nxt(m),
                    bool(pr) and ok(pr)))
    return out


def pool(data, pols, n_boot, sel):
    chains = sorted({r[0] for k in pols for r in data[k]})
    by = {k: defaultdict(list) for k in pols}
    for k in pols:
        for r in data[k]:
            if sel(r):
                by[k][r[0]].append(r[2:5])

    def stats(pick):
        per = []
        for k in pols:
            s = np.array([r for j in pick for r in by[k].get(chains[j], [])], dtype=float)
            per.append(100 * s.mean(axis=0))
        b, p, m = np.mean(per, axis=0)
        return {"base": b, "pre": p, "mid": m, "pre_minus_base": p - b, "mid_minus_base": m - b, "mid_minus_pre": m - p}

    point = stats(np.arange(len(chains)))
    rng = np.random.default_rng(0)
    dr = defaultdict(list)
    for _ in range(n_boot):
        for a, v in stats(rng.integers(0, len(chains), len(chains))).items():
            dr[a].append(v)
    res = {a: {"estimate": float(v), "ci95": [float(x) for x in np.quantile(dr[a], [0.025, 0.975])]}
           for a, v in point.items()}
    res["n_cases"] = {k: sum(len(v) for v in by[k].values()) for k in pols}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="maxA")
    ap.add_argument("--n-boot", type=int, default=5000)
    args = ap.parse_args()
    a = ROOT / "results" / "analysis"
    rep = json.loads((a / f"{args.prefix}_report.json").read_text())
    pols = rep["policies"]
    data = {k: rows(DIR[k]) for k in pols}
    for k in pols:  # check: the two-arm selection reproduces the registered H6 rows
        r = data[k]
        mb = 100 * (np.mean([x[4] for x in r]) - np.mean([x[2] for x in r]))
        print(f"{k:14s} H6 check n={len(r)} (report {rep['H6'][k]['n']})  Mid-Base {mb:+.2f} "
              f"(report {rep['H6'][k]['difference']:+.2f})")
    out = {"two_arm": pool(data, pols, args.n_boot, lambda r: True)}
    for name, sel in (("all", lambda r: r[5]), ("perceptual", lambda r: r[5] and r[1] == "perceptual"),
                      ("physical", lambda r: r[5] and r[1] == "physical")):
        out[name] = pool(data, pols, args.n_boot, sel)
    out["policies"] = pols
    (a / f"{args.prefix}_next_instruction.json").write_text(json.dumps(out, indent=1))
    for name in ("two_arm", "all", "perceptual", "physical"):
        e = out[name]
        print(name, {m: (round(e[m]["estimate"], 1), [round(x, 1) for x in e[m]["ci95"]]) for m in e if m != "n_cases"},
              "n", sum(e["n_cases"].values()))


if __name__ == "__main__":
    main()
