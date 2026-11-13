"""Instruction revision on Lite (D-128, exploratory): success on the revised task in Pre and Mid, and how often the
original task is completed instead, for every policy that ran the 30 Lite intent cases (pairs realized in both arms).
Mid - Pre per policy with a 95% case bootstrap interval and an exact two-sided McNemar test; the equal-weight mean over
policies with a bootstrap that resamples case ids jointly across policies (5,000 draws, seed 0).

    python scripts/analyze_intent.py --out results/analysis/intent_lite.json
"""
from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results/cases"
RUNS = {"FLOWER": "flower_lite_v0", "MoDE": "mode1_lite_v1", "X-VLA": "xvla_lite_v1", "pi0.5": "pi05_lite_ni",
        "GR-1": "gr1_lite_ni", "Seer": "seer_lite_ni", "UniVLA": "univla_lite_ni", "3DDA": "dda_lite_ni", "RoboFlamingo": "rf_lite_ni",
        "OpenVLA-OFT": "oft_lite_ni"}


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    task = {c["case_id"]: c["task"] for c in json.loads((ROOT / "benchmark/manifests/max_v1.json").read_text())["cases"]}
    pairs, rep = {}, {}
    for name, d in RUNS.items():
        f = R / d / "results.jsonl"
        if not f.exists():
            continue
        rows = {}
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if r["case_id"].startswith("intent") and r["status"] == "completed":
                rows[(r["case_id"], r["arm"])] = r
        ids = sorted({c for c, a in rows if (c, "pre") in rows and (c, "mid") in rows})
        pairs[name] = {c: (rows[(c, "pre")]["event_success"], rows[(c, "mid")]["event_success"]) for c in ids}
        e = {"n": len(ids)}
        for arm in ("pre", "mid"):
            rs = [rows[(c, arm)] for c in ids]
            e[f"{arm}_revised"] = sum(r["event_success"] for r in rs)
            e[f"{arm}_original"] = sum(task[r["case_id"]] in (r.get("side_effects_event_subtask") or []) for r in rs)
        v = pairs[name].values()
        e["pre_sr"], e["mid_sr"] = 100 * e["pre_revised"] / e["n"], 100 * e["mid_revised"] / e["n"]
        e["mid_minus_pre"] = e["mid_sr"] - e["pre_sr"]
        e["pre_only"] = sum(p == 1 and m == 0 for p, m in v)
        e["mid_only"] = sum(p == 0 and m == 1 for p, m in v)
        e["mcnemar_p"] = mcnemar(e["pre_only"], e["mid_only"])
        rep[name] = e
    ids = sorted({c for v in pairs.values() for c in v})
    rng = np.random.default_rng(0)
    per, pooled = {n: [] for n in pairs}, []
    for _ in range(5000):
        s = rng.choice(ids, len(ids), replace=True)
        means = []
        for n, v in pairs.items():
            xs = [v[c][1] - v[c][0] for c in s if c in v]
            if xs:
                per[n].append(100 * np.mean(xs))
                means.append(100 * np.mean(xs))
        pooled.append(np.mean(means))
    for n in rep:
        rep[n]["mid_minus_pre_ci95"] = [float(x) for x in np.percentile(per[n], [2.5, 97.5])]
    out = {"policies": rep,
           "pooled": {"mid_minus_pre": float(np.mean([e["mid_minus_pre"] for e in rep.values()])),
                      "ci95": [float(x) for x in np.percentile(pooled, [2.5, 97.5])],
                      "pairs": sum(e["n"] for e in rep.values()),
                      "original_pre": sum(e["pre_original"] for e in rep.values()),
                      "original_mid": sum(e["mid_original"] for e in rep.values())}}
    print(json.dumps(out, indent=1))
    if args.out:
        args.out.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
