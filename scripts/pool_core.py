"""Equal-weight averages over the ten Max-A policies of the Table 2 success and paired-difference columns, with
95% chain-cluster bootstrap intervals that resample a chain jointly across policies (as bootstrap_pooled does for
the channel columns). Descriptive only (not a pre-registered test).

    PYTHONPATH=src python scripts/pool_core.py [--prefix maxA]

Per policy, every case of the eight core events counts as in aggregate.summarize (the source of the table's
per-policy rows): infra errors are skipped, and a case whose event was never realized keeps its Base outcome in
all three arms. Writes results/analysis/<prefix>_core_pooled.json, and the same averages for each embodiment event
(arm push, grasp loss) to <prefix>_embodiment_pooled.json.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate import load, parse_id  # noqa: E402
from report_maxA import DIR, ROOT  # noqa: E402

CORE = {"target_relocation", "receptacle_relocation", "obstacle_insertion", "camera_shift", "sensor_noise_onset",
        "illumination_switch", "visual_theme_switch", "distractor_burst"}
ARMS = ("base", "pre", "mid")


EMBODIMENT = ("arm_push", "grasp_loss")


def core_rows(p, track="core", events=CORE):
    """(chain, y_base, y_pre, y_mid) for every counted case of `events`, following aggregate.summarize."""
    rows = []
    for cid, c in load([ROOT / f"results/cases/{p}_maxA/results.jsonl"]).items():
        d = parse_id(cid)
        if d["track"] != track or d["change_type"] not in events or c.get("infra"):
            continue
        y = c["y0"]
        if c.get("status") in ("trigger_unreached", "event_unrealizable"):
            yp = ym = y
        else:
            if "mid" not in c:
                continue
            ym, yp = c["mid"]["event_success"], c["pre"]["event_success"]
        rows.append((d["chain"], y, yp, ym))
    return rows


def pool(rows, pols, n_boot):
    """Equal-weight mean over policies of each arm and paired difference, chains resampled jointly."""
    chains = sorted({r[0] for rs in rows.values() for r in rs})
    by = {k: defaultdict(list) for k in pols}
    for k, rs in rows.items():
        for r in rs:
            by[k][r[0]].append(r[1:])

    def stats(pick):
        per = []
        for k in pols:
            s = np.array([r for j in pick for r in by[k].get(chains[j], [])], dtype=float)
            m = 100 * s.mean(axis=0)
            per.append({"base": m[0], "pre": m[1], "mid": m[2]})
        out = {arm: float(np.mean([r[arm] for r in per])) for arm in ARMS}
        out["total"] = out["mid"] - out["base"]
        out["config"] = out["pre"] - out["base"]
        out["timing"] = out["mid"] - out["pre"]
        return out

    point = stats(np.arange(len(chains)))
    rng = np.random.default_rng(0)
    draws = defaultdict(list)
    for _ in range(n_boot):
        for m, v in stats(rng.integers(0, len(chains), len(chains))).items():
            draws[m].append(v)
    res = {m: {"estimate": v, "ci95": [float(x) for x in np.quantile(draws[m], [0.025, 0.975])]}
           for m, v in point.items()}
    res["policies"] = pols
    res["n_cases"] = {k: len(v) for k, v in rows.items()}
    return res, point


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="maxA")
    ap.add_argument("--n-boot", type=int, default=5000)
    args = ap.parse_args()
    a = ROOT / "results" / "analysis"
    pols = json.loads((a / f"{args.prefix}_report.json").read_text())["policies"]
    res, point = pool({k: core_rows(DIR[k]) for k in pols}, pols, args.n_boot)
    (a / f"{args.prefix}_core_pooled.json").write_text(json.dumps(res, indent=1))
    print({m: (round(v["estimate"], 2), [round(x, 2) for x in v["ci95"]]) for m, v in res.items()
           if m in point})
    emb = {}
    for ev in EMBODIMENT:
        emb[ev], pt = pool({k: core_rows(DIR[k], "new", {ev}) for k in pols}, pols, args.n_boot)
        print(ev, {m: (round(emb[ev][m]["estimate"], 2), [round(x, 2) for x in emb[ev][m]["ci95"]]) for m in pt})
    (a / f"{args.prefix}_embodiment_pooled.json").write_text(json.dumps(emb, indent=1))


if __name__ == "__main__":
    main()
