"""Position profile on Max-A pooled over the ten policies, with 95% chain-cluster bootstrap intervals that resample a
chain jointly across policies (as pool_core.py does for Table 2's pooled row). Exploratory, descriptive only.

    /opt/homebrew/bin/python3 scripts/pool_position.py [--n-boot 5000]

Per policy and per (channel, position k) cell, cases are counted exactly as in analyze_position.profile /
aggregate.summarize: core and new tracks, infra errors skipped, an unrealized event keeps its Base outcome (and Base
chain length) in every arm. The pooled value of a term is the equal-weight mean over policies of its per-policy value,
so the point estimates equal the means printed in the Max-A position table. Writes
results/analysis/maxA_position_pooled.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aggregate import load, parse_id  # noqa: E402
from compare_policies import CHANNEL  # noqa: E402
from report_maxA import NAME, ROOT  # noqa: E402

POL = ["flower", "mode", "xvla", "pi05", "gr1", "seer", "univla", "dda", "rf", "oft"]
CELLS = [(ch, k) for ch in ("all", "perceptual", "physical") for k in range(1, 6)]
# per-chain sums: n, base, pre, mid, n_len, len_base, len_mid
NCOL = 7


def rows(p):
    """(cell, chain, sums) for every counted case of policy p."""
    out = []
    for cid, c in load([ROOT / f"results/cases/{p}_maxA/results.jsonl"]).items():
        d = parse_id(cid)
        if d["track"] not in ("core", "new") or d["change_type"] not in CHANNEL or c.get("infra"):
            continue
        y = c["y0"]
        if c.get("status") in ("trigger_unreached", "event_unrealizable"):
            yp = ym = y
            lm = c.get("len0")
        else:
            if "mid" not in c:
                continue
            ym, yp, lm = c["mid"]["event_success"], c["pre"]["event_success"], c["mid"]["chain_length"]
        has_len = c.get("len0") is not None and lm is not None
        v = [1, y, yp, ym, int(has_len), c["len0"] if has_len else 0, lm if has_len else 0]
        for ch in (CHANNEL[d["change_type"]], "all"):
            out.append(((ch, d["k"]), d["chain"], v))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=5000)
    args = ap.parse_args()
    data = {p: rows(p) for p in POL}
    chains = sorted({r[1] for rs in data.values() for r in rs})
    cix = {c: i for i, c in enumerate(chains)}
    # S[p, cell, chain, col]
    S = np.zeros((len(POL), len(CELLS), len(chains), NCOL))
    eix = {e: i for i, e in enumerate(CELLS)}
    for i, p in enumerate(POL):
        for cell, ch, v in data[p]:
            S[i, eix[cell], cix[ch]] += v

    def stats(w):
        """w: weight (multiplicity) per chain -> {term: array over cells} pooled over policies."""
        t = np.einsum("pecx,c->pex", S, w)
        n, nl = t[..., 0], t[..., 4]
        base, pre, mid = t[..., 1] / n, t[..., 2] / n, t[..., 3] / n
        ln = (t[..., 6] - t[..., 5]) / nl
        return {"delta_config": 100 * (pre - base).mean(0), "delta_timing": 100 * (mid - pre).mean(0),
                "delta_total": 100 * (mid - base).mean(0), "chain_len_delta": ln.mean(0)}

    point = stats(np.ones(len(chains)))
    rng = np.random.default_rng(0)
    draws = {m: [] for m in point}
    with np.errstate(invalid="ignore", divide="ignore"):
        for _ in range(args.n_boot):
            w = np.bincount(rng.integers(0, len(chains), len(chains)), minlength=len(chains)).astype(float)
            for m, v in stats(w).items():
                draws[m].append(v)
    res = {"policies": [NAME[p] for p in POL], "n_boot": args.n_boot, "unit": "pp; chain length in subtasks"}
    for m, v in point.items():
        q = np.nanquantile(np.array(draws[m]), [0.025, 0.975], axis=0)
        for j, (ch, k) in enumerate(CELLS):
            res.setdefault(f"{ch}/k{k}", {})[m] = [float(v[j]), float(q[0, j]), float(q[1, j])]
    (ROOT / "results/analysis/maxA_position_pooled.json").write_text(json.dumps(res, indent=1) + "\n")
    for ch, k in CELLS:
        e = res[f"{ch}/k{k}"]
        print(f"{ch:10s} k{k}", {m: [round(x, 2) for x in e[m]] for m in point})


if __name__ == "__main__":
    main()
