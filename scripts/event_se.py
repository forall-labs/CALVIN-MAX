"""Bootstrap standard errors for the per-event table in the supplement (Max-A core track): Base success, the three
paired differences, and conditional regression (share of Base successes that fail in Mid).

    /opt/homebrew/bin/python3 scripts/event_se.py

For every policy and core event, the cases are counted exactly as in aggregate.summarize (the source of
maxA_q9_heldout.json): infra errors skipped, an unrealized event keeps its Base outcome in every arm. The draws are
the same chain-cluster bootstrap as stats.chain_bootstrap with summarize's settings (2,000 draws, seed 0), so the
percentile interval reproduces the stored 95% interval; the standard error is the standard deviation of the draws.
Writes results/analysis/maxA_event_se.json: {policy: {core/<event>: {term: [estimate, se, lo, hi]}}}, in % / pp;
terms base_sr, delta_config, delta_timing, delta_total, conditional_regression.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aggregate import load, parse_id  # noqa: E402
from report_maxA import NAME, ROOT  # noqa: E402

POL = ["flower", "mode", "xvla", "pi05", "gr1", "seer", "univla", "dda", "rf", "oft"]
# statistics over the (base, pre, mid) outcome columns of the resampled cases
STATS = {
    "base_sr": lambda z: z[:, 0].mean(),
    "delta_config": lambda z: z[:, 1].mean() - z[:, 0].mean(),
    "delta_timing": lambda z: z[:, 2].mean() - z[:, 1].mean(),
    "delta_total": lambda z: z[:, 2].mean() - z[:, 0].mean(),
    "conditional_regression": lambda z: (z[:, 0] * (1 - z[:, 2])).sum() / max(z[:, 0].sum(), 1),
}
N_BOOT, SEED = 2000, 0


def boot(chains, y, stat):
    """Point, SD of the draws and percentile 95% interval of stat(y), resampling chains as chain_bootstrap."""
    uniq, inverse = np.unique(np.asarray(chains), return_inverse=True)
    members = [np.flatnonzero(inverse == j) for j in range(len(uniq))]
    rng = np.random.default_rng(SEED)
    draws = np.empty(N_BOOT)
    for i in range(N_BOOT):
        pick = rng.integers(0, len(uniq), len(uniq))
        draws[i] = stat(y[np.concatenate([members[j] for j in pick])])
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return [100 * float(stat(y)), 100 * float(draws.std(ddof=1)), 100 * float(lo), 100 * float(hi)]


def main():
    out = {}
    for p in POL:
        groups = defaultdict(lambda: ([], []))
        for cid, c in load([ROOT / f"results/cases/{p}_maxA/results.jsonl"]).items():
            d = parse_id(cid)
            if d["track"] != "core" or c.get("infra"):
                continue
            y = c["y0"]
            if c.get("status") in ("trigger_unreached", "event_unrealizable"):
                yp = ym = y
            else:
                if "mid" not in c:
                    continue
                ym, yp = c["mid"]["event_success"], c["pre"]["event_success"]
            ch, ys = groups[f"core/{d['change_type']}"]
            ch.append(d["chain"]), ys.append((y, yp, ym))
        out[NAME[p]] = {g: {t: boot(ch, np.array(ys, dtype=float), f) for t, f in STATS.items()}
                        for g, (ch, ys) in sorted(groups.items())}
    (ROOT / "results/analysis/maxA_event_se.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
