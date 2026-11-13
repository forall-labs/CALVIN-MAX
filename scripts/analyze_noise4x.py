"""Strongest sensor-noise rung (4x variant A, magnitude_v2 level L5, 60 cases) on all ten policies (D-120, exploratory).

    python scripts/analyze_noise4x.py      # results/analysis/noise4x.json

FLOWER, MoDE and X-VLA ran these cases in their Lite ladders (<p>_magnitude_v2); the other seven ran only this rung
(<p>_noise4x). Per policy, over all assigned cases (full denominator, aggregate.summarize): Base, Pre and Mid
event-subtask success, Pre - Base (the loss under the corrupted view), Mid - Pre (the mid-onset benefit) and the share
of the loss Mid recovers, with 95% chain-cluster bootstrap intervals; and the same contrasts on realized cases only
(both arms ran), as the Lite ladder's timing effect is reported (analyze_magnitude.delta_timing_realized).
The Base, Pre and Mid rates (all assigned) and the realized share recovered also get 95% chain-cluster intervals
(base_ci, pre_ci, mid_ci, real_share_recovered_ci; the share's draws skip resamples in which Pre loses nothing).
"""
import json
import sys

import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate import load, parse_id, summarize  # noqa: E402
from report_maxA import draws, mean_diff  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results/cases"
SRC = {"FLOWER": "flower_magnitude_v2", "MoDE": "mode1_magnitude_v2", "X-VLA": "xvla_magnitude_v2",
       "pi0.5": "pi05_noise4x", "GR-1": "gr1_noise4x", "Seer": "seer_noise4x", "UniVLA": "univla_noise4x",
       "3DDA": "dda_noise4x", "RoboFlamingo": "rf_noise4x", "OpenVLA-OFT": "oft_noise4x"}


def main():
    ids = set(json.loads((ROOT / "benchmark/manifests/magnitude_v2_L5.json").read_text())["case_ids"])
    out = {}
    for name, d in SRC.items():
        f = R / d / "results.jsonl"
        if not f.exists():
            print(f"{name:13s} missing {d}")
            continue
        g = [(cid, c) for cid, c in load([f]).items() if cid in ids]
        s = summarize(g)
        if s.get("delta_config") is None or not s.get("realized"):
            print(f"{name:13s} no complete cases yet")
            continue
        loss, gain = s["delta_config"][0], s["delta_timing"][0]
        out[name] = {"n": s["n"], "realized": s["realized"], "base": 100 * s["base_sr"], "pre": 100 * s["pre_sr"],
                     "mid": 100 * s["mid_sr"], "pre_minus_base": [100 * v for v in s["delta_config"]],
                     "mid_minus_pre": [100 * v for v in s["delta_timing"]],
                     "share_recovered": (gain / -loss) if loss < 0 else None}
        rows = [{"chain": parse_id(cid)["chain"], "base": c["y0"], "pre": c["pre"]["event_success"],
                 "mid": c["mid"]["event_success"]} for cid, c in g
                if not c.get("status") and not c.get("infra") and "pre" in c and "mid" in c]
        for key, (a_, b_) in (("real_pre_minus_base", ("pre", "base")), ("real_mid_minus_pre", ("mid", "pre"))):
            est, dd = draws(rows, mean_diff(a_, b_))
            out[name][key] = [est, *np.quantile(dd, [0.025, 0.975])]
        lr, gr = out[name]["real_pre_minus_base"][0], out[name]["real_mid_minus_pre"][0]
        out[name]["real_share_recovered"] = gr / -lr if lr < 0 else None

        def share(sample):
            loss = np.mean([r["pre"] - r["base"] for r in sample])
            return np.mean([r["mid"] - r["pre"] for r in sample]) / -loss if loss < 0 else np.nan
        _, dd = draws(rows, share)
        out[name]["real_share_recovered_ci"] = list(np.nanquantile(dd, [0.025, 0.975]))
        # all assigned cases, counted as in aggregate.summarize: an unrealized event keeps Base in both arms
        arows = []
        for cid, c in g:
            if c.get("infra"):
                continue
            if c.get("status") in ("trigger_unreached", "event_unrealizable"):
                yp = ym = c["y0"]
            elif "mid" in c:
                yp, ym = c["pre"]["event_success"], c["mid"]["event_success"]
            else:
                continue
            arows.append({"chain": parse_id(cid)["chain"], "base": c["y0"], "pre": yp, "mid": ym})
        for arm in ("base", "pre", "mid"):
            est, dd = draws(arows, lambda smp, a=arm: 100 * float(np.mean([r[a] for r in smp])))
            assert abs(est - out[name][arm]) < 1e-9, (name, arm, est, out[name][arm])
            out[name][f"{arm}_ci"] = list(np.quantile(dd, [0.025, 0.975]))
        e = out[name]
        print(f"{name:13s} n={e['n']:3d} real={e['realized']:3d} Base {e['base']:5.1f} Pre {e['pre']:5.1f} Mid {e['mid']:5.1f}"
              f"  Pre-Base {e['pre_minus_base'][0]:+6.1f} [{e['pre_minus_base'][1]:+.1f}, {e['pre_minus_base'][2]:+.1f}]"
              f"  Mid-Pre {e['mid_minus_pre'][0]:+6.1f} [{e['mid_minus_pre'][1]:+.1f}, {e['mid_minus_pre'][2]:+.1f}]"
              f"  | realized: Pre-Base {e['real_pre_minus_base'][0]:+6.1f}  Mid-Pre {e['real_mid_minus_pre'][0]:+6.1f} "
              f"[{e['real_mid_minus_pre'][1]:+.1f}, {e['real_mid_minus_pre'][2]:+.1f}]")
    (ROOT / "results/analysis/noise4x.json").write_text(json.dumps(out, indent=1, default=float) + "\n")


if __name__ == "__main__":
    main()
