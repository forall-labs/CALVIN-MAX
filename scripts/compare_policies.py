"""Cross-policy table and the channel x timing test (RESEARCH_PLAN Q9, D-034).

    python scripts/compare_policies.py \
        --policy FLOWER results/cases/flower_lite_v0/results.jsonl results/cases/flower_lite_v1/results.jsonl \
                        results/cases/flower_lite_v1b/results.jsonl \
        --policy MoDE results/cases/mode1_lite_v1/results.jsonl

Per policy:
- the per-event Base/Pre/Mid table (core track);
- pooled ΔTiming = Mid − Pre per channel, with a 95% chain-cluster bootstrap CI;
- the interaction: ΔTiming(perceptual) − ΔTiming(physical), bootstrapped jointly over chains.
Q9 predicts the interaction is > 0.
Only realized cases enter ΔTiming. Unreached/unrealizable cases have Pre = Mid = Base by construction,
so they add zeros and dilute rather than inform.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate import group_key, load, parse_id, summarize  # noqa: E402

CHANNEL = {
    "sensor_noise_onset": "perceptual", "visual_theme_switch": "perceptual", "illumination_switch": "perceptual",
    "camera_shift": "perceptual", "distractor_burst": "perceptual",
    "obstacle_insertion": "physical", "receptacle_relocation": "physical", "target_relocation": "physical",
    "grasp_loss": "physical", "arm_push": "physical",
}


def timing_pairs(cases):
    """(chain, channel, y_pre, y_mid) for realized cases that have both arms (core + new tracks)."""
    out = []
    for cid, c in cases.items():
        d = parse_id(cid)
        if d["track"] not in ("core", "new") or d["change_type"] not in CHANNEL:
            continue
        if "pre" in c and "mid" in c and not c.get("infra"):
            out.append((d["chain"], CHANNEL[d["change_type"]], c["pre"]["event_success"], c["mid"]["event_success"]))
    return out


def bootstrap_channels(pairs, n_boot=5000, seed=0):
    chains = sorted({p[0] for p in pairs})
    by_chain = defaultdict(list)
    for p in pairs:
        by_chain[p[0]].append(p)
    rng = np.random.default_rng(seed)

    def stats(sample):
        res = {}
        for ch in ("perceptual", "physical"):
            v = [m - pr for (_, c, pr, m) in sample if c == ch]
            res[ch] = float(np.mean(v)) if v else np.nan
        res["interaction"] = res["perceptual"] - res["physical"]
        return res

    point = stats(pairs)
    draws = defaultdict(list)
    for _ in range(n_boot):
        pick = rng.integers(0, len(chains), len(chains))
        sample = [p for j in pick for p in by_chain[chains[j]]]
        for k, v in stats(sample).items():
            draws[k].append(v)
    out = {}
    for k, v in point.items():
        arr = np.array(draws[k])
        arr = arr[~np.isnan(arr)]
        lo, hi = (np.quantile(arr, [0.025, 0.975]) if len(arr) else (np.nan, np.nan))
        n = sum(1 for p in pairs if p[1] == k) if k != "interaction" else len(pairs)
        out[k] = {"estimate": v, "ci95": [float(lo), float(hi)], "n": n,
                  "p_one_sided_gt0": float(np.mean(arr <= 0)) if len(arr) else None}
    return out


def bootstrap_pooled(pairs_by_policy, n_boot=5000, seed=0):
    """Equal-weight mean over policies of each channel's ΔTiming and of the interaction. Policies share eval
    chains, so a chain is resampled jointly across policies."""
    chains = sorted({p[0] for pairs in pairs_by_policy.values() for p in pairs})
    by_chain = {name: defaultdict(list) for name in pairs_by_policy}
    for name, pairs in pairs_by_policy.items():
        for p in pairs:
            by_chain[name][p[0]].append(p)
    rng = np.random.default_rng(seed)

    def stats(pick):
        per = []
        for name in pairs_by_policy:
            sample = [p for j in pick for p in by_chain[name].get(chains[j], [])]
            v = {ch: [m - pr for (_, c, pr, m) in sample if c == ch] for ch in ("perceptual", "physical")}
            per.append({ch: float(np.mean(x)) if x else np.nan for ch, x in v.items()})
        res = {ch: float(np.mean([r[ch] for r in per])) for ch in ("perceptual", "physical")}
        res["interaction"] = res["perceptual"] - res["physical"]
        return res

    point = stats(np.arange(len(chains)))
    draws = defaultdict(list)
    for _ in range(n_boot):
        for k, v in stats(rng.integers(0, len(chains), len(chains))).items():
            draws[k].append(v)
    out = {}
    for k, v in point.items():
        arr = np.array(draws[k])
        arr = arr[~np.isnan(arr)]
        out[k] = {"estimate": v, "ci95": [float(x) for x in np.quantile(arr, [0.025, 0.975])],
                  "p_one_sided_gt0": float(np.mean(arr <= 0)), "policies": list(pairs_by_policy)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", nargs="+", action="append", required=True, metavar=("NAME", "RESULTS"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    report, pairs_by_policy = {}, {}
    for spec in args.policy:
        name, files = spec[0], [Path(f) for f in spec[1:]]
        cases = load(files)
        groups = defaultdict(list)
        for cid, c in cases.items():
            groups[group_key(cid)].append((cid, c))
        table = {k: summarize(g) for k, g in sorted(groups.items()) if k.startswith("core/")}
        pairs_by_policy[name] = timing_pairs(cases)
        report[name] = {"core": table, "channel_timing": bootstrap_channels(pairs_by_policy[name])}
    pct = lambda x: "" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:+.1f}"  # noqa: E731
    events = sorted({k for r in report.values() for k in r["core"]})
    print(f"{'event':28s}" + "".join(f"{n:>34s}" for n in report))
    print(f"{'':28s}" + "".join(f"{'Base/Pre/Mid   dTot   dTim':>34s}" for _ in report))
    for e in events:
        row = f"{e:28s}"
        for n in report:
            v = report[n]["core"].get(e, {})
            if v.get("n"):
                row += f"{100*v['base_sr']:>9.1f}/{100*v.get('pre_sr', np.nan):.1f}/{100*v['mid_sr']:.1f} {pct(v['delta_total'][0]):>6} {pct((v.get('delta_timing') or [np.nan])[0]):>6}"
            else:
                row += f"{'':>34s}"
        print(row)
    print("\nChannel x timing (ΔTiming = Mid − Pre, realized cases):")
    for n, r in report.items():
        ct = r["channel_timing"]
        print(f"  {n:8s} perceptual {pct(ct['perceptual']['estimate'])} {[pct(x) for x in ct['perceptual']['ci95']]} (n={ct['perceptual']['n']})"
              f" | physical {pct(ct['physical']['estimate'])} {[pct(x) for x in ct['physical']['ci95']]} (n={ct['physical']['n']})"
              f" | interaction {pct(ct['interaction']['estimate'])} {[pct(x) for x in ct['interaction']['ci95']]}"
              f" p(<=0)={ct['interaction']['p_one_sided_gt0']:.3f}")
    if len(pairs_by_policy) > 1:
        pooled = bootstrap_pooled(pairs_by_policy)
        report["pooled"] = pooled
        print("  pooled   " + " | ".join(f"{k} {pct(v['estimate'])} {[pct(x) for x in v['ci95']]}" for k, v in pooled.items())
              + f" p(<=0)={pooled['interaction']['p_one_sided_gt0']:.3f}")
    if args.out:
        args.out.write_text(json.dumps(report, indent=1, default=float))


if __name__ == "__main__":
    main()
