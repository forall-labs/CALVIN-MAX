"""Timing decomposition (D-049) and persistent vs reversible relocation (D-050), per policy and channel.

    python scripts/analyze_decomposition.py \
        --policy FLOWER --native <lite results.jsonl ...> --variants <gate / rewind results.jsonl ...> \
                        --transient results/cases/flower_transient_v1/results.jsonl \
        --policy MoDE ... --out results/analysis/decomposition.json

Decomposition, on realized cases that have Pre, Mid and mid_rewind (and mid_oracle where it was run):
    ΔTiming = Mid − Pre = (Mid − mid_oracle) + (mid_oracle − mid_rewind) + (mid_rewind − Pre)
                          queued actions        robot + policy state     residual
Without mid_oracle the first two terms are reported together as Mid − mid_rewind. Event-subtask success,
percentage points, 95% chain-cluster bootstrap.

Transient: each reversible twin is paired with its persistent case (matched_case) on the same chain.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate import load, parse_id  # noqa: E402
from compare_policies import CHANNEL  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def boot(rows, terms, n=5000, seed=0):
    """rows: dicts with 'chain' and arm outcomes; terms: {name: (arm_a, arm_b)} -> mean(a − b) in points + CI."""
    by = defaultdict(list)
    for r in rows:
        by[r["chain"]].append(r)
    chains = list(by)
    rng = np.random.default_rng(seed)
    point = {t: 100 * float(np.mean([r[a] - r[b] for r in rows])) for t, (a, b) in terms.items()}
    draws = defaultdict(list)
    for _ in range(n):
        s = [r for j in rng.integers(0, len(chains), len(chains)) for r in by[chains[j]]]
        for t, (a, b) in terms.items():
            draws[t].append(100 * np.mean([r[a] - r[b] for r in s]))
    return {t: {"mean": round(point[t], 1), "ci": [round(float(np.quantile(draws[t], q)), 1) for q in (0.025, 0.975)]}
            for t in terms}


def decomposition(native, variants):
    groups = defaultdict(list)
    for cid, c in native.items():
        v = variants.get(cid, {})
        if c.get("status") or not all(a in c for a in ("pre", "mid")) or "mid_rewind" not in v:
            continue
        d = parse_id(cid)
        r = {"chain": d["chain"], "base": c["y0"], "pre": c["pre"]["event_success"], "mid": c["mid"]["event_success"],
             "mid_rewind": v["mid_rewind"]["event_success"]}
        if "mid_oracle" in v:
            r["mid_oracle"] = v["mid_oracle"]["event_success"]
        groups[d["change_type"] + (f"@{cid.split('-')[2]}" if d["track"] == "magnitude" else "")].append(r)  # D-067
        if d["change_type"] in CHANNEL:
            groups["channel:" + CHANNEL[d["change_type"]]].append(r)
    out = {}
    for g, rows in sorted(groups.items()):
        terms = {"timing (Mid−Pre)": ("mid", "pre"), "Mid−rewind": ("mid", "mid_rewind"),
                 "residual (rewind−Pre)": ("mid_rewind", "pre")}
        with_oracle = [r for r in rows if "mid_oracle" in r]
        res = {"n": len(rows), "rates": {a: round(100 * float(np.mean([r[a] for r in rows])), 1)
                                         for a in ("base", "pre", "mid", "mid_rewind")},
               "terms": boot(rows, terms)}
        split = {"queue (Mid−oracle)": ("mid", "mid_oracle"), "robot+policy (oracle−rewind)": ("mid_oracle", "mid_rewind")}
        if len(with_oracle) == len(rows):
            res["rates"]["mid_oracle"] = round(100 * float(np.mean([r["mid_oracle"] for r in rows])), 1)
            res["terms"].update(boot(rows, split))
        elif with_oracle:  # e.g. cases the fixed preflight rejects in the oracle run (D-053): split on the four-arm subset
            res["four_arm"] = {"n": len(with_oracle),
                               "rates": {a: round(100 * float(np.mean([r[a] for r in with_oracle])), 1)
                                         for a in ("base", "pre", "mid", "mid_rewind", "mid_oracle")},
                               "terms": boot(with_oracle, {**terms, **split})}
        out[g] = res
    return out


def transient(native, transient_cases):
    man = {c["case_id"]: c for f in sorted((ROOT / "benchmark/manifests").glob("transient_v*.json"))
           for c in json.loads(f.read_text())["cases"]}  # v1: return after 10, Mid+Pre; v2: sweep, Mid only (D-059)
    groups = defaultdict(list)
    blocked = defaultdict(lambda: [0, 0])
    for cid, t in transient_cases.items():
        m = man[cid]
        p = native.get(m["matched_case"])
        if p is None or t.get("status") or p.get("status") or "mid" not in t or not all(a in p for a in ("pre", "mid")):
            continue
        role, ra = m["params"]["role"], m["params"]["return_after"]
        r = {"chain": m["chain_id"], "base": p["y0"], "persist_mid": p["mid"]["event_success"],
             "persist_pre": p["pre"]["event_success"], "rev_mid": t["mid"]["event_success"]}
        if "pre" in t:
            r["rev_pre"] = t["pre"]["event_success"]
        for key in (f"{role}@r{ra}", f"all@r{ra}"):
            groups[key].append(r)
            b = blocked[key]
            b[0] += any("return_blocked" in keys for _, keys in t["mid"].get("tick_log", []))
            b[1] += 1
    terms = {"reversible − persistent (Mid)": ("rev_mid", "persist_mid"),
             "reversible − persistent (Pre)": ("rev_pre", "persist_pre"),
             "reversible Mid − Base": ("rev_mid", "base"), "persistent Mid − Base": ("persist_mid", "base"),
             "reversible timing (Mid−Pre)": ("rev_mid", "rev_pre"), "persistent timing (Mid−Pre)": ("persist_mid", "persist_pre")}
    return {g: {"n": len(rows), "mid_return_blocked": f"{blocked[g][0]}/{blocked[g][1]}",
                "rates": {a: round(100 * float(np.mean([r[a] for r in rows])), 1) for a in rows[0] if a != "chain"},
                "terms": boot(rows, {t: ab for t, ab in terms.items() if all(a in r for r in rows for a in ab)})}
            for g, rows in sorted(groups.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", action="append", required=True)
    ap.add_argument("--native", nargs="+", type=Path, action="append", required=True)
    ap.add_argument("--variants", nargs="+", type=Path, action="append", required=True)
    ap.add_argument("--transient", nargs="*", type=Path, action="append", default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    report = {}
    for i, pol in enumerate(args.policy):
        native = load(args.native[i])
        variants = defaultdict(dict)
        for path in args.variants[i]:
            for line in path.read_text().splitlines():
                r = json.loads(line) if line.strip() else None
                if r and r["status"] == "completed" and r["arm"] in ("mid_oracle", "mid_rewind"):
                    variants[r["case_id"]][r["arm"]] = r
        report[pol] = {"decomposition": decomposition(native, variants)}
        if args.transient and i < len(args.transient) and args.transient[i]:
            report[pol]["transient"] = transient(native, load(args.transient[i]))
    text = json.dumps(report, indent=1)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
