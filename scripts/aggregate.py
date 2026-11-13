"""Aggregate results.jsonl into the paper's per-event table (RESEARCH_PLAN §4), with chain-cluster CIs.

    python scripts/aggregate.py --results results/cases/flower_lite_v0/results.jsonl --out results/cases/flower_lite_v0/summary.json

Scoring rules (LIBERO-MAX semantics plus D-025):
- Y0 = Base success on the event subtask.
- trigger_unreached / event_unrealizable: Dynamic = Base (Y_mid = Y_pre = Y0); the case stays in the denominator.
- infra_error rows are reported and must be rerun; they are excluded from rates.

Per change_type: n, coverage, Base/Pre/Mid SR, delta_total, delta_config (Pre-Base), delta_timing (Mid-Pre),
four-way transitions, conditional regression, exact McNemar, 95% chain-bootstrap CIs, chain-length change.
instruction_revision has no Base on the revised task, so it reports Pre/Mid SR on that task and the
timing contrast only.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from max_core.stats import chain_bootstrap, exact_mcnemar, paired_summary


def load(paths):
    """Rows from one or more results.jsonl files. A case takes ALL its rows from the LAST file that has it,
    so a rerun of some event types (e.g. after a realization fix) replaces them wholesale."""
    owner, by_file, metadata = {}, [], []
    for i, path in enumerate(paths):
        sidecar = Path(path).parent / "base_outcomes.json"
        metadata.append(json.loads(sidecar.read_text())["outcomes"] if sidecar.exists() else {})
        rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
        by_file.append(rows)
        for r in rows:
            owner[r["case_id"]] = i
    rows = [r for i, rs in enumerate(by_file) for r in rs if owner[r["case_id"]] == i]
    cases = defaultdict(dict)
    for r in rows:
        c = cases[r["case_id"]]
        c.setdefault("rows", []).append(r)
        if r.get("base_event_success") is not None:
            c["y0"] = r["base_event_success"]
        if r.get("base_chain_length") is not None:
            c["len0"] = r["base_chain_length"]
        if r["arm"] is None and r["status"] in ("trigger_unreached", "event_unrealizable"):
            c["status"] = r["status"]
        elif r["status"] == "completed":
            c[r["arm"]] = r
        elif r["status"] == "infra_error":
            c.setdefault("infra", []).append(r["arm"])
    # An arm (or plan) that failed earlier but completed in a later rerun is not an infra error (D-043).
    for cid, c in cases.items():
        chain = parse_id(cid)["chain"]
        success = metadata[owner[cid]].get(chain)
        if success is not None:
            k = parse_id(cid)["k"] - 1
            c.setdefault("y0", int(len(success) > k and success[k]))
            length = next((j for j, ok in enumerate(success) if not ok), len(success))
            c.setdefault("len0", length)
        if c.get("status") in ("trigger_unreached", "event_unrealizable"):
            # Old resume logic ran some arms despite an earlier failed preflight. Preserve raw rows,
            # but quarantine those arms for ALL consumers, including realized-only analyses.
            arms = [key for key, value in c.items() if isinstance(value, dict) and value.get("status") == "completed"]
            c["quarantined_arms"] = arms
            for arm in arms:
                del c[arm]
        if "infra" in c:
            completed = any(isinstance(v, dict) and v.get("status") == "completed" for v in c.values())
            left = [] if "status" in c else [a for a in c["infra"] if (a is None and not completed) or (a is not None and a not in c)]
            if left:
                c["infra"] = left
            else:
                del c["infra"]
    return cases


def parse_id(cid):
    # core/new/test: <track>-<change_type>-<variant>-k<k>-<chain>; phase: phase-<change_type>-<onset>-k<k>-<chain>;
    # null/intent: <track>-<change_type>-k<k>-<chain>
    parts = cid.split("-")
    d = {"track": parts[0], "change_type": parts[1], "chain": parts[-1], "k": int(parts[-2][1:])}
    if parts[0] == "phase":
        d["onset"] = parts[2]
    return d


def group_key(cid):
    d = parse_id(cid)
    if d["track"] == "phase":
        return f"phase/{d['change_type']}@{d['onset']}"
    return f"{d['track']}/{d['change_type']}"


def summarize(group):
    y0, ypre, ymid, chains, len0, lmid = [], [], [], [], [], []
    realized = infra = 0
    for cid, c in group:
        if c.get("infra"):
            infra += 1
            continue
        if "y0" not in c:
            raise ValueError(f"{cid}: missing Base outcome; generate base_outcomes.json with backfill_base_metadata.py")
        y = c["y0"]
        if c.get("status") in ("trigger_unreached", "event_unrealizable"):
            yp = ym = y
            lm = c.get("len0")
        else:
            if "mid" not in c:
                continue
            realized += 1
            ym = c["mid"]["event_success"]
            yp = c["pre"]["event_success"] if "pre" in c else None
            lm = c["mid"]["chain_length"]
        y0.append(y), ypre.append(yp), ymid.append(ym), chains.append(parse_id(cid)["chain"])
        if c.get("len0") is not None and lm is not None:
            len0.append(c["len0"]), lmid.append(lm)
    if not y0:
        return {"n": 0, "infra_errors": infra}
    has_pre = all(v is not None for v in ypre)
    y0, ymid = np.array(y0), np.array(ymid)
    ypre = np.array(ypre, dtype=float) if has_pre else None
    s = paired_summary(y0, ymid)
    nb = 2000
    diff = lambda a, b: b.mean() - a.mean()  # noqa: E731
    out = {
        "n": int(len(y0)), "realized": realized, "coverage": realized / len(y0), "infra_errors": infra,
        "quarantined_arms": sum(len(c.get("quarantined_arms", [])) for _, c in group),
        "base_sr": float(y0.mean()), "mid_sr": float(ymid.mean()),
        "delta_total": chain_bootstrap(diff, chains, y0, ymid, n_boot=nb),
        "transitions": {"11": s.preserved, "01": s.change_success, "10": s.regression, "00": s.persistent_failure},
        "conditional_regression": s.conditional_regression,
        "mcnemar_p": exact_mcnemar(s.regression, s.change_success),
    }
    if has_pre:
        out["pre_sr"] = float(ypre.mean())
        out["delta_config"] = chain_bootstrap(diff, chains, y0, ypre, n_boot=nb)
        out["delta_timing"] = chain_bootstrap(diff, chains, ypre, ymid, n_boot=nb)
    if len0:
        out["chain_len_base"], out["chain_len_mid"] = float(np.mean(len0)), float(np.mean(lmid))
    return out


def summarize_intent(group):
    pre, mid = [], []
    for cid, c in group:
        if "mid" in c and "pre" in c:
            pre.append(c["pre"]["event_success"]), mid.append(c["mid"]["event_success"])
    return {"n_realized": len(pre), "revised_pre_sr": float(np.mean(pre)) if pre else None,
            "revised_mid_sr": float(np.mean(mid)) if mid else None,
            "delta_timing": float(np.mean(mid) - np.mean(pre)) if pre else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, nargs="+", required=True, help="later files override earlier ones per case")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    cases = load(args.results)
    groups = defaultdict(list)
    for cid, c in cases.items():
        groups[group_key(cid)].append((cid, c))
    report = {}
    for key, g in sorted(groups.items()):
        report[key] = summarize_intent(g) if key.endswith("instruction_revision") else summarize(g)
    txt = json.dumps(report, indent=1)
    print(txt)
    if args.out:
        args.out.write_text(txt)


if __name__ == "__main__":
    main()
