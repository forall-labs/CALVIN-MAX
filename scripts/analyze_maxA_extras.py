"""Exploratory analyses on the complete Max-A runs (D-119; not pre-registered, reported as exploratory).

    python scripts/analyze_maxA_extras.py            # writes results/analysis/maxA_extras.json

1. position   ΔTotal / ΔConfig / ΔTiming and chain-length change by event position k (analyze_position.profile).
2. hazard     discrete-time hazard of failing subtask j = k+1..5 after the event subtask, on the H6 cases (event
              subtask solved in both Base and Mid). Logistic model per policy:
                  logit h = α_(j−k) + β_next·Mid·[j = k+1] + β_later·Mid·[j > k+1]
              so β_next is the aftershock on the next subtask and β_later whether it persists further down the
              chain. Pooled: policy-specific α, common β. Odds ratios with 95% chain-cluster bootstrap intervals
              (the pre-registration's chain random effect is replaced by resampling chains; 1,000 draws, seed 0).
3. variants   ΔTotal per event and parameter variant (A, B), with 95% chain-cluster intervals.
4. coverage   core events: trigger coverage (event realized) and response coverage (realized and the policy queried at least
              once after the onset within the event subtask, from the Mid traces), as in LIBERO-MAX Fig. 10.
5. null       null-event checks per policy: Lite null-track cases where they exist, and the onboarding null gates
              (branches bit-identical to Base / branches with identical subtask outcomes).
"""
import json
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aggregate import load, parse_id, summarize  # noqa: E402
from analyze_position import profile  # noqa: E402
from compare_policies import CHANNEL  # noqa: E402
from report_maxA import DIR, NAME, draws  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results/cases"
POL = ["flower", "mode", "xvla", "pi05", "gr1", "seer", "univla", "dda", "rf", "oft"]
NULL_LITE = {"flower": "flower_lite_v0", "mode": "mode1_lite_v1", "xvla": "xvla_lite_v1"}
# D-128: the other seven policies ran the same Lite null (and intent) cases later, served as in Max-A.
NULL_LITE.update({p: f"{p}_lite_ni" for p in ("pi05", "gr1", "seer", "univla", "dda", "rf", "oft")
                  if (Path(__file__).resolve().parents[1] / f"results/cases/{p}_lite_ni/results.jsonl").exists()})
# Onboarding null gates recorded in docs/DECISIONS.md without a log in results/gates.
NULL_GATE_DOC = {"flower": (8, 8, "D-024"), "mode": (12, 12, "D-040, 3 replicas"), "xvla": (8, 8, "D-041")}


# π0.5's one diverged branch (E0016, step 524 of 535) keeps every subtask outcome (D-073); its log omits them.
NULL_SAME_OUTCOME = {"pi05": True}


def leading(succ):
    return next((j for j, ok in enumerate(succ) if not ok), len(succ))


# ---------------------------------------------------------------- hazard
def hazard_rows(cases):
    """One row per (case, arm, subtask j at risk): chain, d = j − k (1..4), mid, fail."""
    out = []
    for cid, c in cases.items():
        d = parse_id(cid)
        if d["track"] not in ("core", "new") or d["change_type"] not in CHANNEL or "mid" not in c or c.get("infra"):
            continue
        k, base_len = d["k"], c.get("len0")
        if base_len is None or k >= 5:
            continue
        m = c["mid"]["subtask_success"]
        if not (len(m) >= k and m[k - 1] and base_len >= k):
            continue
        for arm, L in (("base", base_len), ("mid", leading(m))):
            for j in range(k + 1, min(L + 1, 5) + 1):
                out.append((d["chain"], j - k, int(arm == "mid"), int(j == L + 1)))
    return out


def design(rows, pol_ix=None, npol=1):
    """Columns: α per (policy, d) then β_next, β_later."""
    X = np.zeros((len(rows), npol * 4 + 2))
    y = np.empty(len(rows))
    for i, r in enumerate(rows):
        chain, dd, mid, fail = r[:4]
        p = pol_ix[i] if pol_ix is not None else 0
        X[i, p * 4 + dd - 1] = 1
        if mid:
            X[i, -2 if dd == 1 else -1] = 1
        y[i] = fail
    keep = X.any(axis=0)
    keep[-2:] = True
    return X[:, keep], y


def logit_fit(X, y, iters=50, ridge=1e-6):
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        eta = np.clip(X @ b, -30, 30)
        p = 1 / (1 + np.exp(-eta))
        W = p * (1 - p) + 1e-9
        H = X.T @ (X * W[:, None]) + ridge * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - p) - ridge * b)
        b += step
        if np.abs(step).max() < 1e-8:
            break
    return b[-2:]


def hazard(rows_by_pol, n=1000, seed=0):
    """Per policy and pooled: odds ratios exp(β_next), exp(β_later) with chain-cluster bootstrap intervals."""
    res = {}
    rng = np.random.default_rng(seed)
    for name, rows in list(rows_by_pol.items()) + [("pooled", None)]:
        if rows is None:
            rows = [r + (i,) for i, rs in enumerate(rows_by_pol.values()) for r in rs]
            npol = len(rows_by_pol)
        else:
            rows = [r + (0,) for r in rows]
            npol = 1
        by = defaultdict(list)
        for r in rows:
            by[r[0]].append(r)
        chains = list(by)

        def fit(rs):
            X, y = design(rs, [r[4] for r in rs], npol)
            return logit_fit(X, y)

        est = fit(rows)
        bs = np.array([fit([r for j in rng.integers(0, len(chains), len(chains)) for r in by[chains[j]]])
                       for _ in range(n)])
        base_next = np.mean([r[3] for r in rows if r[1] == 1 and not r[2]])
        mid_next = np.mean([r[3] for r in rows if r[1] == 1 and r[2]])
        base_later = np.mean([r[3] for r in rows if r[1] > 1 and not r[2]])
        mid_later = np.mean([r[3] for r in rows if r[1] > 1 and r[2]])
        res[name] = {"n_rows": len(rows), "n_chains": len(chains),
                     "hazard_next": {"base": 100 * base_next, "mid": 100 * mid_next},
                     "hazard_later": {"base": 100 * base_later, "mid": 100 * mid_later},
                     "or_next": float(np.exp(est[0])), "or_next_ci95": list(np.exp(np.quantile(bs[:, 0], [.025, .975]))),
                     "or_later": float(np.exp(est[1])), "or_later_ci95": list(np.exp(np.quantile(bs[:, 1], [.025, .975])))}
        e = res[name]
        print(f"  {name:13s} next OR {e['or_next']:.2f} [{e['or_next_ci95'][0]:.2f}, {e['or_next_ci95'][1]:.2f}]  "
              f"later OR {e['or_later']:.2f} [{e['or_later_ci95'][0]:.2f}, {e['or_later_ci95'][1]:.2f}]  "
              f"h_next {base_next:.3f}->{mid_next:.3f}  h_later {base_later:.3f}->{mid_later:.3f}")
    return res


# ---------------------------------------------------------------- variants, coverage, null
def variants(cases):
    groups = defaultdict(list)
    for cid, c in cases.items():
        d = parse_id(cid)
        if d["track"] in ("core", "new") and d["change_type"] in CHANNEL:
            groups[(d["change_type"], cid.split("-")[2])].append((cid, c))
    return {f"{ev}/{v}": {"n": s["n"], "base_sr": s["base_sr"], "mid_sr": s.get("mid_sr"), "delta_total": s["delta_total"]}
            for (ev, v), g in sorted(groups.items()) for s in [summarize(g)]}


def coverage(p, cases):
    """Core cases (the eight shared events) with the full denominator, as in the diagnostics figure: trigger =
    realized; response = realized and a policy query after the onset inside the event subtask (Mid trace)."""
    T = R / f"{p}_maxA/traces"
    n = trig = resp = missing = 0
    for cid, c in cases.items():
        d = parse_id(cid)
        if d["track"] != "core" or d["change_type"] not in CHANNEL or c.get("infra"):
            continue
        n += 1
        if "mid" not in c:
            continue
        trig += 1
        f = T / f"{cid}.mid.pkl"
        if not f.exists():  # a handful of traces were not mirrored; count the case as responded only if realized
            missing += 1
            continue
        tr = pickle.loads(f.read_bytes())
        k = d["k"] - 1
        s = tr.subtask_starts
        end = s[k + 1] if len(s) > k + 1 else len(tr.actions)
        resp += int(any(tr.queried[tr.onset:end]))
    return {"n": n, "trigger": 100 * trig / n, "response": 100 * resp / (n - missing), "traces_missing": missing}


def null_rows(p):
    out = {}
    if p in NULL_LITE:
        g = [(cid, c) for cid, c in load([R / f"{NULL_LITE[p]}/results.jsonl"]).items() if cid.startswith("null")]
        s = summarize(g)
        out["lite_null"] = {"n": s["n"], "base_sr": 100 * s["base_sr"], "mid_sr": 100 * s["mid_sr"],
                            "pre_sr": 100 * s["pre_sr"] if s.get("pre_sr") is not None else None,
                            "delta_total": 100 * s["delta_total"][0]}
    logs = [f for f in (ROOT / f"results/gates/{p}_null.log", ROOT / f"results/gates/{p}_smoke_null.log") if f.exists()]
    summ = []
    for f in logs:  # the largest null gate run for the policy; logs list every diverged branch, not always the rest
        lines = [json.loads(x) for x in f.read_text().splitlines() if x.startswith("{")]
        tot = next(x for x in lines if "branches" in x)
        diverged = [x for x in lines if "chain_id" in x and not x["identical"]]
        same = sum((x["success"][0] == x["success"][1]) if "success" in x else NULL_SAME_OUTCOME.get(p, False)
                   for x in diverged)
        summ.append({"branches": tot["branches"], "identical": tot["identical"], "same_outcomes": tot["identical"] + same,
                     "source": str(f.relative_to(ROOT))})
    if summ:
        out["gate"] = max(summ, key=lambda x: x["branches"])
    elif p in NULL_GATE_DOC:
        b, i, src = NULL_GATE_DOC[p]
        out["gate"] = {"branches": b, "identical": i, "same_outcomes": i, "source": src}
    return out


def main():
    rep = {"position": {}, "variants": {}, "coverage": {}, "null": {}}
    hz = {}
    for p in POL:
        cases = load([R / f"{p}_maxA/results.jsonl"])
        n = NAME[p]
        rep["position"][n] = profile(cases)
        rep["variants"][n] = variants(cases)
        rep["coverage"][n] = coverage(p, cases)
        rep["null"][n] = null_rows(p)
        hz[n] = hazard_rows(cases)
        cv = rep["coverage"][n]
        print(f"{n:13s} coverage trigger {cv['trigger']:.1f} response {cv['response']:.1f}  null {rep['null'][n]}")
    print("hazard (odds ratio of failing, Mid vs Base)")
    rep["hazard"] = hazard(hz)
    (ROOT / "results/analysis/maxA_extras.json").write_text(json.dumps(rep, indent=1, default=float) + "\n")


if __name__ == "__main__":
    main()
