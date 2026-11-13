"""Exploratory (D-116): how much of the H3 residual (mid_rewind − Pre) is the step budget.

mid_rewind is a Mid variant: after the reset to the subtask-start pose it keeps Mid's remaining budget,
360 − (onset − subtask start), while Pre has the full 360. A budget-matched Pre counts a Pre success as a failure
when it needed more steps than rewind had. Perceptual core cases with Pre, Mid and mid_rewind, as in H3.

    python scripts/explore_rewind_budget.py flower mode xvla gr1 dda rf pi05 oft
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from report_maxA import NAME, draws, mean_diff  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EP = 360


def steps(tr, k):
    """Steps spent on subtask k (0-based) in a trace."""
    s = tr.subtask_starts
    end = s[k + 1] if len(s) > k + 1 else len(tr.actions)
    return end - s[k]


def budget_rows(p):
    """Per-case rows (chain, pre, pre_matched, mid, mid_rewind, lost) for policy dir p; also used by pool_h3.py."""
    T = ROOT / f"results/cases/{p}_maxA/traces"
    TH = ROOT / f"results/cases/{p}_maxA_h3/traces"
    ids = {}
    for line in (ROOT / f"results/cases/{p}_maxA_h3/results.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r.get("arm") == "mid_rewind" and r["status"] == "completed":
            ids[r["case_id"]] = r
    base_rows = {}
    for cid, rw in ids.items():
        fp, fr = T / f"{cid}.pre.pkl", TH / f"{cid}.mid_rewind.pkl"
        fm = T / f"{cid}.mid.pkl"
        if not (fp.exists() and fr.exists() and fm.exists()):
            continue
        d = cid.split("-")
        if d[0] != "core" or d[1] not in ("sensor_noise_onset", "visual_theme_switch", "illumination_switch",
                                          "camera_shift", "distractor_burst"):
            continue
        k = int(d[-2][1:]) - 1
        pre, rew, mid = (pickle.load(open(f, "rb")) for f in (fp, fr, fm))
        lost = rew.onset - rew.subtask_starts[k]
        budget = EP - lost
        pre_ok = pre.subtask_success[k] if len(pre.subtask_success) > k else 0
        rew_ok = rew.subtask_success[k] if len(rew.subtask_success) > k else 0
        mid_ok = mid.subtask_success[k] if len(mid.subtask_success) > k else 0
        pre_used = steps(pre, k)
        pre_matched = int(pre_ok and pre_used <= budget)
        base_rows[cid] = {"chain": d[-1], "pre": int(pre_ok), "pre_matched": pre_matched, "mid": int(mid_ok),
                          "mid_rewind": int(rew_ok), "lost": lost}
    return list(base_rows.values())


def main():
    out = {}
    for p in sys.argv[1:]:
        rows = budget_rows(p)
        if not rows:
            continue
        lost_all = [r["lost"] for r in rows]
        pre_long = sum(r["pre"] and not r["pre_matched"] for r in rows)
        res, rd = draws(rows, mean_diff("mid_rewind", "pre"))
        resm, rdm = draws(rows, mean_diff("mid_rewind", "pre_matched"))
        fails = sum(1 - r["mid_rewind"] for r in rows)
        out[NAME[p]] = {"n": len(rows), "steps_lost_median": float(np.median(lost_all)),
                        "residual": res, "residual_ci90": list(np.quantile(rd, [0.05, 0.95])),
                        "residual_budget_matched": resm, "residual_budget_matched_ci90": list(np.quantile(rdm, [0.05, 0.95])),
                        "pre_successes_beyond_rewind_budget": pre_long, "rewind_failures": fails}
        e = out[NAME[p]]
        print(f"{NAME[p]:13s} n={e['n']:4d} lost~{e['steps_lost_median']:.0f}  residual {res:+.1f} "
              f"[{e['residual_ci90'][0]:+.1f}, {e['residual_ci90'][1]:+.1f}]  budget-matched {resm:+.1f} "
              f"[{e['residual_budget_matched_ci90'][0]:+.1f}, {e['residual_budget_matched_ci90'][1]:+.1f}]  "
              f"Pre successes beyond rewind's budget {pre_long}")
    (ROOT / "results/analysis/maxA_interim_rewind_budget.json").write_text(json.dumps(out, indent=1, default=float) + "\n")


if __name__ == "__main__":
    main()
