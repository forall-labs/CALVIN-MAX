"""Calibrate the disagreement-gate threshold on no-event Base runs that only logged probes (D-044).

    python scripts/calibrate_gate.py --traces results/base/flower_dev_gatelog/traces --quantile 0.99 --out tau.json

Uses no events and no outcomes: tau is the given quantile of probe disagreements under normal execution, so
the gate fires on about (1 - quantile) of probes when nothing has changed.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", type=Path, required=True)
    ap.add_argument("--quantile", type=float, default=0.99)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    d = []
    for p in sorted(args.traces.glob("*.pkl")):
        d += [g[1] for g in pickle.loads(p.read_bytes()).gate_log]
    d = np.asarray(d)
    tau = float(np.quantile(d, args.quantile))
    rep = {"probes": int(len(d)), "quantile": args.quantile, "tau": tau,
           "percentiles": {q: float(np.quantile(d, q)) for q in (0.5, 0.9, 0.95, 0.99, 0.999)}}
    args.out.write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep))


if __name__ == "__main__":
    main()
