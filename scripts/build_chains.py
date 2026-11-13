"""Freeze CALVIN's 1,000 evaluation chains, plus a disjoint dev set for calibration (D-011).

    PYTHONPATH=src python3 scripts/build_chains.py --out benchmark/chains

eval_1000.json reproduces the official `get_sequences(1000)` order and initial states.
dev_200.json uses the same sampler with different seeds (worker seed i+1000, shuffle seed 1) and drops
any (initial condition, subtasks) pair that also appears in eval.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from calvin_max.vendor import calvin_multistep_sequences as ms
from calvin_max.vendor.calvin_utils import get_env_state_for_initial_condition, temp_seed


def records(pairs, prefix):
    out = []
    for i, (cond, seq) in enumerate(pairs):
        robot_obs, scene_obs = get_env_state_for_initial_condition(cond)
        out.append(
            {
                "chain_id": f"{prefix}{i:04d}",
                "initial_condition": {k: (int(v) if isinstance(v, (int, np.integer)) else v) for k, v in cond.items()},
                "subtasks": [str(s) for s in seq],
                "robot_obs": robot_obs.tolist(),
                "scene_obs": scene_obs.tolist(),
            }
        )
    return out


def dev_pairs(n, eval_keys):
    possible = {  # the official condition grid and filter from get_sequences
        "led": [0, 1], "lightbulb": [0, 1], "slider": ["right", "left"], "drawer": ["closed", "open"],
        "red_block": ["table", "slider_right", "slider_left"], "blue_block": ["table", "slider_right", "slider_left"],
        "pink_block": ["table", "slider_right", "slider_left"], "grasped": [0],
    }
    from itertools import product

    f = lambda l: l.count("table") in [1, 2] and l.count("slider_right") < 2 and l.count("slider_left") < 2
    states = [dict(zip(possible, v)) for v in product(*possible.values()) if f(v)]
    per = list(map(len, np.array_split(range(2 * n), len(states))))
    pairs = []
    for i, (s, k) in enumerate(zip(states, per)):
        for seq in ms.get_sequences_for_state2((s, k, i + 1000)):
            key = json.dumps([s, list(map(str, seq))], sort_keys=True)
            if key not in eval_keys:
                pairs.append((s, seq))
    with temp_seed(1):
        np.random.shuffle(pairs)
    return pairs[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("benchmark/chains"))
    ap.add_argument("--dev", type=int, default=200)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    eval_pairs = ms.get_sequences(1000, num_workers=1)
    ev = records(eval_pairs, "E")
    keys = {json.dumps([c, list(map(str, s))], sort_keys=True) for c, s in eval_pairs}
    dv = records(dev_pairs(args.dev, keys), "D")
    for name, recs in (("eval_1000.json", ev), (f"dev_{args.dev}.json", dv)):
        blob = json.dumps(recs, indent=0)
        (args.out / name).write_text(blob)
        print(name, len(recs), "sha256", hashlib.sha256(blob.encode()).hexdigest()[:16])
    print("first eval chain:", ev[0]["subtasks"], ev[0]["initial_condition"])


if __name__ == "__main__":
    main()
