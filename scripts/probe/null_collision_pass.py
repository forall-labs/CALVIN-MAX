"""Is a late null-branch divergence caused by the onset collision pass (D-048) the branch adds and Base lacks?

Reruns one null branch twice in-process, with and without p.performCollisionDetection, against the Base trace.
    python scripts/probe/null_collision_pass.py --base results/base/pi05_smoke --chain E0016 --onset 47 --H 5 --Q 5 --port 6301
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pybullet  # noqa: E402
from run_null_check import branch  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--base", type=Path, required=True)
ap.add_argument("--chain", required=True)
ap.add_argument("--onset", type=int, required=True)
ap.add_argument("--arm", default="mid", choices=["mid", "pre"])
ap.add_argument("--chains", type=Path, default=Path("benchmark/chains/eval_1000.json"))
ap.add_argument("--adapter", default="chunked")
ap.add_argument("--H", type=int, required=True)
ap.add_argument("--Q", type=int, required=True)
ap.add_argument("--port", default="6301")
args = ap.parse_args()
chain = next(c for c in json.loads(args.chains.read_text()) if c["chain_id"] == args.chain)
job = (chain, str(args.base / "traces" / f"{args.chain}.pkl"), args.arm, args.onset, args.port, args.H, args.Q, 0, args.adapter)
print("with collision pass   ", json.dumps(branch(job)), flush=True)
pybullet.performCollisionDetection = lambda *a, **k: None
print("without collision pass", json.dumps(branch(job)), flush=True)
