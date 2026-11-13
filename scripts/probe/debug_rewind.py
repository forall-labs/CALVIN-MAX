"""Colab-only: check mid_rewind returns the robot to its subtask-start state for one case (D-049)."""
import json, pickle, sys
from pathlib import Path
import numpy as np
ROOT = Path("/content/CALVIN-MAX")
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from run_cases import _setup
from calvin_max.cases import plan_case
from calvin_max.events import make_event
cid = sys.argv[1]
case = next(c for c in json.loads((ROOT / "benchmark/manifests/max_v1.json").read_text())["cases"] if c["case_id"] == cid)
chain = next(c for c in json.loads((ROOT / "benchmark/chains/eval_1000.json").read_text()) if c["chain_id"] == case["chain_id"])
base = pickle.loads((ROOT / "results/base/flower_abc/traces" / (chain["chain_id"] + ".pkl")).read_bytes())
plan = plan_case(case, base)
env, runner, init = _setup(chain)
k, info = runner.replay_prefix(base, init, plan["mid_onset"])
print("k", k, "start tcp", np.round(runner.start_robot_state["tcp_pos"], 3), "now", np.round(env.robot.get_observation()[1]["tcp_pos"], 3))
make_event(case["change_type"], plan["payload"])(env)
try:
    runner._rewind_robot()
except Exception as e:
    print("raised", repr(e))
print("after rewind", np.round(env.robot.get_observation()[1]["tcp_pos"], 3))
