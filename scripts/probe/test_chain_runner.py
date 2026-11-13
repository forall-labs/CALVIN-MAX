"""Phase 1 protocol test with a scripted policy, one fresh process per stage (D-018).

    python test_chain_runner.py --stage base   --out /content/ct
    python test_chain_runner.py --stage branch --out /content/ct --arm mid --onset 150 --event null
    python test_chain_runner.py --stage branch --out /content/ct --arm pre --onset 0   --event null
    python test_chain_runner.py --stage branch --out /content/ct --arm mid --onset 150 --event teleport
    python test_chain_runner.py --stage oracle --out /content/ct --onset 150

Pass criteria:
- null branches reproduce Base exactly: every hash, chain length, and side effect;
- the teleport branch matches Base up to the onset and diverges after it;
- after rebasing, the oracle reports no task right after the teleport.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from calvin_max.adapter import ScriptedAdapter
from calvin_max.envs import block_entities, load_annotations, make_env, make_task_oracle
from calvin_max.oracle import rebase_start_info
from calvin_max.rollout import ChainRunner

CHAINS = Path(__file__).resolve().parents[2] / "benchmark/chains/eval_1000.json"


def scripted(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    a = np.cumsum(rng.normal(0, 0.15, size=(n, 6)), axis=0)
    a = np.clip(a / (1 + np.abs(a)), -1, 1)
    grip = np.where((np.arange(n) // 60) % 2 == 0, 1.0, -1.0)
    return np.concatenate([a, grip[:, None]], axis=1)


def setup(chain_idx):
    chain = json.loads(CHAINS.read_text())[chain_idx]
    env = make_env(egl=True)
    runner = ChainRunner(env, make_task_oracle(), load_annotations(), block_entities(env))
    init = (np.array(chain["robot_obs"]), np.array(chain["scene_obs"]))
    return chain, env, runner, init


def teleport_target(env, dx=0.08):
    """Move the red block 8 cm in x on the table (a stand-in for target_relocation)."""
    obj = next(o for o in env.scene.movable_objects if o.name == "block_red")
    pos, orn = env.p.getBasePositionAndOrientation(obj.uid, physicsClientId=env.cid)
    env.p.resetBasePositionAndOrientation(obj.uid, [pos[0] + dx, pos[1], pos[2]], orn, physicsClientId=env.cid)
    env.p.resetBaseVelocity(obj.uid, [0, 0, 0], [0, 0, 0], physicsClientId=env.cid)
    return {"moved_objects": ["block_red"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["base", "branch", "oracle"], required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--chain", type=int, default=0)
    ap.add_argument("--arm", default="mid")
    ap.add_argument("--onset", type=int, default=150)
    ap.add_argument("--event", choices=["null", "teleport"], default="null")
    ap.add_argument("--dx", type=float, default=0.08)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    chain, env, runner, init = setup(args.chain)
    adapter = ScriptedAdapter(scripted(), horizon=10, query_interval=5)

    if args.stage == "base":
        tr = runner.run_base(adapter, init, chain["subtasks"])
        (args.out / "base.pkl").write_bytes(pickle.dumps(tr))
        print(json.dumps({"steps": len(tr.actions), "chain_length": tr.chain_length,
                          "subtask_success": tr.subtask_success, "snapshots": len(tr.snapshots),
                          "side_effects": tr.side_effects}))
        return

    base = pickle.loads((args.out / "base.pkl").read_bytes())
    report = {"arm": args.arm, "onset": args.onset, "event": args.event, "dx": args.dx}
    if args.stage == "oracle":
        # Right after a teleport, the raw oracle may fire a false push; the rebased one must report nothing.
        k, start_info = runner.replay_prefix(base, init, args.onset)
        teleport_target(env, args.dx)
        for _ in range(30):  # hold still: no-op test
            env.step(np.r_[np.zeros(6), base.actions[args.onset - 1][-1]])
        post = env.get_info()
        report["oracle_without_rebase"] = sorted(runner.oracle.get_task_info(start_info, post))
        report["oracle_with_rebase"] = sorted(
            runner.oracle.get_task_info(rebase_start_info(start_info, post, ["block_red"]), post)
        )
        print(json.dumps(report))
        return
    event = None if args.event == "null" else (lambda e: teleport_target(e, args.dx))
    tr = runner.run_branch(adapter, base, init, args.arm, args.onset, event)
    n = min(len(tr.hashes), len(base.hashes))
    same = [a == b for a, b in zip(tr.hashes[:n], base.hashes[:n])]
    report.update(
        steps=len(tr.actions),
        base_steps=len(base.actions),
        hashes_identical=all(same) and len(tr.hashes) == len(base.hashes),
        prefix_identical=all(same[: args.onset]),
        first_divergence=next((i for i, s in enumerate(same) if not s), None),
        chain_length=(tr.chain_length, base.chain_length),
        pending_at_onset=tr.pending_at_onset,
        side_effects_equal=tr.side_effects == base.side_effects,
    )
    print(json.dumps(report))


if __name__ == "__main__":
    main()
