"""Chain runner: record Base, then branch Pre / Mid arms by exact action replay (D-007, D-013, D-017).

Step convention: global step t means the t-th action executed in the chain. An onset `o` means
the event is applied after `o` actions have been executed and before action `o` runs.
The loop mirrors CALVIN's `evaluate_sequence` / `rollout`: model.reset() per subtask, start_info
captured at subtask start, EP_LEN steps, and the chain stops at the first failure.
"""

from __future__ import annotations

import hashlib
from dataclasses import MISSING, dataclass, field, fields
from typing import Callable

import numpy as np

from calvin_max.oracle import rebase_start_info

EP_LEN = 360


def state_hash(env) -> str:
    s = env.get_state_obs()
    buf = np.concatenate([s["robot_obs"], s["scene_obs"]]).astype(np.float64).tobytes()
    return hashlib.sha256(buf).hexdigest()[:16]


def env_action(action):
    """Map a stored float64 action to calvin_env's apply_action input.

    7 values = relative (dpos3, deuler3, gripper), the CALVIN default. 8 values = absolute
    (pos3, quat4, gripper), sent as cartesian_abs; used by absolute-pose policies such as X-VLA (D-038).
    """
    a = np.asarray(action, dtype=np.float64)
    if a.shape[-1] == 8:
        return {"type": "cartesian_abs", "action": [*a[:7].tolist(), int(1 if a[7] > 0 else -1)]}
    return a


def physics_step(env, action):
    """env.step's physics path without rendering or info collection."""
    env.robot.apply_action(env_action(action))
    for _ in range(env.action_repeat):
        env.p.stepSimulation(physicsClientId=env.cid)
    env.scene.step()


@dataclass
class Entity:
    """Something whose pose the trace follows: a block (link=-1) or a fixed-body link (handle, button)."""

    name: str
    body: int
    link: int = -1

    def position(self, env):
        if self.link < 0:
            return env.p.getBasePositionAndOrientation(self.body, physicsClientId=env.cid)[0]
        return env.p.getLinkState(self.body, self.link, physicsClientId=env.cid)[0]

    def aabb(self, env):
        lo, hi = env.p.getAABB(self.body, self.link, physicsClientId=env.cid)
        return list(lo) + list(hi)

    def touching_robot(self, env):
        kw = {} if self.link < 0 else {"linkIndexB": self.link}
        return bool(env.p.getContactPoints(bodyA=env.robot.robot_uid, bodyB=self.body, physicsClientId=env.cid, **kw))


@dataclass
class Trace:
    subtasks: list[str]
    actions: list = field(default_factory=list)
    subtask_of_step: list = field(default_factory=list)
    subtask_starts: list = field(default_factory=list)
    subtask_success: list = field(default_factory=list)
    hashes: list = field(default_factory=list)
    scene_obs: list = field(default_factory=list)  # per step, 24 values (CALVIN_NOTES layout)
    queried: list = field(default_factory=list)
    tcp: list = field(default_factory=list)
    entity_pos: dict = field(default_factory=dict)
    entity_aabb: dict = field(default_factory=dict)  # name -> per-step [min xyz, max xyz]
    contact: dict = field(default_factory=dict)
    snapshots: dict = field(default_factory=dict)  # global step of a query -> adapter state after that step
    side_effects: dict = field(default_factory=dict)  # subtask index -> tasks completed beyond the instructed one
    gate_log: list = field(default_factory=list)  # disagreement-gate probes: (global_step, d, replaced)
    oracle_flush: bool = False
    rewind: bool = False  # mid_rewind arm (D-049)
    tick_log: list = field(default_factory=list)  # transient-event actions: (global_step, keys) (D-050)
    lang_override: dict = field(default_factory=dict)  # subtask index -> instruction text (paraphrase_swap)
    onset: int | None = None
    pending_at_onset: int | None = None

    def __setstate__(self, state):
        """Traces pickled before a field existed get its default (fields are only ever added)."""
        for f in fields(self):
            if f.name not in state:
                state[f.name] = f.default_factory() if f.default_factory is not MISSING else f.default
        self.__dict__.update(state)

    @property
    def chain_length(self) -> int:
        n = 0
        for s in self.subtask_success:
            if not s:
                break
            n += 1
        return n

    def arrays(self):
        return (
            np.asarray(self.tcp),
            {k: np.asarray(v) for k, v in self.entity_pos.items()},
            {k: np.asarray(v, dtype=bool) for k, v in self.contact.items()},
        )


class ChainRunner:
    def __init__(self, env, task_oracle, annotations, entities: list[Entity], ep_len: int = EP_LEN,
                 render_every_step: bool = False):
        """render_every_step=True is needed only for policies that consume every frame (history)."""
        self.env, self.oracle, self.ann, self.entities, self.ep_len = env, task_oracle, annotations, entities, ep_len
        self.render_every_step = render_every_step

    # --- bookkeeping ----------------------------------------------------------------------
    def _log_state(self, tr: Trace, k: int, action, queried: bool):
        env = self.env
        tr.actions.append(np.asarray(action, dtype=np.float64))
        tr.subtask_of_step.append(k)
        tr.queried.append(queried)
        tr.hashes.append(state_hash(env))
        tr.scene_obs.append(env.scene.get_obs())
        tr.tcp.append(env.robot.get_observation()[1]["tcp_pos"])
        for e in self.entities:
            tr.entity_pos.setdefault(e.name, []).append(e.position(env))
            tr.entity_aabb.setdefault(e.name, []).append(e.aabb(env))
            tr.contact.setdefault(e.name, []).append(e.touching_robot(env))

    def _goal(self, subtask):
        return self.ann[subtask][0]

    def _run_subtasks(self, tr, adapter, k0, start_info, budget, first_obs, snapshot_queries, gate=None,
                      tick=None, pre_event_info=None):
        """Run the policy from subtask k0 (already started) to the end of the chain.

        tick(env, i) runs before step i of subtask k0 (transient events, D-050); restore_objects in its result
        copies those objects' references back from pre_event_info.
        """
        env, obs = self.env, first_obs
        if getattr(adapter, "needs_env", False):
            adapter.env = env  # camera calibration and link poses (DDAAdapter)
        for k in range(k0, len(tr.subtasks)):
            subtask = tr.subtasks[k]
            if k > k0:
                adapter.reset()
                start_info = env.get_info()
                tr.subtask_starts.append(len(tr.actions))
                budget = self.ep_len
                obs = None
            success = False
            for i in range(budget):
                if tick is not None and k == k0:
                    sub = tick(env, i)
                    if sub:
                        tr.tick_log.append((len(tr.actions), sorted(sub)))
                        if sub.get("restore_objects"):
                            env.p.performCollisionDetection(physicsClientId=env.cid)
                            start_info = rebase_start_info(start_info, pre_event_info, sub["restore_objects"])
                            obs = None  # the world changed since the last render
                if gate is not None and obs is not None and gate(obs):
                    adapter.flush()
                if obs is None and adapter.will_query:
                    obs = env.get_obs()  # lazy rendering: only when the policy will look (D-019)
                adapter.global_step = len(tr.actions)
                action, log = adapter.step(obs, tr.lang_override.get(k) or self._goal(subtask))
                if self.render_every_step:
                    obs, _, _, info = env.step(env_action(action))
                else:
                    physics_step(env, action)
                    obs, info = None, env.get_info()
                self._log_state(tr, k, action, log.queried)
                if log.queried and snapshot_queries:
                    tr.snapshots[len(tr.actions) - 1] = adapter.snapshot()
                if self.oracle.get_task_info_for_set(start_info, info, {subtask}):
                    success = True
                    break
            done = self.oracle.get_task_info(start_info, env.get_info())
            tr.side_effects[k] = sorted(set(done) - {subtask})
            tr.subtask_success.append(int(success))
            if not success:
                break
        return tr

    # --- arms -----------------------------------------------------------------------------
    def run_base(self, adapter, initial_state, subtasks) -> Trace:
        robot_obs, scene_obs = initial_state
        self.env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
        tr = Trace(subtasks=list(subtasks))
        adapter.reset()
        tr.subtask_starts.append(0)
        tr = self._run_subtasks(tr, adapter, 0, self.env.get_info(), self.ep_len, self.env.get_obs(), True)
        tr.gate_log = list(getattr(adapter, "gate_log", []))
        return tr

    def replay_prefix(self, base: Trace, initial_state, upto: int, check_hash=True, render_at=()):
        """Replay base.actions[:upto] without rendering, verifying the state hash at every step.

        Returns the start_info of the subtask that is active at step `upto`. That info is captured
        during the replay at that subtask's first step, matching what Base saw. The observations Base's
        policy saw before the steps in `render_at` are re-rendered into self.replay_obs (history windows).
        """
        robot_obs, scene_obs = initial_state
        self.env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
        k = base.subtask_of_step[upto] if upto < len(base.actions) else base.subtask_of_step[-1]
        start_step = base.subtask_starts[k]
        start_info = self.env.get_info() if start_step == 0 else None
        render_at = set(render_at)
        self.replay_obs = {0: self.env.get_obs()} if 0 in render_at else {}
        for t in range(upto):
            physics_step(self.env, base.actions[t])
            if check_hash and state_hash(self.env) != base.hashes[t]:
                raise ReplayMismatch(f"prefix diverged at step {t}")
            if t + 1 == start_step:
                start_info = self.env.get_info()
            if t + 1 in render_at:
                self.replay_obs[t + 1] = self.env.get_obs()
        self.start_robot_state = start_info["robot_info"]  # for the mid_rewind arm (D-049)
        return k, start_info

    def run_branch(
        self,
        adapter,
        base: Trace,
        initial_state,
        arm: str,
        onset: int,
        event: Callable | None,
        gate=None,
        oracle_flush: bool = False,
        rewind: bool = False,
    ) -> Trace:
        """arm 'mid': restore the native queue at `onset`. arm 'pre': `onset` must be a subtask start.

        `event(env) -> dict(moved_objects=..., moved_doors=..., lights=..., goal=...)`, or None for
        the null event.
        """
        q = max((s for s in base.snapshots if s <= onset - 1), default=None) if arm == "mid" else None
        window = []  # steps whose observations a restored history window holds (HistoryAdapter)
        if q is not None and hasattr(adapter, "set_frames"):
            window = list(range(q - base.snapshots[q].policy_state + 1, q + 1))
        k, start_info = self.replay_prefix(base, initial_state, onset, render_at=window)
        tr = Trace(subtasks=list(base.subtasks))
        # Carry over the shared prefix so chain metrics and hashes line up with Base.
        tr.actions = list(base.actions[:onset])
        tr.subtask_of_step = list(base.subtask_of_step[:onset])
        tr.hashes = list(base.hashes[:onset])
        tr.scene_obs = list(base.scene_obs[:onset])
        tr.queried = list(base.queried[:onset])
        tr.tcp = list(base.tcp[:onset])
        for name in base.entity_pos:
            tr.entity_pos[name] = list(base.entity_pos[name][:onset])
            if name in base.entity_aabb:
                tr.entity_aabb[name] = list(base.entity_aabb[name][:onset])
            tr.contact[name] = list(base.contact[name][:onset])
        tr.subtask_starts = list(base.subtask_starts[: k + 1])
        tr.subtask_success = list(base.subtask_success[:k])
        tr.side_effects = {i: base.side_effects[i] for i in range(k)}

        if arm == "mid":
            if q is None or base.subtask_of_step[q] != k:
                adapter.reset()  # onset lands before subtask k's first query; nothing queued
            else:
                adapter.restore(base.snapshots[q], pop=(onset - 1) - q)
                if window:
                    adapter.set_frames([self.replay_obs[s] for s in window])
            budget = self.ep_len - (onset - base.subtask_starts[k])
        elif arm == "pre":
            if onset != base.subtask_starts[k]:
                raise ValueError("pre arm must branch at a subtask start")
            adapter.reset()
            budget = self.ep_len
        else:
            raise ValueError(arm)

        tr.onset, tr.pending_at_onset = onset, adapter.pending
        pre_event_info = start_info  # Mid: the subtask-start reference; Pre: the state at subtask start
        effect = event(self.env) if event is not None else {}
        if oracle_flush:  # upper bound / mechanism test: drop the stale queue exactly at the event (D-044)
            adapter.flush()
            tr.oracle_flush = True
        if rewind:  # mechanism test: the robot and policy return to subtask start; the world keeps the event (D-049)
            self._rewind_robot()
            adapter.reset()
            tr.rewind = True
        if "goal" in effect:  # instruction_revision
            tr.subtasks[k] = effect["goal"]
        if "lang" in effect:  # paraphrase_swap
            tr.lang_override[k] = effect["lang"]
        # Contacts are only recomputed by a collision pass; without one, a teleported block keeps reporting its
        # pre-event contacts (a dropped block still "touches" the gripper). Dynamics are unchanged (D-048).
        self.env.p.performCollisionDetection(physicsClientId=self.env.cid)
        post = self.env.get_info()
        if arm == "mid":
            start_info = rebase_start_info(
                start_info,
                post,
                effect.get("moved_objects", ()),
                effect.get("moved_doors", ()),
                effect.get("lights", ()),
                effect.get("shifted_objects"),
            )
        else:
            start_info = post
        tr = self._run_subtasks(tr, adapter, k, start_info, budget, self.env.get_obs(), False, gate,
                                effect.get("tick"), pre_event_info)
        tr.gate_log = list(getattr(adapter, "gate_log", []))
        return tr


    def _rewind_robot(self):
        """Reset the arm and gripper to the subtask-start state with zero velocity.

        Refused (RewindUnrealizable) when the robot touches a movable object before or after the reset: a held or
        pushed block would be dropped or teleported into.
        """
        env, r = self.env, self.env.robot
        movable = {o.uid for o in env.scene.movable_objects}

        def touching():
            env.p.performCollisionDetection(physicsClientId=env.cid)
            return {c[2] for c in env.p.getContactPoints(bodyA=r.robot_uid, physicsClientId=env.cid)} & movable

        if touching():
            raise RewindUnrealizable("robot touches an object at onset")
        s = self.start_robot_state
        state = np.array([*s["tcp_pos"], *s["tcp_orn"], s["gripper_opening_width"], *s["arm_joint_states"],
                          s["gripper_action"]])
        r.reset(robot_state=state)
        r.gripper_action = int(s["gripper_action"])
        r.control_gripper(r.gripper_action)
        if touching():
            raise RewindUnrealizable("rewound robot touches an object")


class ReplayMismatch(RuntimeError):
    pass


class RewindUnrealizable(RuntimeError):
    pass
