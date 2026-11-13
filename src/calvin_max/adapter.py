"""Policy adapter contract (D-017).

Wraps a CALVIN policy behind the evaluator's `reset()` / `step(obs, goal)` interface, adds the
native action queue (horizon H, query interval Q), and supports `snapshot()` / `restore()`.

A Mid branch must hand control back with the policy's *native* queue intact, exactly as it
was in Base at the onset step. Re-running the policy over the whole replayed prefix would cost
a render and an inference per step. Instead, Base stores a snapshot after every query; a branch
restores the last snapshot at or before the onset and pops the actions executed since.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class StepLog:
    queried: bool
    action: np.ndarray  # float64, exactly what was sent to env.step


@dataclass
class AdapterState:
    queue: list = field(default_factory=list)
    steps_since_query: int = 0
    policy_state: Any = None  # obs history, RNG state, anything the policy carries between steps


class ChunkedAdapter:
    """Base class: subclasses implement `infer(obs, goal) -> (H, 7) array` and, if the policy
    keeps state between calls, `get_policy_state()` / `set_policy_state()`."""

    def __init__(self, horizon: int, query_interval: int):
        self.H, self.Q = horizon, query_interval
        self.state = AdapterState()
        self.queries_since_reset = 0
        self.global_step = 0  # set by the runner before each step; keys per-query policy noise (D-023)
        # Disagreement gate (D-044). Off unless gate_every is set. With gate_tau = inf it only logs.
        self.gate_every, self.gate_tau, self.gate_m = None, float("inf"), 5
        self.gate_log = []  # (global_step, disagreement, replaced)

    # --- to implement per policy -------------------------------------------------------
    def infer(self, obs, goal) -> np.ndarray:
        raise NotImplementedError

    def reset_policy(self):
        """Clear policy-internal state at subtask start (CALVIN calls model.reset())."""

    def get_policy_state(self):
        return None

    def set_policy_state(self, s):
        pass

    # --- protocol ------------------------------------------------------------------------
    def reset(self):
        self.state = AdapterState()
        self.queries_since_reset = 0
        self.reset_policy()

    def flush(self):
        """Drop pending actions so the next step re-queries (used by gates, oracle flush)."""
        self.state.queue = []

    def step(self, obs, goal) -> tuple[np.ndarray, StepLog]:
        queried = False
        if self._gate_due():
            fresh = np.asarray(self.infer(obs, goal), dtype=np.float64)[: self.H]
            m = min(len(self.state.queue), self.gate_m)
            d = self.disagreement(fresh[:m], self.state.queue[:m])
            replace = d > self.gate_tau
            self.gate_log.append((self.global_step, d, bool(replace)))
            if replace:  # the world no longer matches the plan: adopt the fresh chunk
                self.state.queue = [a for a in fresh[: self.Q]]
                self.state.steps_since_query = 0
                self.queries_since_reset += 1
                queried = True
        if not self.state.queue:
            chunk = np.asarray(self.infer(obs, goal), dtype=np.float64)[: self.H]
            self.state.queue = [a for a in chunk[: self.Q]]
            self.state.steps_since_query = 0
            self.queries_since_reset += 1
            queried = True
        action = self.state.queue.pop(0)
        action = np.asarray(action, dtype=np.float64).copy()
        action[-1] = 1.0 if action[-1] > 0 else -1.0  # CALVIN asserts gripper in {-1, 1}
        self.state.steps_since_query += 1
        return action, StepLog(queried=queried, action=action)

    def snapshot(self) -> AdapterState:
        s = copy.deepcopy(self.state)
        s.policy_state = copy.deepcopy(self.get_policy_state())
        return s

    def restore(self, snap: AdapterState, pop: int = 0):
        """Restore a post-query snapshot, then drop `pop` actions already executed since."""
        self.state = copy.deepcopy(snap)
        self.set_policy_state(copy.deepcopy(snap.policy_state))
        del self.state.queue[:pop]
        self.state.steps_since_query += pop

    def _gate_due(self) -> bool:
        s = self.state
        return bool(self.gate_every and s.queue and s.steps_since_query > 0 and s.steps_since_query % self.gate_every == 0)

    @staticmethod
    def disagreement(fresh, queued, dims=6):
        """Mean L2 distance between a fresh chunk and the still-queued actions (arm dims only)."""
        q = np.asarray(queued, dtype=np.float64)[:, :dims]
        return float(np.mean(np.linalg.norm(np.asarray(fresh, dtype=np.float64)[: len(q), :dims] - q, axis=1)))

    @property
    def will_query(self) -> bool:
        """True if the next step() calls the policy (a regular query or a gate probe), so the runner must render."""
        return (not self.state.queue) or self._gate_due()

    @property
    def pending(self) -> int:
        """Queued actions not yet executed: the stale-action exposure if an event lands now."""
        return len(self.state.queue)


class ScriptedAdapter(ChunkedAdapter):
    """Deterministic stand-in policy for protocol tests: replays a fixed action sequence."""

    def __init__(self, actions: np.ndarray, horizon: int = 10, query_interval: int = 5):
        super().__init__(horizon, query_interval)
        self.actions = np.asarray(actions, dtype=np.float64)
        self.t = 0

    def infer(self, obs, goal):
        chunk = self.actions[self.t : self.t + self.H]
        if len(chunk) < self.H:
            chunk = np.vstack([chunk, np.repeat(self.actions[-1:], self.H - len(chunk), axis=0)])
        self.t += self.Q
        return chunk

    def get_policy_state(self):
        return self.t

    def set_policy_state(self, s):
        self.t = s


class RemoteAdapter(ChunkedAdapter):
    """Queries a policy server (scripts/adapters/*_server.py) over a local socket.

    The model stays loaded in one long-lived process while every rollout runs in a fresh simulator
    process (D-018). Each query carries `noise_key = (policy_seed, chain_id, global_step)`, so the server
    can seed the policy's sampling noise and a null branch reproduces Base (D-023).
    """

    def __init__(self, address, horizon: int, query_interval: int, policy_seed: int, chain_id: str,
                 cameras=("rgb_static", "rgb_gripper"), authkey: bytes = b"calvin-max"):
        super().__init__(horizon, query_interval)
        from multiprocessing.connection import Client

        self.conn = Client(address, authkey=authkey)
        self.policy_seed, self.chain_id, self.cameras = policy_seed, chain_id, cameras

    def infer(self, obs, goal):
        self.conn.send({
            "op": "infer",
            "rgb": {k: obs["rgb_obs"][k] for k in self.cameras},
            "robot_obs": obs["robot_obs"],
            "lang": goal,
            "noise_key": [self.policy_seed, self.chain_id, self.global_step],
        })
        reply = self.conn.recv()
        if "error" in reply:
            raise RuntimeError(reply["error"])
        return np.asarray(reply["actions"], dtype=np.float64)

    def close(self):
        self.conn.close()


def pick_port(ports: str, key: str) -> int:
    """`ports` is "6101" or "6101,6102,...": spread jobs over policy-server replicas by a stable hash of key."""
    import zlib

    ps = [int(p) for p in str(ports).split(",")]
    return ps[zlib.crc32(key.encode()) % len(ps)]


class XVLAAdapter(RemoteAdapter):
    """X-VLA (absolute ee6d actions) following 2toINF/X-VLA evaluation/calvin/calvin_client.py exactly.

    - proprio (20) = [tcp_pos3, rot6d(euler xyz), gripper_open>0] + zeros(10), set from obs at subtask start;
      after every executed action, proprio[:10] = that predicted action[:10]. This is policy-internal state,
      so it is part of the snapshot.
    - each query returns 30 actions; the first 20 are executed (H=30, Q=20).
    - env action = [pos3, quat4(from rot6d), +1 if gripper < 0.8 else -1] (8 values -> cartesian_abs).
    """

    def __init__(self, address, policy_seed, chain_id, **kw):
        super().__init__(address, horizon=30, query_interval=20, policy_seed=policy_seed, chain_id=chain_id, **kw)
        self.proprio = None

    def reset_policy(self):
        self.proprio = None

    def get_policy_state(self):
        return None if self.proprio is None else self.proprio.copy()

    def set_policy_state(self, s):
        self.proprio = None if s is None else np.array(s, dtype=np.float64)

    @staticmethod
    def _euler_to_rot6d(e):
        from scipy.spatial.transform import Rotation as R

        return R.from_euler("xyz", e, degrees=False).as_matrix()[..., :, :2].reshape(6)

    @staticmethod
    def _rot6d_to_quat(v6):
        from scipy.spatial.transform import Rotation as R

        a1, a2 = v6[0:6:2], v6[1:6:2]
        b1 = a1 / np.linalg.norm(a1)
        proj = np.sum(b1 * a2) * b1
        b2 = (a2 - proj) / np.linalg.norm(a2 - proj)
        return R.from_matrix(np.stack((b1, b2, np.cross(b1, b2)), axis=-1)).as_quat()

    def infer(self, obs, goal):
        if self.proprio is None:
            ro = obs["robot_obs"]
            p = np.concatenate([ro[:3], self._euler_to_rot6d(ro[3:6]), [float(ro[-1] > 0.0)]])
            self.proprio = np.concatenate([p, np.zeros_like(p)])
        self.conn.send({"op": "infer", "rgb": {k: obs["rgb_obs"][k] for k in self.cameras}, "lang": goal,
                        "proprio": self.proprio.tolist(), "noise_key": [self.policy_seed, self.chain_id, self.global_step]})
        reply = self.conn.recv()
        if "error" in reply:
            raise RuntimeError(reply["error"])
        return np.asarray(reply["actions"], dtype=np.float64)  # (30, 20): raw predicted ee6d actions

    def restore(self, snap, pop: int = 0):
        """Base updated proprio[:10] on every executed action, so after popping, proprio must match the last
        popped action, not the snapshot's."""
        popped = list(snap.queue[:pop])
        super().restore(snap, pop)
        if popped:
            self.proprio[:10] = np.asarray(popped[-1], dtype=np.float64)[:10]

    def step(self, obs, goal):
        queried = False
        if not self.state.queue:
            chunk = self.infer(obs, goal)[: self.H]
            self.state.queue = [a for a in chunk[: self.Q]]
            self.state.steps_since_query = 0
            self.queries_since_reset += 1
            queried = True
        raw = np.asarray(self.state.queue.pop(0), dtype=np.float64)
        self.proprio[:10] = raw[:10]
        action = np.concatenate([raw[:3], self._rot6d_to_quat(raw[3:9]), [1.0 if raw[9] < 0.8 else -1.0]])
        self.state.steps_since_query += 1
        return action, StepLog(queried=queried, action=action)


class DDAAdapter(RemoteAdapter):
    """3D Diffuser Actor (absolute trajectories from RGB-D point clouds), following nickgkan/3d_diffuser_actor
    online_evaluation_calvin (evaluate_policy.rollout, evaluate_utils.prepare_visual_states / prepare_proprio_states).

    - Each query sends both cameras' RGB-D and calibration; the server deprojects them as the reference does. The
      gripper camera's view comes from its link pose (utils_with_calvin.get_gripper_camera_view_matrix). The static
      camera's is the nominal one ("stale": a bumped camera is not re-calibrated, D-079) or, with calibration
      "live", whatever the env camera has now, as the reference reads env.cameras[0].
    - Proprio history: robot_obs at the last 3 queries of the subtask, the first repeated to fill (the reference
      keeps obs['proprio'] across the executed actions and appends one entry per query). Policy state.
    - The server returns the trajectory (n, 7) = absolute [pos, euler XYZ, gripper +-1], all executed (H = Q = n).
      Env action = [pos, pybullet quaternion of the euler angles, gripper] (8 values -> cartesian_abs), which is
      what calvin_env makes of the reference's (pos, euler, gripper) tuple.
    """

    needs_env = True  # the runner sets self.env: camera calibration and the gripper camera's link pose

    def __init__(self, address, horizon: int, query_interval: int, policy_seed: int, chain_id: str,
                 calibration: str = "stale", **kw):
        super().__init__(address, horizon, query_interval, policy_seed, chain_id, **kw)
        self.calibration, self.env, self.hist = calibration, None, ()

    def reset_policy(self):
        self.hist = ()

    def get_policy_state(self):
        return [np.array(h) for h in self.hist]

    def set_policy_state(self, s):
        self.hist = tuple(np.asarray(h) for h in (s or ()))

    def _cameras(self):
        import pybullet as pb

        out = {}
        for cam in self.env.cameras:
            if cam.name == "gripper":
                pos, orn = pb.getLinkState(bodyUniqueId=cam.robot_uid, linkIndex=cam.gripper_cam_link,
                                           physicsClientId=cam.cid)[:2]
                rot = np.array(pb.getMatrixFromQuaternion(orn)).reshape(3, 3)
                view, fov = pb.computeViewMatrix(pos, pos + rot[:, 1], -rot[:, 2]), cam.fov
            elif self.calibration == "stale":
                view, fov = pb.computeViewMatrix(cam.look_from, cam.look_at, cam.up_vector), cam.fov0
            else:
                view, fov = cam.viewMatrix, cam.fov
            out[cam.name] = {"viewMatrix": list(view), "fov": float(fov), "width": cam.width, "height": cam.height}
        return out

    def infer(self, obs, goal):
        h = self.hist + (np.asarray(obs["robot_obs"], dtype=np.float64),)
        self.hist = ((h[0],) * 3 + h)[-3:]
        self.conn.send({
            "op": "infer",
            "rgb": {k: obs["rgb_obs"][k] for k in ("rgb_static", "rgb_gripper")},
            "depth": {k: obs["depth_obs"][k] for k in ("depth_static", "depth_gripper")},
            "cams": self._cameras(),
            "proprio_obs": [x.tolist() for x in self.hist],
            "lang": goal,
            "noise_key": [self.policy_seed, self.chain_id, self.global_step],
        })
        reply = self.conn.recv()
        if "error" in reply:
            raise RuntimeError(reply["error"])
        import pybullet as pb

        traj = np.asarray(reply["actions"], dtype=np.float64)
        return np.array([[*a[:3], *pb.getQuaternionFromEuler(a[3:6]), a[6]] for a in traj])  # (n, 8)


class _Frame:
    """One observation in a history window; `key` names it on this adapter's server connection."""

    __slots__ = ("key", "rgb", "robot_obs")

    def __init__(self, key, rgb, robot_obs):
        self.key, self.rgb, self.robot_obs = key, rgb, robot_obs


class HistoryAdapter(RemoteAdapter):
    """Policies conditioned on the last `window` observations of the current subtask (GR-1, Seer: 10 frames,
    e.g. bytedance/GR-1 evaluation/calvin_evaluation.py). The window is policy state, kept here rather than in
    the server so a branch resumes with Base's history.

    A snapshot stores only the window's length: Base snapshots after every step (Q = 1), and storing frames
    would make traces ~100x larger. A branch re-renders those frames while replaying the Base prefix
    (Rollout.run_branch -> set_frames); renders are deterministic, so they equal what Base saw.
    Each query sends the window's frame keys, with pixels only for frames this connection has not sent yet;
    the server caches their preprocessed tensors per connection.
    """

    def __init__(self, address, horizon: int, query_interval: int, policy_seed: int, chain_id: str,
                 window: int = 10, **kw):
        super().__init__(address, horizon, query_interval, policy_seed, chain_id, **kw)
        self.window = window
        self.hist = ()
        self.frames_needed = 0
        self._next_key, self._sent = 0, set()

    def _frame(self, obs):
        self._next_key += 1
        return _Frame(self._next_key, {k: obs["rgb_obs"][k] for k in self.cameras}, np.asarray(obs["robot_obs"]))

    def reset_policy(self):
        self.hist, self.frames_needed = (), 0

    def get_policy_state(self):
        return len(self.hist)

    def set_policy_state(self, s):
        self.hist, self.frames_needed = (), int(s or 0)

    def set_frames(self, observations):
        """Refill the window after restore() with the re-rendered observations of its steps, oldest first."""
        if len(observations) != self.frames_needed:
            raise ValueError(f"window needs {self.frames_needed} frames, got {len(observations)}")
        self.hist, self.frames_needed = tuple(self._frame(o) for o in observations), 0

    def infer(self, obs, goal):
        if self.frames_needed:
            raise RuntimeError("restored history window was not refilled (set_frames)")
        self.hist = (self.hist + (self._frame(obs),))[-self.window:]
        frames = []
        for h in self.hist:
            f = {"key": h.key, "robot_obs": h.robot_obs}
            if h.key not in self._sent:
                f["rgb"] = h.rgb
                self._sent.add(h.key)
            frames.append(f)
        self.conn.send({
            "op": "infer",
            "frames": frames,
            "lang": goal,
            "noise_key": [self.policy_seed, self.chain_id, self.global_step],
        })
        reply = self.conn.recv()
        if "error" in reply:
            raise RuntimeError(reply["error"])
        return np.asarray(reply["actions"], dtype=np.float64)


def make_adapter(kind: str, port, H: int, Q: int, seed: int, chain_id: str, key: str):
    """Runner-side factory. kind: 'chunked' (relative-action servers: FLOWER, MoDE, pi0.5), 'xvla', or
    'history[:window]' (GR-1, Seer; RoboFlamingo: history:13), 'dda[:live]' (3D Diffuser Actor)."""
    addr = ("127.0.0.1", pick_port(port, key))
    if kind == "xvla":
        return XVLAAdapter(addr, seed, chain_id)
    if kind.startswith("dda"):  # "dda" (stale static calibration, D-079) or "dda:live"
        return DDAAdapter(addr, H, Q, seed, chain_id, calibration=kind.split(":")[1] if ":" in kind else "stale")
    if kind.startswith("history"):  # "history" (10 frames) or "history:<window>"
        window = int(kind.split(":")[1]) if ":" in kind else 10
        return HistoryAdapter(addr, H, Q, seed, chain_id, window=window)
    return RemoteAdapter(addr, H, Q, seed, chain_id)
