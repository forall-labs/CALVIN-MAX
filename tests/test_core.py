import unittest

import numpy as np

from calvin_max import behaviour, oracle, triggers
from max_core import schema, stats


class TestStats(unittest.TestCase):
    def test_paired_summary_counts(self):
        y0 = [1, 1, 1, 0, 0, 1]
        y1 = [1, 0, 1, 1, 0, 0]
        s = stats.paired_summary(y0, y1)
        self.assertEqual((s.preserved, s.regression, s.change_success, s.persistent_failure), (2, 2, 1, 1))
        self.assertAlmostEqual(s.delta, 3 / 6 - 4 / 6)
        self.assertAlmostEqual(s.conditional_regression, 2 / 4)

    def test_mcnemar(self):
        self.assertEqual(stats.exact_mcnemar(0, 0), 1.0)
        self.assertAlmostEqual(stats.exact_mcnemar(10, 0), 2 / 2**10)
        self.assertAlmostEqual(stats.exact_mcnemar(3, 3), 1.0)

    def test_decompose_adds_up(self):
        d = stats.decompose([1, 1, 1, 1], [1, 1, 0, 1], [1, 0, 0, 0])
        self.assertAlmostEqual(d["delta_config"] + d["delta_timing"], d["delta_total"])

    def test_chain_bootstrap_contains_point(self):
        rng = np.random.default_rng(1)
        chains = np.repeat(np.arange(50), 4)
        y0 = rng.integers(0, 2, 200)
        y1 = y0 * rng.integers(0, 2, 200)
        point, lo, hi = stats.chain_bootstrap(lambda a, b: b.mean() - a.mean(), chains, y0, y1, n_boot=500)
        self.assertLessEqual(lo, point)
        self.assertLessEqual(point, hi)

    def test_holm(self):
        self.assertEqual(stats.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


class TestSchema(unittest.TestCase):
    def test_rejects_boundary_on_last_subtask(self):
        with self.assertRaises(ValueError):
            schema.Case("c", 0, tuple("abcde"), 5, "target_relocation", "A", "subtask_boundary",
                        schema.Delivery.TELEPORT, 0, "block_red")

    def test_rejects_unknown_change_type(self):
        with self.assertRaises(ValueError):
            schema.Case("c", 0, tuple("abcde"), 1, "teleport_robot", "A", "contact",
                        schema.Delivery.TELEPORT, 0, "block_red")


class TestTriggers(unittest.TestCase):
    def setUp(self):
        t = np.linspace(0, 1, 100)[:, None]
        self.tcp = np.hstack([0.5 - 0.5 * t, np.zeros((100, 2))])  # moves from x=0.5 to x=0
        self.entity = np.zeros((100, 3))

    def test_pre_grasp(self):
        step, inside = triggers.pre_grasp_proximity(self.tcp, self.entity)
        self.assertFalse(inside)
        self.assertGreater(np.linalg.norm(self.tcp[step - 1]), 0.10)
        self.assertLess(np.linalg.norm(self.tcp[step]), 0.10)

    def test_starts_inside(self):
        step, inside = triggers.pre_grasp_proximity(self.tcp * 0.1, self.entity)
        self.assertIsNone(step)
        self.assertTrue(inside)

    def test_transport_and_place(self):
        z = np.r_[np.zeros(40), np.linspace(0, 0.1, 60)]
        ent = np.c_[np.linspace(0, 0.3, 100), np.zeros(100), z]
        held = np.r_[np.zeros(30, bool), np.ones(70, bool)]
        t0 = triggers.transport(ent, held)
        self.assertTrue(held[t0] and ent[t0, 2] > 0.03)
        rec = np.tile([0.3, 0, 0.1], (100, 1))
        self.assertGreaterEqual(triggers.place(ent, rec, held), t0)


class TestBehaviour(unittest.TestCase):
    def test_retargeting_and_stale(self):
        old, new = np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.3, 0.0])
        go_old = np.c_[np.linspace(0.3, 0.05, 20), np.zeros(20), np.zeros(20)]
        go_new = np.c_[np.full(30, 0.05), np.linspace(0.0, 0.3, 30), np.zeros(30)]
        tcp = np.vstack([go_old, go_new])
        lat = behaviour.retargeting_latency(tcp, old, new)
        self.assertTrue(18 <= lat <= 21)
        self.assertAlmostEqual(behaviour.stale_commitment(tcp, old, until=19), 0.25, places=6)
        self.assertFalse(behaviour.ghost_reach(tcp, old, new, radius=0.03))
        self.assertTrue(behaviour.ghost_reach(tcp, old, new, radius=0.06))

    def test_classify(self):
        self.assertEqual(behaviour.classify_response(False, None, False, {"push_red_block_left"}, 0.5), "wrong_task")
        self.assertEqual(behaviour.classify_response(False, None, False, set(), 0.0), "frozen")
        self.assertEqual(behaviour.classify_response(True, 5, True, set(), 0.4), "stale_then_recovered")


class TestOracle(unittest.TestCase):
    def test_rebase_copies_only_moved(self):
        start = {"scene_info": {"movable_objects": {"block_red": {"current_pos": [0, 0, 0]},
                                                    "block_blue": {"current_pos": [1, 1, 1]}},
                                "doors": {}, "lights": {}}}
        post = {"scene_info": {"movable_objects": {"block_red": {"current_pos": [0.1, 0, 0]},
                                                   "block_blue": {"current_pos": [9, 9, 9]}},
                               "doors": {}, "lights": {}}}
        r = oracle.rebase_start_info(start, post, moved_objects=["block_red"])
        self.assertEqual(r["scene_info"]["movable_objects"]["block_red"]["current_pos"], [0.1, 0, 0])
        self.assertEqual(r["scene_info"]["movable_objects"]["block_blue"]["current_pos"], [1, 1, 1])
        self.assertEqual(start["scene_info"]["movable_objects"]["block_red"]["current_pos"], [0, 0, 0])

    def test_shift_keeps_lift_progress_and_support_contact(self):
        # Lifted 4 cm off the table (uid 5), then pushed 10 cm in x while carried (D-048).
        start = {"scene_info": {"movable_objects": {"block_red": {"current_pos": [0.0, 0.0, 0.46],
                                                                  "contacts": [(0, 0, 5, -1, -1)]}},
                                "doors": {}, "lights": {}}}
        post = {"scene_info": {"movable_objects": {"block_red": {"current_pos": [0.1, 0.0, 0.50],
                                                                 "contacts": [(0, 0, 0, 10, -1)]}},
                               "doors": {}, "lights": {}}}
        r = oracle.rebase_start_info(start, post, shifted_objects={"block_red": [0.1, 0.0, 0.0]})
        ref = r["scene_info"]["movable_objects"]["block_red"]
        self.assertAlmostEqual(post["scene_info"]["movable_objects"]["block_red"]["current_pos"][2]
                               - ref["current_pos"][2], 0.04)  # lift progress kept
        self.assertAlmostEqual(ref["current_pos"][0], 0.1)  # push displacement removed
        self.assertEqual(ref["contacts"], [(0, 0, 5, -1, -1)])  # original support contact kept


if __name__ == "__main__":
    unittest.main()


class TestResolve(unittest.TestCase):
    def test_block_relocation_stays_on_surface(self):
        from calvin_max import resolve

        pos = {"block_red": [0.10, -0.09, 0.46], "block_blue": [0.25, -0.09, 0.46], "block_pink": [-0.24, 0.09, 0.46]}
        for seed in range(50):
            r = resolve.resolve_block_relocation(pos, "block_red", 0.06, seed)
            self.assertIsNotNone(r)
            self.assertEqual(resolve.surface_of(r["pos"]), "table")
            self.assertAlmostEqual(np.linalg.norm(np.array(r["pos"][:2]) - np.array(pos["block_red"][:2])), 0.06)
            self.assertGreaterEqual(np.linalg.norm(np.array(r["pos"][:2]) - np.array(pos["block_blue"][:2])), 0.06)

    def test_slider_block_moves_within_slider(self):
        from calvin_max import resolve

        pos = {"block_red": [-0.24, 0.09, 0.46], "block_blue": [0.1, -0.1, 0.46], "block_pink": [0.25, -0.1, 0.46]}
        r = resolve.resolve_block_relocation(pos, "block_red", 0.06, 0)
        self.assertEqual(r["surface"], "slider_left")
        self.assertIsNone(resolve.resolve_block_relocation(pos, "block_red", 0.30, 0))  # cannot fit

    def test_held_block_has_no_surface(self):
        from calvin_max import resolve

        self.assertIsNone(resolve.surface_of([0.1, -0.1, 0.55]))

    def test_distractors_separated(self):
        from calvin_max import resolve

        pos = {"block_red": [0.10, -0.09, 0.46], "block_blue": [0.25, -0.09, 0.46], "block_pink": [-0.24, 0.09, 0.46]}
        r = resolve.resolve_distractors(pos, [0.0, -0.2, 0.6], seed=3)
        cs = [np.array(o["pos"][:2]) for o in r["objects"]]
        self.assertEqual(len(cs), 5)
        self.assertTrue(all(np.linalg.norm(a - b) >= 0.05 for i, a in enumerate(cs) for b in cs[i + 1:]))


class TestTasks(unittest.TestCase):
    def test_all_34_tasks_resolve(self):
        from calvin_max.tasks import designated_entity, task_kind
        from calvin_max.vendor.calvin_multistep_sequences import tasks

        pos = {"block_red": [0.1, -0.1, 0.52], "block_blue": [0.25, -0.1, 0.46], "block_pink": [-0.24, 0.09, 0.46]}
        tcp = [0.1, -0.1, 0.53]  # holding red
        self.assertEqual(len(tasks), 34)
        for t in tasks:
            d = designated_entity(t, pos, tcp)
            self.assertIn("target", d)
            self.assertIn(task_kind(t), ("articulated", "block", "receptacle"))
        self.assertEqual(designated_entity("stack_block", pos, tcp), {"target": "block_red", "receptacle": "block_blue"})
        self.assertEqual(designated_entity("turn_on_led", pos, tcp)["target"], "button_link")

    def test_aabb_distance(self):
        from calvin_max.tasks import aabb_distance

        d = aabb_distance([[0, 0, 0], [2, 0, 0], [0.5, 0.5, 0.5]], [0, 0, 0], [1, 1, 1])
        np.testing.assert_allclose(d, [0, 1, 0])


class TestTracePickle(unittest.TestCase):
    def test_old_pickle_gets_new_fields(self):
        import pickle

        from calvin_max.rollout import Trace

        tr = Trace(subtasks=["a"])
        state = dict(tr.__dict__)
        del state["entity_aabb"], state["scene_obs"]
        old = Trace.__new__(Trace)
        old.__setstate__(state)
        self.assertEqual(old.entity_aabb, {})
        self.assertEqual(pickle.loads(pickle.dumps(old)).scene_obs, [])


class TestOnsetRules(unittest.TestCase):
    def _trace(self):
        from calvin_max.rollout import Trace

        T = 60
        tr = Trace(subtasks=["lift_red_block_table", "place_in_drawer", "turn_on_led", "open_drawer", "close_drawer"])
        tr.subtask_starts, tr.subtask_success = [0], [1]
        tr.actions = [np.zeros(7)] * T
        x = np.linspace(0.30, 0.0, T)
        tr.tcp = [[float(v), 0.0, 0.5] for v in x]
        red_z = np.r_[np.full(40, 0.46), np.linspace(0.46, 0.55, 20)]
        tr.entity_pos = {"block_red": [[0.0, 0.0, float(z)] for z in red_z],
                         "block_blue": [[0.2, 0.1, 0.46]] * T, "block_pink": [[-0.2, 0.1, 0.46]] * T}
        tr.entity_aabb = {"block_red": [[-0.02, -0.02, 0.44, 0.02, 0.02, 0.48]] * T}
        tr.contact = {"block_red": [False] * 35 + [True] * 25}
        return tr

    def test_rules_follow_convention(self):
        from calvin_max.cases import onset_step

        tr = self._trace()
        base_case = {"position": 1, "task": "lift_red_block_table"}
        t, _ = onset_step({**base_case, "onset": "pre_grasp_proximity"}, tr, {"radius": 0.10, "min_steps": 10})
        d_prev = float(np.linalg.norm(np.array(tr.tcp[t - 2][:1])) - 0.02)
        self.assertGreaterEqual(d_prev, 0.10)  # state before the crossing was still outside r
        self.assertEqual(onset_step({**base_case, "onset": "contact"}, tr)[0], 36)
        t_tr, _ = onset_step({**base_case, "onset": "transport"}, tr)
        self.assertTrue(tr.contact["block_red"][t_tr - 1] and tr.entity_pos["block_red"][t_tr - 1][2] > 0.49)
        self.assertEqual(onset_step({**base_case, "onset": "subtask_boundary"}, tr)[0], 0)
        self.assertEqual(onset_step({**base_case, "onset": "place"}, tr)[1], "rule_not_applicable")
        self.assertEqual(onset_step({**base_case, "position": 2, "onset": "contact"}, tr)[1], "subtask_not_reached")


try:
    import scipy  # noqa: F401
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


@unittest.skipUnless(HAVE_SCIPY, "scipy not installed (runs on the Colab venv)")
class TestXVLAAdapter(unittest.TestCase):
    def _adapter(self):
        from calvin_max.adapter import XVLAAdapter

        a = XVLAAdapter.__new__(XVLAAdapter)
        XVLAAdapter.__bases__[0].__bases__[0].__init__(a, 30, 20)  # ChunkedAdapter init, no socket
        a.proprio, a.cameras, a.policy_seed, a.chain_id = None, ("rgb_static", "rgb_gripper"), 0, "E0"
        rng = np.random.default_rng(0)

        def fake_infer(obs, goal):
            if a.proprio is None:
                a.proprio = np.zeros(20)
            ch = rng.normal(size=(30, 20))
            ch[:, 3:9] = np.tile([1, 0, 0, 1, 0, 0], (30, 1))  # identity rotation columns
            return ch
        a.infer = fake_infer
        return a

    def test_restore_matches_continuous_execution(self):
        a = self._adapter()
        obs = {"robot_obs": np.zeros(15)}
        acts = []
        for t in range(7):
            act, log = a.step(obs, "g")
            if t == 0:
                snap = a.snapshot()
            acts.append(act)
        prop_after_7 = a.proprio.copy()
        b = self._adapter()
        b.restore(snap, pop=6)
        np.testing.assert_allclose(b.proprio, prop_after_7)
        self.assertEqual(len(b.state.queue), len(a.state.queue))
        self.assertEqual(acts[0].shape, (8,))  # pos3 + quat4 + gripper


class TestHistoryAdapter(unittest.TestCase):
    def _adapter(self, sent):
        from calvin_max.adapter import HistoryAdapter

        a = HistoryAdapter.__new__(HistoryAdapter)
        HistoryAdapter.__bases__[0].__bases__[0].__init__(a, 1, 1)  # ChunkedAdapter init, no socket
        a.cameras, a.policy_seed, a.chain_id, a.window, a.hist = ("rgb_static",), 0, "E0", 3, ()
        a.frames_needed, a._next_key, a._sent = 0, 0, set()
        server = {}  # what a server's per-connection cache would hold

        class Conn:
            def send(self, msg):
                for f in msg["frames"]:
                    if "rgb" in f:
                        server[f["key"]] = float(f["robot_obs"][0])
                sent.append([server[f["key"]] for f in msg["frames"]])

            def recv(self):
                return {"actions": np.zeros((1, 7))}
        a.conn = Conn()
        return a

    def test_restore_resumes_with_base_window(self):
        def obs(t):
            return {"rgb_obs": {"rgb_static": np.zeros((2, 2, 3))}, "robot_obs": np.full(15, float(t))}

        sent_a, sent_b = [], []
        a = self._adapter(sent_a)
        for t in range(6):
            a.step(obs(t), "g")
            if t == 3:
                snap = a.snapshot()
        self.assertEqual(snap.policy_state, 3)  # only the window length is stored
        b = self._adapter(sent_b)
        b.restore(snap)
        with self.assertRaises(RuntimeError):  # a branch must refill the window before querying
            b.step(obs(4), "g")
        b.set_frames([obs(t) for t in (1, 2, 3)])  # Rollout re-renders these while replaying the prefix
        for t in range(4, 6):
            b.step(obs(t), "g")
        self.assertEqual(sent_a[3], [1.0, 2.0, 3.0])  # window of 3, oldest dropped
        self.assertEqual(sent_b, sent_a[4:])
        a.reset()
        self.assertEqual(a.hist, ())


class TestGate(unittest.TestCase):
    def _adapter(self, seed_chunks):
        from calvin_max.adapter import ChunkedAdapter

        class Stateless(ChunkedAdapter):  # like the policy servers: output depends on (obs, step) only
            def infer(self, obs, goal):
                rng = np.random.default_rng([self.global_step, obs["shift"]])
                return rng.normal(size=(10, 7))

        return Stateless(10, 10)

    def _run(self, a, shift_at=None, steps=40):
        out = []
        for t in range(steps):
            a.global_step = t
            obs = {"shift": 1 if (shift_at is not None and t >= shift_at) else 0}
            act, _ = a.step(obs, "g")
            out.append(act)
        return np.array(out)

    def test_log_only_gate_does_not_change_actions(self):
        native = self._run(self._adapter(0))
        g = self._adapter(0)
        g.gate_every, g.gate_tau = 2, float("inf")
        logged = self._run(g)
        np.testing.assert_array_equal(native, logged)
        self.assertGreater(len(g.gate_log), 0)
        self.assertFalse(any(x[2] for x in g.gate_log))

    def test_gate_replaces_on_disagreement(self):
        g = self._adapter(0)
        g.gate_every, g.gate_tau = 2, 0.0  # any disagreement triggers
        self._run(g)
        self.assertTrue(any(x[2] for x in g.gate_log))


class TestOnsetBoundary(unittest.TestCase):
    def test_onset_on_final_action_is_unreached(self):
        from types import SimpleNamespace
        from calvin_max import cases
        # Subtask 0 spans actions 0..4; the condition first holds at index 4, the successful final action.
        base = SimpleNamespace(subtask_success=[1], subtask_starts=[0, 5], actions=[0] * 9)
        orig = cases._onset_step
        try:
            cases._onset_step = lambda c, b, o: (5, {"rule": "contact"})
            self.assertEqual(cases.onset_step({"position": 1, "onset": "contact"}, base)[1], "fired_at_subtask_end")
            cases._onset_step = lambda c, b, o: (4, {"rule": "contact"})
            self.assertEqual(cases.onset_step({"position": 1, "onset": "contact"}, base)[0], 4)
        finally:
            cases._onset_step = orig
