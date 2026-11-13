"""Regression tests for validity and denominators across interrupted experiment runs."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from aggregate import load, summarize
from analyze_mechanism import boot_ci
from backfill_base_metadata import backfill
from compare_policies import bootstrap_channels, bootstrap_pooled, timing_pairs
from run_cases import resume_state
from max_core.stats import chain_bootstrap


class ResultValidityTests(unittest.TestCase):
    cid = "core-target_relocation-A-k2-E0001"

    def rows(self):
        return [{"case_id": self.cid, "arm": None, "status": "event_unrealizable", "detail": {"valid": False}},
                {"case_id": self.cid, "arm": "mid", "status": "completed", "event_success": 0,
                 "chain_length": 1, "base_event_success": 1, "base_chain_length": 5},
                {"case_id": self.cid, "arm": "pre", "status": "completed", "event_success": 1,
                 "chain_length": 5}]

    def write(self, root, rows, name="results.jsonl"):
        p = Path(root) / name
        p.write_text("".join(json.dumps(r) + "\n" for r in rows))
        return p

    def test_failed_preflight_survives_resume_and_all_analyses(self):
        rows = self.rows()
        done, blocked = resume_state(rows)
        self.assertIn(self.cid, blocked)
        self.assertIn((self.cid, "mid"), done)
        with tempfile.TemporaryDirectory() as d:
            cases = load([self.write(d, rows)])
            self.assertNotIn("mid", cases[self.cid])
            self.assertEqual(timing_pairs(cases), [])
            report = summarize(list(cases.items()))
            self.assertEqual((report["n"], report["realized"], report["quarantined_arms"]), (1, 0, 2))
            self.assertEqual(report["delta_total"][0], 0)

    def test_rejected_case_keeps_denominator_after_metadata_backfill(self):
        with tempfile.TemporaryDirectory() as d:
            p = self.write(d, self.rows()[:1])
            base = Path(d) / "summary.json"
            base.write_text(json.dumps({"results": [{"chain_id": "E0001", "success": [1, 1, 1, 1, 1]}]}))
            raw = p.read_bytes()
            with self.assertRaisesRegex(ValueError, "missing Base"):
                summarize(list(load([p]).items()))
            backfill(base, [p])
            report = summarize(list(load([p]).items()))
            self.assertEqual((report["n"], report["base_sr"], report["mid_sr"]), (1, 1, 1))
            self.assertEqual(p.read_bytes(), raw)

    def test_infrastructure_failure_is_retryable_and_completion_resolves_it(self):
        error = {"case_id": self.cid, "arm": None, "status": "infra_error"}
        self.assertEqual(resume_state([error]), (set(), set()))
        with tempfile.TemporaryDirectory() as d:
            self.assertTrue(load([self.write(d, [error])])[self.cid]["infra"])
            cases = load([self.write(d, [error, self.rows()[1]])])
            self.assertNotIn("infra", cases[self.cid])
            self.assertIn("mid", cases[self.cid])

    def test_later_version_replaces_old_validity_decision(self):
        with tempfile.TemporaryDirectory() as d:
            old = self.write(d, self.rows(), "old.jsonl")
            new = self.write(d, self.rows()[1:], "new.jsonl")
            cases = load([old, new])
            self.assertIn("mid", cases[self.cid])
            self.assertEqual(len(timing_pairs(cases)), 1)

    def test_aftershock_bootstrap_resamples_chains(self):
        x, y, chains = [0, 0, 1, 1], [1, 1, 1, 1], ["A", "A", "B", "B"]
        point, ci = boot_ci(x, y, chains, n=200)
        expected = chain_bootstrap(lambda a, b: a.mean() - b.mean(), chains, x, y, n_boot=200)
        self.assertEqual((point, *ci), expected)

    def test_pooled_interaction_weights_policies_equally(self):
        a = [("E1", "perceptual", 0, 1), ("E2", "physical", 1, 1)]  # interaction +1
        b = [("E1", "perceptual", 0, 0), ("E2", "physical", 1, 1), ("E3", "physical", 1, 1)]  # 0
        pooled = bootstrap_pooled({"A": a, "B": b}, n_boot=50)
        self.assertEqual(pooled["interaction"]["estimate"], 0.5)
        self.assertEqual(bootstrap_pooled({"A": a}, n_boot=50)["interaction"]["estimate"],
                         bootstrap_channels(a, n_boot=50)["interaction"]["estimate"])


if __name__ == "__main__":
    unittest.main()
