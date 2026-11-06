#!/usr/bin/env python3
"""NEXUS artifact regressions; run with python -B scripts/test_nexus_harness.py."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from nexus.results import evaluation_complete


class EvaluationCompletionTest(unittest.TestCase):
    def test_rejects_error_only_legacy_artifact(self):
        root = Path(__file__).resolve().parents[1]
        for seed in (44, 45, 46):
            path = root / f"tmp/thesis_final_results/eval/tf_seaquest_nexus_symbolic_seed{seed}_eval_mods.json"
            if path.exists():
                self.assertFalse(evaluation_complete(path))

    def test_scores_and_requested_mods_are_required(self):
        good = {"num_episodes": 2, "num_seeds": 1, "results": {
            "base": {"mean_return": 10., "n_episodes": 2, "returns": [0., 20.]},
        }}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "eval.json"
            path.write_text(json.dumps(good))
            self.assertTrue(evaluation_complete(path))
            self.assertFalse(evaluation_complete(path, mod_keys=["base", "no_enemies"]))
            self.assertFalse(evaluation_complete(path, num_seeds=3))
            for field, value in [("mean_return", None), ("mean_return", float("nan")),
                                 ("n_episodes", 0), ("returns", []),
                                 ("returns", [0., float("inf")]), ("error", "bad shape")]:
                bad = copy.deepcopy(good)
                bad["results"]["base"][field] = value
                path.write_text(json.dumps(bad))
                self.assertFalse(evaluation_complete(path), (field, value))
            path.write_text("unfinished JSON {")
            self.assertFalse(evaluation_complete(path))


if __name__ == "__main__":
    unittest.main()
