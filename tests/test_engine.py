import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from evolution_elite_table.config import Config
from evolution_elite_table.engine import Engine
from evolution_elite_table.process import run_process
from evolution_elite_table.programs import canonical, parse_metrics, select_archive, validate
from evolution_elite_table.providers import ProviderError, propose
from evolution_elite_table.report import summary
from evolution_elite_table.storage import Store


class FakeRunner:
    def __init__(self):
        self.ids = []
        self.fail_ids = set()
        self.interrupt_id = None

    def preflight(self):
        pass

    def evaluate(self, row, heldout=False):
        self.ids.append(row["id"])
        if row["id"] == self.interrupt_id:
            raise KeyboardInterrupt()
        if row["id"] in self.fail_ids:
            return {"status": "failed", "error": "out of memory", "wall_seconds": 0.1}
        return {
            "status": "success",
            "metrics": {
                "test_bpb" if heldout else "val_bpb": 10 / row["id"],
                "training_seconds": 1,
                "peak_vram_mb": 1,
            },
            "wall_seconds": 1,
        }


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.bench = self.root / "bench"
        self.bench.mkdir()
        (self.bench / "train.py").write_text("x = 0\n")
        self.requests = []
        self.runner = FakeRunner()

    def tearDown(self):
        self.tmp.cleanup()

    def make(self, **kwargs):
        options = dict(
            benchmark=str(self.bench),
            files=["train.py"],
            adapter="generic",
            backend="local",
            command=[sys.executable, "train.py"],
            provider={"kind": "demo"},
            training_seconds=1,
            timeout_seconds=5,
            evaluations=7,
            max_proposals=30,
            population=3,
            elite_size=3,
            heldout_command=[sys.executable, "train.py", "--heldout"],
        )
        options.update(kwargs)
        engine = Engine.create(Config(**options), self.root / "run")
        engine.runner = self.runner

        def provider(request):
            self.requests.append(request)
            return {
                "source": f"x = {request['candidate_id']}\n",
                "description": "test idea",
                "usage": {"cost_usd": 0.01},
            }

        engine.provider = provider
        return engine

    def run_engine(self, engine):
        with redirect_stdout(StringIO()):
            return engine.run()

    def test_budget_lineage_and_git_provenance(self):
        engine = self.make()
        state = self.run_engine(engine)
        self.assertEqual(state["evaluation_count"], 7)
        self.assertEqual(len(self.runner.ids), 7)
        self.assertEqual(len(state["archive"]), 3)
        self.assertEqual(state["archive"][0], 7)
        row = state["candidates"][-1]
        source = subprocess.check_output(
            [
                "git",
                "--git-dir",
                str(engine.store.root / "provenance.git"),
                "show",
                row["commit"] + ":train.py",
            ],
            text=True,
        )
        self.assertEqual(source, row["source"])
        self.assertEqual(summary(state)["known_api_cost_usd"], 0.06)

    def test_sequential_only_uses_current_best(self):
        engine = self.make(mode="sequential")
        state = self.run_engine(engine)
        self.assertEqual(len(state["archive"]), 1)
        for req in self.requests:
            self.assertEqual(len(req["archive"]), 1)
            self.assertEqual(req["parents"][0]["id"], req["candidate_id"] - 1)

    def test_memory_keeps_context_but_edits_only_best(self):
        engine = self.make(mode="memory")
        self.run_engine(engine)
        self.assertEqual(len(self.requests[-1]["archive"]), 3)
        self.assertEqual(self.requests[-1]["parents"][0]["id"], 6)

    def test_edit_branches_and_remix_has_distinct_parents(self):
        engine = self.make(evaluations=25)
        self.run_engine(engine)
        remixes = [r for r in self.requests if r["operation"] == "remix"]
        self.assertTrue(remixes)
        for r in remixes:
            self.assertGreaterEqual(len(r["parents"]), 2)
            self.assertEqual(len({p["id"] for p in r["parents"]}), len(r["parents"]))

    def test_duplicate_does_not_spend_training_slot(self):
        engine = self.make(max_proposals=4)
        engine.provider = lambda r: {
            "source": "# reformat\nx=0\n",
            "description": "duplicate",
            "usage": {"cost_usd": 0},
        }
        state = self.run_engine(engine)
        self.assertEqual(state["evaluation_count"], 1)
        self.assertEqual(state["proposal_count"], 4)
        self.assertEqual(state["status"], "proposal_limit")
        self.assertEqual(summary(state)["outcomes"]["duplicate"], 4)

    def test_syntax_repair_before_evaluation(self):
        engine = self.make(evaluations=3, population=1)

        def provider(r):
            return {
                "source": "x =" if r["operation"] != "repair" else f"x={r['candidate_id']}\n",
                "description": "repair",
                "usage": {"cost_usd": 0},
            }

        engine.provider = provider
        state = self.run_engine(engine)
        self.assertEqual(state["evaluation_count"], 3)
        self.assertTrue(any(c["status"] == "invalid" for c in state["candidates"]))
        self.assertTrue(
            any(
                c["operation"] == "repair" and c["status"] == "success" for c in state["candidates"]
            )
        )

    def test_recovered_repair_cannot_overrun_queued_proposal_budget(self):
        engine = self.make(evaluations=5, max_proposals=3, population=3)
        baseline = engine._row("baseline", [])
        baseline.update(source=engine.baseline, description="baseline", usage={"cost_usd": 0})
        engine.store.record_source(baseline, engine.state["candidates"])
        baseline["fingerprint"] = canonical(engine.baseline)
        with redirect_stdout(StringIO()):
            engine._evaluate(baseline)
        engine._select()
        engine._plan(3)
        invalid = engine.state["candidates"][1]
        engine._generate(invalid)
        engine._accept(
            invalid, {"source": "x =", "description": "invalid", "usage": {"cost_usd": 0}}
        )
        state = self.run_engine(engine)
        self.assertEqual(state["proposal_count"], 3)
        self.assertEqual(len(self.requests), 2)  # one repair plus one of the two queued requests
        self.assertEqual(state["status"], "proposal_limit")

    def test_failed_training_and_repairs_share_total_budget(self):
        engine = self.make(evaluations=4, max_repairs=2)
        self.runner.fail_ids = set(range(2, 100))
        state = self.run_engine(engine)
        self.assertEqual(len(self.runner.ids), 4)
        self.assertEqual(state["evaluation_count"], 4)
        self.assertLessEqual(max(c["repair_depth"] for c in state["candidates"]), 2)

    def test_interrupt_resume_never_repeats_training_attempt(self):
        engine = self.make(evaluations=5, population=1)
        self.runner.interrupt_id = 2
        with self.assertRaises(KeyboardInterrupt):
            self.run_engine(engine)
        self.assertEqual(engine.state["evaluation_count"], 2)
        self.runner.interrupt_id = None
        state = self.run_engine(engine)
        self.assertEqual(self.runner.ids.count(2), 1)
        self.assertEqual(state["candidates"][1]["status"], "interrupted")
        self.assertEqual(state["evaluation_count"], 5)
        self.assertFalse(summary(state)["evaluation_time_complete"])

    def test_unknown_billing_is_not_zero(self):
        engine = self.make(max_proposals=2)

        def provider(r):
            raise ProviderError("upstream timeout")

        engine.provider = provider
        state = self.run_engine(engine)
        self.assertEqual(summary(state)["unknown_cost_calls"], 2)
        self.assertEqual(state["evaluation_count"], 1)

    def test_provider_failure_is_not_retried_without_bound(self):
        engine = self.make(max_proposals=2)
        engine.provider = lambda r: (_ for _ in ()).throw(ValueError("bad json"))
        state = self.run_engine(engine)
        self.assertEqual(state["proposal_count"], 2)

    def test_frozen_snapshot_detects_modification(self):
        engine = self.make()
        (engine.store.root / "benchmark" / "train.py").write_text("x = 999\n")
        with self.assertRaisesRegex(ValueError, "frozen benchmark"):
            engine.store.load()

    def test_added_file_in_frozen_snapshot_is_rejected(self):
        engine = self.make()
        (engine.store.root / "benchmark" / "extra.py").write_text("x=999")
        with self.assertRaisesRegex(ValueError, "file set"):
            engine.store.load()

    def test_missing_api_key_fails_before_training(self):
        engine = self.make(provider={"kind": "openrouter", "model": "test/model"})
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(ValueError, "API key"):
                self.run_engine(engine)
        self.assertEqual(self.runner.ids, [])
        self.assertEqual(engine.state["proposal_count"], 0)

    def test_source_checksum_detects_modification(self):
        engine = self.make(evaluations=1)
        self.run_engine(engine)
        engine.state["candidates"][0]["source"] = "x=999\n"
        engine.save()
        with self.assertRaisesRegex(ValueError, "checksum"):
            engine.store.load()

    def test_heldout_is_sealed_and_never_visible_to_provider(self):
        engine = self.make()
        self.run_engine(engine)
        chosen = engine.state["archive"][0]
        result = engine.finalize()
        self.assertEqual(result["candidate_id"], chosen)
        self.assertIn("test_bpb", result["metrics"])
        for req in self.requests:
            self.assertNotIn("heldout", json.dumps(req))
            self.assertNotIn("test_bpb", json.dumps(req))
        with self.assertRaisesRegex(ValueError, "sealed"):
            self.run_engine(engine)
        with self.assertRaisesRegex(ValueError, "already reserved"):
            engine.finalize()

    def test_heldout_reservation_survives_interruption(self):
        engine = self.make(evaluations=1)
        self.run_engine(engine)
        self.runner.interrupt_id = 1
        with self.assertRaises(KeyboardInterrupt):
            engine.finalize()
        self.assertEqual(engine.store.load()["heldout"]["status"], "reserved")
        with self.assertRaisesRegex(ValueError, "already reserved"):
            engine.finalize()

    def test_missing_heldout_does_not_seal(self):
        engine = self.make(evaluations=1, heldout_command=[])
        self.run_engine(engine)
        with self.assertRaisesRegex(ValueError, "independent"):
            engine.finalize()
        self.assertIsNone(engine.state["heldout"])

    def test_run_directory_cannot_be_overwritten(self):
        engine = self.make()
        with self.assertRaises(FileExistsError):
            Engine.create(engine.config, engine.store.root)

    def test_lock_prevents_competing_writer(self):
        engine = self.make()
        with engine.store.lock():
            with self.assertRaisesRegex(ValueError, "another process"):
                with Store(engine.store.root).lock():
                    pass


class ProgramTests(unittest.TestCase):
    def test_canonical_ignores_comments_and_docstrings(self):
        self.assertEqual(canonical('"doc"\nx = 1\n'), canonical("# comment\nx=1"))
        self.assertNotEqual(canonical("x=1"), canonical("x=2"))

    def test_compile_rejects_invalid_top_level_return(self):
        with self.assertRaises(SyntaxError):
            canonical("return 42")

    def test_freezes_autoresearch_tail(self):
        baseline = "x=1\n# Final eval\nprint(x)\n"
        validate("x=2\n# Final eval\nprint(x)\n", baseline, "autoresearch")
        with self.assertRaisesRegex(ValueError, "frozen"):
            validate("x=2\n# Final eval\nprint(0)\n", baseline, "autoresearch")

    def test_metric_validation(self):
        valid = "val_bpb: 1.25\ntraining_seconds: 300\npeak_vram_mb: 2048\n"
        self.assertEqual(parse_metrics(valid)["val_bpb"], 1.25)
        for bad in (
            valid + "val_bpb: 0\n",
            valid.replace("1.25", "nan"),
            valid.replace("1.25", "-1"),
            "val_bpb: 0.1\n",
        ):
            with self.assertRaises(ValueError):
                parse_metrics(bad)

    def test_archive_preserves_global_best_with_diversity(self):
        rows = [
            {"id": i, "status": "success", "source": source, "metrics": {"val_bpb": score}}
            for i, source, score in [
                (1, "x=1", 1.0),
                (2, "x=2", 1.1),
                (3, "def f():\n return 2", 1.2),
            ]
        ]
        self.assertEqual(select_archive(rows, 2, 0), [1, 2])
        self.assertEqual(select_archive(rows, 2, 10), [1, 3])

    def test_path_traversal_rejected(self):
        for filename in ("../prepare.py", "/etc/passwd"):
            with self.assertRaises(ValueError):
                Config(
                    benchmark=".", files=["train.py", filename], provider={"kind": "demo"}
                ).validate()

    def test_command_provider_contract(self):
        script = 'import json,sys; r=json.load(sys.stdin); print(json.dumps({"source":"x=1", "description":r["operation"], "usage":{"cost_usd":0.25}}))'
        result = propose(
            {"kind": "command", "command": [sys.executable, "-c", script]}, {"operation": "edit"}
        )
        self.assertEqual(result["description"], "edit")
        self.assertEqual(result["usage"]["cost_usd"], 0.25)

    def test_process_timeout(self):
        code, _, _, duration, timeout = run_process(
            [sys.executable, "-c", "import time; time.sleep(20)"], timeout=0.1
        )
        self.assertTrue(timeout)
        self.assertNotEqual(code, 0)
        self.assertLess(duration, 3)


if __name__ == "__main__":
    unittest.main()
