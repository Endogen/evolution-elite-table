"""Regressions discovered in the second implementation review."""

import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from evolution_elite_table.config import Config
from evolution_elite_table.engine import Engine
from evolution_elite_table.programs import select_archive
from evolution_elite_table.providers import ProviderError, propose
from evolution_elite_table.report import summary


class ValidationRegressions(unittest.TestCase):
    def test_command_string_is_not_an_argv_list(self):
        with self.assertRaises(ValueError):
            Config(benchmark=".", command="python train.py", provider={"kind": "demo"}).validate()

    def test_heldout_command_is_validated_before_search(self):
        with self.assertRaises(ValueError):
            Config(
                benchmark=".", heldout_command="python test.py", provider={"kind": "demo"}
            ).validate()

    def test_malformed_numeric_values_raise_actionable_errors(self):
        for value in (True, "300", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                Config(benchmark=".", training_seconds=value, provider={"kind": "demo"}).validate()

    def test_parent_mount_cannot_expose_heldout_data(self):
        with self.assertRaises(ValueError):
            Config(
                benchmark=".",
                provider={"kind": "demo"},
                mounts=[{"source": "/data", "target": "/data"}],
                heldout_mounts=[{"source": "/data/test", "target": "/test-data"}],
            ).validate()

    def test_files_cannot_have_normalization_aliases(self):
        with self.assertRaises(ValueError):
            Config(
                benchmark=".", files=["train.py", "./train.py"], provider={"kind": "demo"}
            ).validate()

    def test_provider_json_array_becomes_provider_error(self):
        with self.assertRaises(ProviderError):
            propose({"kind": "command", "command": [sys.executable, "-c", "print('[]')"]}, {})

    def test_zero_diversity_does_not_parse_ast_for_ranking(self):
        rows = [
            {"id": i, "status": "success", "source": f"x={i}", "metrics": {"val_bpb": i}}
            for i in range(1, 5)
        ]
        with patch(
            "evolution_elite_table.programs.features",
            side_effect=AssertionError("unnecessary parsing"),
        ):
            self.assertEqual(select_archive(rows, 3), [1, 2, 3])


class RecoveryRegressions(unittest.TestCase):
    def test_buffered_proposal_survives_training_interruption(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            bench = root / "bench"
            bench.mkdir()
            (bench / "train.py").write_text("x=0\n")
            cfg = Config(
                benchmark=str(bench),
                files=["train.py"],
                adapter="generic",
                backend="local",
                provider={"kind": "demo"},
                command=[sys.executable, "train.py"],
                evaluations=4,
                population=3,
                training_seconds=0,
                timeout_seconds=5,
            )
            engine = Engine.create(cfg, root / "run")
            calls = []

            def provider(req):
                calls.append(req["candidate_id"])
                return {
                    "source": f"x={req['candidate_id']}\n",
                    "description": "hypothesis",
                    "usage": {"cost_usd": 0.1},
                }

            class Runner:
                stop = True

                def preflight(self):
                    pass

                def evaluate(self, row, heldout=False):
                    if row["id"] == 2 and self.stop:
                        raise KeyboardInterrupt()
                    return {
                        "status": "success",
                        "metrics": {
                            "val_bpb": 1 / row["id"],
                            "training_seconds": 0,
                            "peak_vram_mb": 0,
                        },
                    }

            runner = Runner()
            engine.provider, engine.runner = provider, runner
            with redirect_stdout(StringIO()), self.assertRaises(KeyboardInterrupt):
                engine.run()
            runner.stop = False
            with redirect_stdout(StringIO()):
                engine.run()
            self.assertEqual(sorted(calls), [2, 3, 4])
            self.assertEqual(summary(engine.state)["unknown_cost_calls"], 0)
            self.assertAlmostEqual(summary(engine.state)["known_api_cost_usd"], 0.3)
            self.assertEqual(engine.state["candidates"][2]["status"], "success")


class ConcurrencyRegressions(unittest.TestCase):
    def test_repairs_and_background_proposals_share_one_limit(self):
        import threading
        import time
        from concurrent.futures import ThreadPoolExecutor

        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            bench = root / "bench"
            bench.mkdir()
            (bench / "train.py").write_text("x=0")
            cfg = Config(
                benchmark=str(bench),
                files=["train.py"],
                adapter="generic",
                backend="local",
                command=[sys.executable, "train.py"],
                provider={"kind": "demo"},
                proposal_workers=2,
            )
            engine = Engine.create(cfg, root / "run")
            active = peak = 0
            lock = threading.Lock()

            def provider(request):
                nonlocal active, peak
                with lock:
                    active += 1
                    peak = max(peak, active)
                time.sleep(0.02)
                with lock:
                    active -= 1
                return {"source": "x=1", "description": "test", "usage": {"cost_usd": 0}}

            engine.provider = provider
            with ThreadPoolExecutor(max_workers=6) as pool:
                list(pool.map(engine._invoke, [{"candidate_id": i} for i in range(1, 7)]))
            self.assertEqual(peak, 2)


class StudyRegressions(unittest.TestCase):
    def test_environment_guard_rejects_cross_run_drift(self):
        from evolution_elite_table.study import StudyGuard

        with tempfile.TemporaryDirectory() as root:
            guard = StudyGuard(Path(root) / "study-manifest.json", [{"seed": 0}])
            original = {
                "manifest": {"prepare.py": "original"},
                "provenance": {"image_id": "sha256:original"},
            }
            guard(original)
            with self.assertRaisesRegex(ValueError, "environment changed"):
                guard(
                    {"manifest": {"prepare.py": "different"}, "provenance": original["provenance"]}
                )
            with self.assertRaisesRegex(ValueError, "environment changed"):
                guard(
                    {
                        "manifest": original["manifest"],
                        "provenance": {"image_id": "sha256:different"},
                    }
                )
            with self.assertRaisesRegex(ValueError, "seed panel changed"):
                StudyGuard(guard.path, [{"seed": 1}], resume=True)

    def test_demo_dry_run_has_no_filesystem_side_effects(self):
        from evolution_elite_table.cli import main

        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "absent"
            with redirect_stdout(StringIO()):
                main(["study", "--demo", "--output", str(output), "--dry-run"])
            self.assertFalse(output.exists())

    def test_baseline_command_needs_no_api_key_or_model(self):
        from evolution_elite_table.cli import main

        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            (root / "train.py").write_text(
                'print("val_bpb: 1\\ntraining_seconds: 0\\npeak_vram_mb: 0")'
            )
            config = {
                "benchmark": ".",
                "files": ["train.py"],
                "adapter": "generic",
                "backend": "local",
                "training_seconds": 0,
                "timeout_seconds": 10,
                "command": [sys.executable, "train.py"],
                "provider": {"kind": "openrouter", "model": "REPLACE_WITH_MODEL_ID"},
            }
            path = root / "config.json"
            path.write_text(json.dumps(config))
            with patch.dict("os.environ", {}, clear=True), redirect_stdout(StringIO()):
                main(["baseline", str(path), "--output", str(root / "baseline")])
            result = json.loads((root / "baseline/summary.json").read_text())
            self.assertEqual(result["evaluation_attempts"], 1)
            self.assertEqual(result["proposal_calls"], 0)
            self.assertFalse(result["demo"])


if __name__ == "__main__":
    unittest.main()
