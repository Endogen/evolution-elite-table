"""Verify GPU preflight decisions and command contracts without claiming CUDA coverage."""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evolution_elite_table.config import Config
from evolution_elite_table.engine import Engine


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        bench = root / "bench"
        bench.mkdir()
        for name, source in {
            "train.py": "x=1\n# Final eval\nprint(x)\n",
            "prepare.py": "TIME_BUDGET=300\n",
            "pyproject.toml": "# dependencies",
            "uv.lock": "# lock",
        }.items():
            (bench / name).write_text(source)
        cfg = Config(benchmark=str(bench), evaluations=1, provider={"kind": "none"})
        self.engine = Engine.create(cfg, root / "run")
        self.dependencies = {
            name: self.engine.state["manifest"][name] for name in ("pyproject.toml", "uv.lock")
        }

    def tearDown(self):
        self.tmp.cleanup()

    def docker(self, args, **kwargs):
        return subprocess.CompletedProcess(args, 0, json.dumps([{"Id": "sha256:test-image"}]), "")

    def gpu(self, *args, **kwargs):
        return (
            0,
            json.dumps(
                {"gpu": "GPU-test, Test GPU, 81920 MiB, 570", "dependencies": self.dependencies}
            ),
            "",
            0.1,
            False,
        )

    def test_image_lock_mismatch_stops_before_training(self):
        self.dependencies["uv.lock"] = "wrong-lock"
        with (
            patch("evolution_elite_table.runner.subprocess.run", side_effect=self.docker),
            patch("evolution_elite_table.runner.run_process", side_effect=self.gpu),
        ):
            with self.assertRaisesRegex(ValueError, "dependencies differ"):
                self.engine.runner.preflight()
        self.assertEqual(self.engine.state["evaluation_count"], 0)

    def test_prepare_budget_mismatch_stops_before_docker(self):
        self.engine.config.training_seconds = 20
        with patch("evolution_elite_table.runner.subprocess.run") as docker:
            with self.assertRaisesRegex(ValueError, "TIME_BUDGET"):
                self.engine.runner.preflight()
            docker.assert_not_called()

    def test_probe_saves_and_validates_details(self):
        def execute(argv, **kwargs):
            if "log" not in kwargs:
                return self.gpu()
            Path(kwargs["log"]).write_text(
                "eet_check: "
                + json.dumps({"dependency_hashes": self.dependencies, "device": "Test GPU"})
                + "\n"
            )
            self.assertIn("--network=none", argv)
            self.assertIn("--read-only", argv)
            self.assertEqual(argv[-1], "--check")
            return 0, "", "", 0.2, False

        with (
            patch("evolution_elite_table.runner.subprocess.run", side_effect=self.docker),
            patch("evolution_elite_table.runner.run_process", side_effect=execute),
        ):
            result = self.engine.runner.probe()
        self.assertEqual(result["status"], "ready")
        self.assertEqual(self.engine.store.load()["provenance"]["cuda_check"]["device"], "Test GPU")
        self.assertEqual(self.engine.state["evaluation_count"], 0)


if __name__ == "__main__":
    unittest.main()
