import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evolution_elite_table.config import Config
from evolution_elite_table.engine import Engine


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.bench = self.root / "bench"
        self.bench.mkdir()
        (self.bench / "train.py").write_text("pass\n")
        self.config = Config(
            benchmark=str(self.bench),
            files=["train.py"],
            adapter="generic",
            backend="local",
            provider={"kind": "demo"},
            command=[sys.executable, "train.py"],
            training_seconds=0,
            timeout_seconds=3,
            training_tolerance_seconds=1,
        )
        self.engine = Engine.create(self.config, self.root / "run")
        self.runner = self.engine.runner
        self.config = self.engine.config

    def tearDown(self):
        self.tmp.cleanup()

    def row(self, source):
        return {"id": 1, "source": source}

    def test_real_subprocess_metrics(self):
        source = 'print("val_bpb: 1.2\\ntraining_seconds: 0.1\\npeak_vram_mb: 0")'
        result = self.runner.evaluate(self.row(source))
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["metrics"]["val_bpb"], 1.2)

    def test_timeout_and_bad_duration_fail(self):
        self.config.timeout_seconds = 0.1
        result = self.runner.evaluate(self.row("import time; time.sleep(10)"))
        self.assertEqual(result["status"], "failed")
        self.assertIn("timeout", result["error"])
        self.config.timeout_seconds = 3
        result = self.runner.evaluate(
            self.row('print("val_bpb: 1\\ntraining_seconds: 100\\npeak_vram_mb: 0")')
        )
        self.assertEqual(result["status"], "failed")
        self.assertIn("duration", result["error"])

    def test_memory_rejection_retains_measured_resources(self):
        self.config.max_memory_mb = 1
        result = self.runner.evaluate(
            self.row('print("val_bpb: 1\\ntraining_seconds: 0\\npeak_vram_mb: 2")')
        )
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["metrics"]["peak_vram_mb"], 2)

    def test_docker_infrastructure_failure_is_not_a_candidate_failure(self):
        self.config.backend = "docker"
        self.engine.state["provenance"]["image_id"] = "sha256:pinned"

        def fake(argv, **kwargs):
            Path(kwargs["log"]).write_text("Docker daemon is unavailable")
            return 125, "", "", 0.01, False

        with (
            patch("evolution_elite_table.runner.run_process", side_effect=fake),
            patch("evolution_elite_table.runner.subprocess.run"),
        ):
            with self.assertRaisesRegex(RuntimeError, "Docker could not launch"):
                self.runner.evaluate(self.row("pass"))

    def test_exit_and_missing_metric_fail(self):
        for code in ('raise RuntimeError("crash")', 'print("hello")'):
            self.assertEqual(self.runner.evaluate(self.row(code))["status"], "failed")

    def test_local_work_does_not_mutate_snapshot(self):
        source = 'from pathlib import Path; Path("train.py").write_text("changed"); print("val_bpb: 1\\ntraining_seconds: 0\\npeak_vram_mb: 0")'
        self.runner.evaluate(self.row(source))
        self.assertEqual((self.engine.store.root / "benchmark/train.py").read_text(), "pass\n")

    def test_docker_log_mount_allows_capability_dropped_root(self):
        self.config.backend = "docker"
        self.engine.state["provenance"]["image_id"] = "sha256:pinned"

        def inspect_log_permissions(argv, **kwargs):
            mount = next(arg for arg in argv if arg.endswith(",dst=/logs"))
            folder = Path(mount.removeprefix("type=bind,src=").removesuffix(",dst=/logs"))
            self.assertIn("--cap-drop=ALL", argv)
            self.assertEqual(folder.parent.stat().st_mode & 0o777, 0o700)
            # UID 0 without DAC_OVERRIDE needs ordinary write/traverse bits
            # when the host user, not container root, owns the bind mount.
            self.assertEqual(folder.stat().st_mode & 0o003, 0o003)
            Path(kwargs["log"]).write_text("val_bpb: 1\ntraining_seconds: 0\npeak_vram_mb: 0\n")
            return 0, "", "", 0.01, False

        with (
            patch("evolution_elite_table.runner.run_process", side_effect=inspect_log_permissions),
            patch("evolution_elite_table.runner.subprocess.run"),
        ):
            self.runner.evaluate(self.row("pass"))

    def test_training_log_survives_a_container_timeout(self):
        self.config.backend = "docker"
        self.engine.state["provenance"]["image_id"] = "sha256:pinned"

        def fake(argv, **kwargs):
            mount = next(arg for arg in argv if arg.endswith(",dst=/logs"))
            folder = Path(mount.removeprefix("type=bind,src=").removesuffix(",dst=/logs"))
            (folder / "training.log").write_text("last completed training step")
            Path(kwargs["log"]).write_text("")
            return -9, "", "", 0.1, True

        with (
            patch("evolution_elite_table.runner.run_process", side_effect=fake),
            patch("evolution_elite_table.runner.subprocess.run"),
        ):
            result = self.runner.evaluate(self.row("pass"))
        self.assertEqual(result["status"], "failed")
        saved = self.engine.store.root / "candidates/00001/evaluation.training.log"
        self.assertEqual(saved.read_text(), "last completed training step")

    def test_docker_command_isolated_and_cleanup_runs(self):
        self.config.backend = "docker"
        self.engine.state["provenance"]["image_id"] = "sha256:pinned"
        self.config.mounts = [{"source": str(self.bench), "target": "/data"}]
        self.config.heldout_mounts = [{"source": str(self.bench), "target": "/test-data"}]
        captured = []

        def fake(argv, **kwargs):
            captured.extend(argv)
            Path(kwargs["log"]).write_text("val_bpb: 1\ntraining_seconds: 0\npeak_vram_mb: 0\n")
            return 0, "", "", 0.01, False

        with (
            patch("evolution_elite_table.runner.run_process", side_effect=fake),
            patch("evolution_elite_table.runner.subprocess.run") as cleanup,
        ):
            self.runner.evaluate(self.row("pass"))
            self.assertEqual(cleanup.call_args.args[0][:3], ["docker", "rm", "--force"])
        self.assertIn("--network=none", captured)
        self.assertIn("--read-only", captured)
        self.assertIn("sha256:pinned", captured)
        self.assertNotIn("/test-data", " ".join(captured))
        self.assertIn("dst=/data,readonly", " ".join(captured))


if __name__ == "__main__":
    unittest.main()
