"""Run the actual frozen worker with small fake CUDA/evaluator modules.

Tests Python execution and metric ownership, not CUDA correctness.
"""

import importlib.util
import sys
import tempfile
import unittest
from contextlib import nullcontext, redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import evolution_elite_table
from evolution_elite_table.programs import parse_metrics


class WorkerTests(unittest.TestCase):
    def test_candidate_has_real_module_and_frozen_score(self):
        torch = SimpleNamespace(
            no_grad=nullcontext,
            bfloat16="bf16",
            amp=SimpleNamespace(autocast=lambda **kw: nullcontext()),
            cuda=SimpleNamespace(max_memory_allocated=lambda: 1024**2, synchronize=lambda: None),
        )
        prepare = SimpleNamespace(TIME_BUDGET=1, evaluate_bpb=lambda *args: 1.25)
        source = """from __future__ import annotations
from dataclasses import dataclass, fields
from typing import ClassVar
@dataclass
class Model:
    count: ClassVar[int] = 1
    def eval(self): pass
assert len(fields(Model)) == 0, "ClassVar annotation resolved against the wrong module"
model = Model()
tokenizer = None
DEVICE_BATCH_SIZE = 1
total_training_time = 1
print("val_bpb: 0.000001")
# Final eval
raise RuntimeError("candidate evaluation suffix must not execute")
"""
        path = Path(evolution_elite_table.__file__).parent / "assets/autoresearch_worker.py"
        spec = importlib.util.spec_from_file_location("eet_worker_test", path)
        worker = importlib.util.module_from_spec(spec)
        original_main = sys.modules["__main__"]
        with (
            patch.dict(sys.modules, {"torch": torch, "prepare": prepare}),
            tempfile.TemporaryDirectory() as root,
        ):
            spec.loader.exec_module(worker)
            root = Path(root)
            (root / "train.py").write_text(source)
            output = StringIO()
            with redirect_stdout(output):
                worker.main(root, root)
            self.assertEqual(parse_metrics(output.getvalue())["val_bpb"], 1.25)
            self.assertIn("0.000001", (root / "training.log").read_text())
            self.assertIs(sys.modules["__main__"], original_main)


if __name__ == "__main__":
    unittest.main()
