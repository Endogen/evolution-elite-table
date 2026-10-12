"""Frozen NVIDIA worker. Container isolation does not prevent Python metric gaming."""

import contextlib
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import ModuleType

# Set this before torch/prepare import, as the original training script does.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")


def check():
    """Exercise CUDA, compilation, offline kernels, tokenizer and both search splits."""
    import torch
    from kernels import get_kernel
    from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access CUDA")
    capability = torch.cuda.get_device_capability()
    kernel = (
        "varunneal/flash-attention-3" if capability == (9, 0) else "kernels-community/flash-attn3"
    )
    get_kernel(kernel)
    compiled = torch.compile(lambda x: x + 1)
    compiled(torch.ones(4, device="cuda"))
    torch.cuda.synchronize()
    tokenizer = Tokenizer.from_directory()
    for split in ("train", "val"):
        batch = next(make_dataloader(tokenizer, 1, MAX_SEQ_LEN, split))
        if not batch[0].numel():
            raise RuntimeError(f"empty {split} batch")
    dependencies = {}
    for name in ("pyproject.toml", "uv.lock"):
        path = Path("/opt/autoresearch") / name
        dependencies[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    print(
        "eet_check: "
        + json.dumps(
            {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "device": torch.cuda.get_device_name(),
                "capability": capability,
                "kernel": kernel,
                "time_budget": TIME_BUDGET,
                "vocabulary_size": tokenizer.get_vocab_size(),
                "dependency_hashes": dependencies,
            }
        )
    )


def main(benchmark_dir=Path("/bench"), work_dir=Path("/logs")):
    import torch
    from prepare import TIME_BUDGET, evaluate_bpb

    heldout = "--heldout" in sys.argv
    evaluator = evaluate_bpb
    if heldout:
        from heldout_evaluator import evaluate_test_bpb

        evaluator = evaluate_test_bpb
    path = benchmark_dir / "train.py"
    candidate = path.read_text()
    marker = "# Final eval\n"
    if candidate.count(marker) != 1:
        raise ValueError("expected exactly one final evaluation marker")
    prefix = candidate.partition(marker)[0]
    # A real module matters for dataclasses, annotations, pickling and torch.compile.
    # The training script must see its own globals when libraries inspect __main__.
    module = ModuleType("__main__")
    module.__file__ = str(path)
    previous_main = sys.modules.get("__main__")
    started = time.monotonic()
    try:
        sys.modules["__main__"] = module
        with (
            (work_dir / "training.log").open("w", buffering=1) as log,
            contextlib.redirect_stdout(log),
            contextlib.redirect_stderr(log),
        ):
            exec(compile(prefix, str(path), "exec", dont_inherit=True), module.__dict__)
            namespace = module.__dict__
            training_seconds = float(namespace["total_training_time"])
            if training_seconds < TIME_BUDGET - 0.5:
                raise ValueError("candidate did not complete the fixed training budget")
            model = namespace["model"]
            model.eval()
            with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                score = evaluator(model, namespace["tokenizer"], namespace["DEVICE_BATCH_SIZE"])
            torch.cuda.synchronize()
    finally:
        if previous_main is None:
            sys.modules.pop("__main__", None)
        else:
            sys.modules["__main__"] = previous_main
    print(f"{'test_bpb' if heldout else 'val_bpb'}: {score:.8f}")
    print(f"training_seconds: {training_seconds:.3f}")
    print(f"peak_vram_mb: {torch.cuda.max_memory_allocated() / 1024**2:.3f}")
    print(f"total_seconds: {time.monotonic() - started:.3f}")


if __name__ == "__main__":
    try:
        if "--check" in sys.argv:
            check()
        else:
            main()
    except BaseException:
        path = Path("/logs/training.log")
        if path.exists():
            with path.open("rb") as log:
                log.seek(0, 2)
                log.seek(max(0, log.tell() - 6000))
                print(log.read().decode(errors="replace"))
        raise
