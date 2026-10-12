# NVIDIA GPU setup

Run these steps on a Linux NVIDIA machine. This repository's CPU demo works on macOS; the original autoresearch benchmark requires CUDA. The GPU path has not been executed on the development Mac.

Install the engine on the server from this project directory (Python 3.11+ and Git are required):

```sh
uv venv
uv pip install -e .
```

Clone [the public repository](https://github.com/Endogen/evolution-elite-table) or transfer a source archive; create a new environment on the server instead of copying an existing `.venv`. The [README](../README.md#nvidia-setup-and-first-search) contains the complete installation walkthrough.

## 1. Prepare the benchmark and image

Install Docker with NVIDIA Container Toolkit and verify `docker run --rm --gpus all ubuntu:22.04 nvidia-smi` works. Clone the benchmark to a separate directory. This revision was inspected during implementation:

```sh
git clone https://github.com/karpathy/autoresearch.git /path/to/autoresearch
git -C /path/to/autoresearch checkout 228791fb499afffb54b46200aca536f79142f117

docker build \
  -f /path/to/evolution-elite-table/Dockerfile.autoresearch \
  -t eet-autoresearch:local /path/to/autoresearch
```

The image installs the benchmark's frozen dependency lock. Each run records and uses the resolved image ID, rather than following a moving tag. Keep the same image across comparisons. A rebuild may resolve a different base image, so it is a new experimental environment.

Prepare data and kernel caches once with network access:

```sh
mkdir -p "$HOME/.cache/autoresearch" "$HOME/.cache/eet-huggingface"

docker run --rm --gpus 'device=0' \
  --mount type=bind,src=/path/to/autoresearch,dst=/bench,readonly \
  --mount "type=bind,src=$HOME/.cache/autoresearch,dst=/root/.cache/autoresearch" \
  eet-autoresearch:local python /bench/prepare.py

docker run --rm --gpus 'device=0' \
  --mount "type=bind,src=$HOME/.cache/eet-huggingface,dst=/root/.cache/huggingface" \
  eet-autoresearch:local python -c \
  'import torch; from kernels import get_kernel; get_kernel("varunneal/flash-attention-3" if torch.cuda.get_device_capability() == (9, 0) else "kernels-community/flash-attn3")'
```

The second command warms the actual hardware-specific kernel before networking is disabled. Generated experiments cannot download packages, data or kernels. Keep preparation caches unchanged after the run starts; their contents are hashed on startup/resume. Large datasets make that check take time. `HF_HUB_OFFLINE=1` is set inside experiment containers.

## 2. Verify the NVIDIA environment without API calls

Copy `configs/autoresearch.example.json` to `configs/my-autoresearch.json`. Set `benchmark`, the cache `mounts`, and `gpu` (one device index or UUID). You can leave the model placeholder and API key unset for these commands:

```sh
.venv/bin/eet check configs/my-autoresearch.json --output runs/gpu-check
.venv/bin/eet baseline configs/my-autoresearch.json --output runs/gpu-baseline
```

Use fresh output directories. `check` verifies the image's dependency hashes against the benchmark, GPU identity, the prepared caches, PyTorch CUDA access, a compiled CUDA operation, the offline attention kernel, tokenizer loading and one batch from each search split. It saves `check.json` and `check.log`. It does not train the full model or call a proposal provider.

`baseline` executes the unmodified training program for the full frozen budget, followed by validation. It uses exactly one evaluation slot and zero proposal calls. Inspect `runs/gpu-baseline/REPORT.md` and `candidates/00001/evaluation.log`. Full training output is in `candidates/00001/evaluation.training.log`, including when the container times out. The baseline does not become an existing search checkpoint: start a fresh run for the actual search so every variant pays the same baseline cost.

If the baseline is too large for the GPU, adjust `DEPTH` and `DEVICE_BATCH_SIZE` in the upstream `train.py` before starting a new run. Keep the chosen baseline identical across all variants. `training_seconds` must equal `TIME_BUDGET` in frozen `prepare.py`; lowering only the configuration is rejected before training. The environment check is not a substitute for the full baseline's memory and timing check.

## 3. Configure proposals

Set `provider.model` in `configs/my-autoresearch.json` to the chosen OpenRouter model ID. Choose the same model for every comparison. Set the key without writing it to configuration or source:

```sh
export OPENROUTER_API_KEY=your-key
```

OpenRouter proposals send training source and benchmark context to the selected provider. The engine makes no calls during installation or dry runs. Cost is taken from reported usage; missing billing stays unknown. It does not estimate unknown charges or apply a dollar spending cap. `max_proposals` and `max_tokens` bound request volume and output size. No automatic HTTP retry is performed after an ambiguous error.

Alternatively, replace `provider` with a command provider:

```json
{
  "kind": "command",
  "command": ["/absolute/path/to/your-proposal-adapter"],
  "timeout_seconds": 180
}
```

The executable receives one JSON object on stdin containing `prompt`, `operation`, `candidate_id`, `proposal_seed`, `baseline`, `parents`, `archive`, and benchmark context. It returns one JSON object on stdout:

```json
{
  "source": "complete Python training source",
  "description": "what this candidate changes and why",
  "usage": {"cost_usd": 0.02, "input_tokens": 5000, "output_tokens": 7000}
}
```

Usage is optional; omitted cost is unknown. Diagnostics go to stderr. Commands run on the host, with your environment, and must be trusted. Arguments are passed directly without a shell. Multiple proposal calls may be in flight, so adapters must support separate processes.

## 4. Launch and inspect

```sh
.venv/bin/eet run configs/my-autoresearch.json --output runs/nvidia-0 --dry-run
.venv/bin/eet run configs/my-autoresearch.json --output runs/nvidia-0
```

Each new run checks image availability, dependency-lock hashes, selected GPU identity and mount contents, then executes the unmodified baseline. Resume validates the frozen environment before further execution and does not repeat completed evaluations, including the baseline. A failed baseline stops the search before any proposal calls. Inspect `candidates/00001/evaluation.log` and `evaluation.training.log`, correct the environment, and start a fresh run.

All candidate training/evaluation happens serially on `gpu` (default `0`). Avoid competing workloads on that GPU. The default 30 evaluation slots reserve up to 2.5 training hours, plus startup/evaluation overhead; each attempt has a ten-minute wall limit. GPU memory in parsed results (including resource-limit rejections) is PyTorch's reported peak allocated memory, not a hard CUDA allocation quota. `max_memory_mb` rejects results over a threshold after execution; timeouts and the GPU's physical limit still apply.

## 5. Supply a real untouched test protocol

Before search, add `heldout_evaluator.py` to the benchmark and to `files`. It must provide:

```python
def evaluate_test_bpb(model, tokenizer, batch_size):
    # Load independently reserved test examples from /test-data.
    # Use the same tokenization, context-length and bits-per-byte accounting
    # as prepare.evaluate_bpb. Return one finite, nonnegative float.
    ...
```

The project deliberately supplies no fake test shard or invented scientific test set. The evaluator implementation depends on your reserved dataset and must be reviewed and tested before search. Set:

```json
{
  "heldout_command": ["python", "/bench/_eet_worker.py", "--heldout"],
  "heldout_mounts": [{"source": "/absolute/path/to/reserved-test-data", "target": "/test-data"}]
}
```

Do not also include the test data in a normal search mount. The test evaluator is frozen in the benchmark snapshot; the model sees only training source and search metrics. The test-data mount is added only for the sealed finalization. Data independence is a protocol responsibility: the engine cannot prove that externally supplied datasets do not overlap.

After the search, run `.venv/bin/eet finalize runs/nvidia-0`. It retrains the selected source with the same training protocol, then evaluates test BPB once. The final result and extra wall time are in `summary.json` under `heldout`. A new independent search requires a new output directory and should not be designed using this test result.
