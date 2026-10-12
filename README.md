# Evolution Elite Table

Evolutionary search over language-model training programs, with an elite archive, multi-parent remixing, fixed evaluation budgets, and recoverable experiments.

The engine starts from `train.py`, evaluates it, asks a coding model for new programs, trains each candidate on one NVIDIA GPU, and keeps the strongest programs by validation bits per byte (`val_bpb`, lower is better). Candidate sources, parent relationships, failures, resource use, and reported API costs are saved for inspection.

This is an independent implementation inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) and [Stuart Farmer's EliteTable](https://github.com/StuartFarmer/elitetable). It implements population search and program remixing; it does not evolve the research strategy itself. Whether it outperforms sequential autoresearch must be measured.

**Validation status:** 52 automated tests passed on Python 3.11 and 3.12, alongside lint, formatting, packaging, and end-to-end CPU experiments. The NVIDIA execution path still needs a real GPU run. Start with the environment check and baseline below before spending money on proposals. See [the implementation review](docs/REVIEW.md).

## Contents

- [Requirements](#requirements)
- [NVIDIA setup and first search](#nvidia-setup-and-first-search)
- [Inspect, interrupt, and resume](#inspect-interrupt-and-resume)
- [Search variants and repeated comparisons](#search-variants-and-repeated-comparisons)
- [Independent held-out evaluation](#independent-held-out-evaluation)
- [Configuration reference](#configuration-reference)
- [Use another model provider](#use-another-model-provider)
- [Budgets and reproducibility](#budgets-and-reproducibility)
- [Troubleshooting](#troubleshooting)
- [CPU demonstration](#cpu-demonstration)
- [Development and repository layout](#development-and-repository-layout)
- [Execution boundaries](#execution-boundaries)

## Requirements

| Component | Requirement |
| --- | --- |
| GPU host | Linux with one CUDA-capable NVIDIA GPU supported by the pinned upstream PyTorch and attention kernels. |
| GPU memory | Enough for the upstream baseline. Upstream targets an H100; smaller GPUs may need a smaller model or device batch. No minimum VRAM claim has been validated here. |
| Driver | A working NVIDIA driver compatible with the container's CUDA runtime; `nvidia-smi` must work on the host. |
| Containers | Docker Engine and NVIDIA Container Toolkit. Your user must be able to run `docker` without `sudo`. |
| Host tools | Git and Python 3.11 or newer. The instructions use `uv` to install Python and the engine. |
| Storage/RAM | Space for the image, dataset/tokenizer, kernel caches, candidate sources and logs; host RAM for preprocessing and training data. Actual demand depends on the benchmark. |
| Network | Needed for installation, data/kernel preparation, and hosted proposal calls. Candidate containers run without network access. |
| Proposal model | An OpenRouter API key and explicit model ID, or a trusted command adapter for another provider. Checks and baseline need neither. |

The engine has no third-party Python runtime dependencies. PyTorch and the benchmark dependencies live in the Docker image. You do not need a separate host PyTorch installation. macOS supports the CPU demo; the NVIDIA instructions target Linux. Windows/WSL2 and multi-GPU distributed training have not been validated.

## NVIDIA setup and first search

Run these commands in **Bash on the GPU machine**. Keep using the same shell after setting the directory variables. Later `eet` commands assume the current directory is this repository. Stop and resolve a failed step before continuing.

### 1. Prepare Docker and GPU access

If the host already runs NVIDIA containers as your user, skip to the verification commands at the end of this step.

Install the NVIDIA driver for your GPU and Linux distribution using the [NVIDIA driver installation guide](https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/latest/index.html), including any required reboot. Verify:

```bash
nvidia-smi
```

For a **fresh Ubuntu 22.04/24.04 installation**, the following installs Docker Engine from its official package repository. On a host with existing Docker/container packages, follow [Docker's installation instructions](https://docs.docker.com/engine/install/ubuntu/) to resolve conflicts first. Use the corresponding official instructions for other distributions.

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg git
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
sudo tee /etc/apt/sources.list.d/docker.sources > /dev/null <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER"
```

Log out and back in so Docker group membership takes effect. This group grants administrative access to the host. Then install and configure the NVIDIA runtime using the [NVIDIA Container Toolkit instructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html):

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

These commands assume the system Docker service. Rootless Docker needs the separate configuration in NVIDIA's guide. Verify both ordinary Docker access and GPU access **without sudo**:

```bash
docker info
docker run --rm --gpus all ubuntu:22.04 nvidia-smi
nvidia-smi --query-gpu=index,uuid,name,memory.total,driver_version --format=csv
```

Choose one GPU index or UUID from the last command. The examples use device `0`. A search evaluates one candidate at a time; avoid other workloads on that device during comparisons.

### 2. Install this project

Install `uv` if needed; see [uv installation](https://docs.astral.sh/uv/getting-started/installation/) for alternatives:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

git clone https://github.com/Endogen/evolution-elite-table.git
cd evolution-elite-table
EET_ROOT="$PWD"
AUTORESEARCH_DIR="$HOME/autoresearch"
EET_GPU="0"

uv python install 3.11
uv venv --python 3.11
uv pip install -e .
.venv/bin/eet --help
```

If Python 3.11+ is already installed, you can instead create a standard virtual environment and run `.venv/bin/python -m pip install -e .`. Always create a fresh environment on the GPU server; do not copy a macOS `.venv`.

### 3. Pin the benchmark and build its image

Clone upstream into a separate directory and use the revision inspected for this adapter:

```bash
git clone https://github.com/karpathy/autoresearch.git "$AUTORESEARCH_DIR"
git -C "$AUTORESEARCH_DIR" checkout 228791fb499afffb54b46200aca536f79142f117

docker build \
  -f "$EET_ROOT/Dockerfile.autoresearch" \
  -t eet-autoresearch:local \
  "$AUTORESEARCH_DIR"
```

**The Docker build context is the upstream autoresearch directory.** The Dockerfile uses that checkout's `pyproject.toml` and `uv.lock`, installs its frozen dependencies, and supplies the runtime used by this engine. Building with this project's directory as the context is incorrect.

The engine records the resolved Docker image ID and verifies the image's dependency files against the benchmark snapshot. Keep the same image throughout a comparison. Rebuilding can change the base image and therefore the experimental environment.

### 4. Prepare data, tokenizer, and attention kernels

Preparation has network access and writes caches. Search containers later mount these caches read-only and run offline.

```bash
mkdir -p "$HOME/.cache/autoresearch" "$HOME/.cache/eet-huggingface"

docker run --rm --gpus "device=$EET_GPU" \
  --mount "type=bind,src=$AUTORESEARCH_DIR,dst=/bench,readonly" \
  --mount "type=bind,src=$HOME/.cache/autoresearch,dst=/root/.cache/autoresearch" \
  eet-autoresearch:local python /bench/prepare.py

docker run --rm --gpus "device=$EET_GPU" \
  --mount "type=bind,src=$HOME/.cache/eet-huggingface,dst=/root/.cache/huggingface" \
  eet-autoresearch:local python -c \
  'import torch; from kernels import get_kernel; get_kernel("varunneal/flash-attention-3" if torch.cuda.get_device_capability() == (9, 0) else "kernels-community/flash-attn3")'
```

The pinned `prepare.py` defaults to ten training shards plus its fixed validation shard and prepares the tokenizer. Use identical prepared data and tokenizer for every compared run. The kernel command follows the pinned upstream hardware selection; successfully downloading a kernel alone does not establish full model compatibility.

Caches must be readable by the host user running `eet`. Their contents are hashed at startup and when execution resumes, which can take time for large datasets. Finish all preparation before starting a search and leave the caches unchanged afterward. Do not store independent test data inside either search cache.

### 5. Create your configuration

Create a local configuration with the selected checkout and GPU:

```bash
.venv/bin/python - "$AUTORESEARCH_DIR" "$EET_GPU" <<'PY'
import json
import sys
from pathlib import Path

config = json.loads(Path("configs/autoresearch.example.json").read_text())
config["benchmark"] = str(Path(sys.argv[1]).expanduser().resolve())
config["gpu"] = sys.argv[2]
Path("configs/my-autoresearch.json").write_text(json.dumps(config, indent=2) + "\n")
PY
```

The cache paths in the example already match step 4. If using different locations, update each mount's `source`; preserve its container `target`. `configs/my-*.json` is ignored by Git for local settings. Never put an API key in configuration.

The template allocates **30 evaluations × 300 training seconds = 2.5 training hours**, including the baseline. Compilation, validation, model calls and other overhead add time. Each evaluation has a separate 600-second wall timeout. You can lower `evaluations` and `max_proposals` for a short initial search; leave `training_seconds` at the benchmark's fixed 300 seconds.

### 6. Check the environment and train the baseline

No API key or provider model selection is needed yet:

```bash
.venv/bin/eet check configs/my-autoresearch.json --output runs/gpu-check
.venv/bin/eet baseline configs/my-autoresearch.json --output runs/gpu-baseline
```

Use a fresh output directory for each command. `check` verifies frozen inputs, dependency hashes, GPU identity, CUDA access, a compiled CUDA operation, offline kernel loading, the tokenizer, and one training and validation batch. It writes `check.json` and `check.log`. It does not train the full model or execute an attention forward pass, so the full baseline remains necessary.

`baseline` trains the unmodified program for the full budget and validates it. It uses one evaluation and zero proposal calls. Inspect:

```bash
cat runs/gpu-baseline/REPORT.md
cat runs/gpu-baseline/candidates/00001/evaluation.log
tail -n 40 runs/gpu-baseline/candidates/00001/evaluation.training.log
```

Confirm a successful evaluation, a finite `val_bpb`, and acceptable duration and memory use. Full training output is preserved separately, including on timeout. A baseline failure stops a search before any proposal calls.

If the model runs out of GPU memory, reduce `DEVICE_BATCH_SIZE` and/or `DEPTH` in the upstream `train.py`, then use fresh check/baseline output directories. Preserve the upstream divisibility condition:

```text
TOTAL_BATCH_SIZE % (DEVICE_BATCH_SIZE * MAX_SEQ_LEN) == 0
```

Keep the chosen baseline identical across all search variants. A smaller model is a different experimental baseline; record that choice. Do not change the final-evaluation suffix. Changing only `training_seconds` cannot shorten this benchmark: preflight requires it to match `TIME_BUDGET` in the frozen `prepare.py`.

The baseline smoke run is separate from a search. Each new search pays for its own unchanged baseline evaluation.

### 7. Select a proposal model

Edit `provider.model` in `configs/my-autoresearch.json`, replacing `REPLACE_WITH_MODEL_ID` with your chosen [OpenRouter model ID](https://openrouter.ai/models). Choose a coding-capable model with sufficient context for full programs and archive entries, and support for the requested JSON response format. Use the same model for controlled comparisons. The engine calls the [OpenRouter chat-completions API](https://openrouter.ai/docs/api_reference/overview).

Enter the API key without storing it in shell history:

```bash
read -r -s -p "OpenRouter API key: " OPENROUTER_API_KEY
printf '\n'
export OPENROUTER_API_KEY
```

The engine reads the key from the host environment; `.env` files are not automatically loaded. Credentials are not passed to candidate containers. Proposal requests send training source, search context, parent programs and search diagnostics to the selected provider.

`max_proposals` limits calls, including repairs; `provider.max_tokens` limits output per request. There is **no dollar spending cap**. Use provider-side spending controls if needed. Reported usage cost is retained; missing billing remains unknown, not zero. There are no automatic HTTP retries after an ambiguous failure.

### 8. Preview and launch

```bash
.venv/bin/eet run configs/my-autoresearch.json --output runs/gpu-search --dry-run
.venv/bin/eet run configs/my-autoresearch.json --output runs/gpu-search
```

The dry run validates configuration and prints the allocation without starting Docker, training, or model calls. It is not a connectivity or credential check. The real run freezes its configuration and inputs, checks the environment, evaluates the baseline, and then begins search.

For long remote runs, use a persistent terminal session such as `tmux` so disconnecting SSH does not terminate the process. Run the engine as your ordinary Docker-enabled user, not through `sudo`.

## Inspect, interrupt, and resume

The CLI prints a JSON summary when a command finishes. Regenerate a report after stopping or completing a run:

```bash
.venv/bin/eet report runs/gpu-search
cat runs/gpu-search/REPORT.md
git --git-dir runs/gpu-search/provenance.git log --all --graph --oneline
```

| Output | Purpose |
| --- | --- |
| `REPORT.md` | Human-readable results, failures, allocations and costs. |
| `summary.json` | Machine-readable summary, including a separate held-out result when configured. |
| `results.tsv` | Candidate-level results for analysis. |
| `best_train.py` | Best successfully evaluated training source, when available. |
| `state.json` | Durable search state and lineage. |
| `state.json` → `manifest` / `provenance`, `benchmark/` | Frozen experiment identity and benchmark snapshot. |
| `provenance.git` | Candidate Git history; remix candidates have multiple parents. |
| `candidates/00001/`, etc. | Candidate sources, durable provider responses, evaluation output and training logs. |

A per-run lock prevents simultaneous writers, including report regeneration while a run holds the lock. Existing log files can be read during execution. Container training output is copied to `evaluation.training.log` when an evaluation ends; that host file is not a live stream of the active container.

Interrupt with **Ctrl+C**, then continue from saved state:

```bash
.venv/bin/eet resume runs/gpu-search
```

Resume uses the saved configuration and checks the frozen environment before more execution. Re-export the provider key in a new terminal. Editing the original configuration does not modify an existing run. Completed baseline/evaluations are not repeated; an interrupted reserved evaluation consumes its slot and is not automatically rerun. Completed provider responses are recovered without repeating their calls.

Normal shutdown may wait for already submitted proposal requests. A forced kill can leave billing unknown and may leave a container behind; inspect `docker ps --filter name=eet-` before restarting and remove only a confirmed abandoned container. Resume of a completed search does not extend its budget. A finalized run is sealed against more search.

Keep the entire output directory, original caches, Docker image and the same engine version to resume reliably. Changing hardware, drivers, data, frozen files or other recorded environment identity can prevent continuation. Start a fresh run when deliberately changing the experiment.

## Search variants and repeated comparisons

| Variant | `mode` | Behavior |
| --- | --- | --- |
| A | `sequential` | Edit the current best; retain one elite. |
| B | `memory` | Edit the current best; include the top-K archive in proposal context. |
| C | `edit` | 20% new proposals, 80% edits branching from archive members. |
| D | `remix` | 20% new, 40% edits, 40% multi-parent remixes. |

A is a controlled sequential baseline within this harness, not an exact reproduction of the original free-form agent. All variants use the same provider, repair rules and evaluator. Fractions are sampling probabilities; remix falls back to editing until two elites exist.

A/B update after each evaluation. C/D select at generation boundaries from a fixed archive. Proposals can run concurrently while training runs serially. Evaluation follows proposal ID order, so a slow early proposal can stall GPU work. Selection minimizes validation BPB, with shorter source as a tie-breaker. Optional AST-feature diversity always retains the score winner; leave it at zero for the main comparison.

Run all four variants over paired search seeds:

```bash
.venv/bin/eet study configs/my-autoresearch.json \
  --output runs/nvidia-study --seeds 0 1 2 --dry-run

.venv/bin/eet study configs/my-autoresearch.json \
  --output runs/nvidia-study --seeds 0 1 2

# Continue the same study after interruption, preserving configuration and seeds:
.venv/bin/eet study configs/my-autoresearch.json \
  --output runs/nvidia-study --seeds 0 1 2 --resume
```

With the template's 30 evaluations, this is **12 searches and 30 allocated training hours**, plus overhead. `--evaluations 4` can override the per-run limit for a short study; use the same override when resuming. All runs execute serially on the selected GPU. `study.json` contains each result and descriptive means/sample standard deviations.

`study-manifest.json` pins the full configurations, seed panel, benchmark hashes, image, GPU, mounted data and host environment. Drift is rejected before further evaluations. Search seeds control proposal choices; the upstream training RNG retains its original fixed seed. Hosted model outputs and routing need not be deterministic. Inspect failures, actual wall time and API costs before interpreting score differences. See [the comparison protocol](docs/EXPERIMENT_PROTOCOL.md).

## Independent held-out evaluation

The stock upstream benchmark supplies a validation shard, not an independent untouched test protocol. The default configuration therefore has no test evaluation. A normal search does not require one.

To measure a held-out endpoint, prepare it **before starting search**:

1. Reserve independent test data outside the benchmark directory and all search mounts.
2. Add `heldout_evaluator.py` to the upstream checkout and to the configuration's `files` list.
3. Implement and verify the following interface with the same tokenization, context length and bits-per-byte accounting as `prepare.evaluate_bpb`:

```python
def evaluate_test_bpb(model, tokenizer, batch_size):
    # Load independently reserved examples from /test-data.
    # Return one finite, nonnegative float.
    ...
```

This is an interface to implement, not a supplied working test evaluator. The project does not invent a test dataset or relabel validation as test.

4. Merge these fields into your configuration:

```json
{
  "files": ["train.py", "prepare.py", "pyproject.toml", "uv.lock", "heldout_evaluator.py"],
  "heldout_command": ["python", "/bench/_eet_worker.py", "--heldout"],
  "heldout_mounts": [
    {"source": "/absolute/path/to/reserved-test-data", "target": "/test-data"}
  ]
}
```

Do not add test data to regular `mounts`. The evaluator is frozen with the benchmark; test-data mounts are only added for finalization. The engine rejects overlapping configured source paths, but cannot prove dataset independence.

After search:

```bash
.venv/bin/eet finalize runs/gpu-search
```

Finalization selects the validation winner, records that selection, seals the run, retrains the selected source, and evaluates test BPB once. It uses **one additional training run**, separately reported under `heldout` in `summary.json`. It does not evaluate a saved weight checkpoint. Failed or interrupted finalization stays reserved and is not automatically retried. Test results never enter subsequent proposal context, and sealed runs cannot resume search.

For studies with this protocol configured, add `--finalize` to the study command and retain it on resume. Each constituent search gets one extra endpoint run. Do not use test results to redesign the same comparison.

## Configuration reference

Start from [configs/autoresearch.example.json](configs/autoresearch.example.json). Unknown top-level fields and unknown provider settings are rejected. Paths support `~`; relative benchmark and mount-source paths resolve relative to the JSON file, not your terminal directory. Commands are argument arrays, without shell expansion.

| Field | Template value | Meaning |
| --- | --- | --- |
| `benchmark` | Replace with checkout path | Directory containing the baseline and frozen benchmark files. |
| `files` | `train.py`, `prepare.py`, `pyproject.toml`, `uv.lock` | Explicit snapshot inputs; only `train.py` varies between candidates. Add the test evaluator here if used. |
| `adapter` / `backend` | `autoresearch` / `docker` | NVIDIA adapter requires Docker. `generic` / `local` is for trusted fixtures. |
| `image` | `eet-autoresearch:local` | Built benchmark image; execution uses its recorded image ID. |
| `gpu` | `"0"` | One GPU index or UUID as a string; `all` is rejected. |
| `command` | `["python", "/bench/_eet_worker.py"]` | Required frozen worker for this adapter. |
| `mounts` | Two cache mounts | Host `source` and absolute container `target`; mounted read-only during experiments. |
| `mode` | `remix` | `sequential`, `memory`, `edit`, or `remix`. |
| `seed` | `0` | Nonnegative search seed. |
| `evaluations` | `30` | Total training attempts, including baseline and runtime repairs. |
| `max_proposals` | `120` | Total provider calls, including failed proposals and repairs. |
| `population` | `10` | Generation proposal batch size in population modes. |
| `elite_size` | `5` | Archive size; sequential mode retains one. |
| `proposal_workers` | `3` | Maximum concurrent proposal calls, shared with repairs. |
| `max_repairs` | `2` | Maximum repair attempts in a candidate repair chain. |
| `diversity_weight` | `0.0` | Optional structural AST-diversity weighting. |
| `training_seconds` | `300` | Fixed training allocation; must match upstream `TIME_BUDGET`. |
| `training_tolerance_seconds` | `30` | Allowed reported training overshoot; undershoot beyond 0.5 seconds is rejected. |
| `timeout_seconds` | `600` | Per-execution wall limit, including compile/train/evaluation; must exceed training duration. |
| `max_memory_mb` | `null` | Optional rejection threshold for reported peak PyTorch allocated memory, in MiB. Not a hard allocation quota. |
| `heldout_command` / `heldout_mounts` | `[]` / `[]` | Optional frozen endpoint command and isolated test-data mounts. |
| `context` | Benchmark-specific prompt | Research objective and constraints included in proposals. |
| `provider.kind` | `openrouter` | `openrouter`, trusted `command`, scripted `demo`, or baseline-only `none`. |
| `provider.model` | Must be replaced | Explicit OpenRouter model ID; not used by other provider kinds. |
| `provider.api_key_env` | `OPENROUTER_API_KEY` | Environment variable containing the secret; not the secret itself. |
| `provider.temperature` | `0.8` | Requested sampling temperature, between 0 and 2. |
| `provider.max_tokens` | `16000` | Maximum requested completion tokens. |
| `provider.timeout_seconds` | `180` | Timeout for an individual proposal request/process. |

`check` and `baseline` automatically replace the configured provider with `none` and the evaluation count with one. They accept the template's unchanged model placeholder. A normal search requires a valid provider configuration.

## Use another model provider

Replace the entire `provider` object with a command adapter:

```json
{
  "kind": "command",
  "command": ["/absolute/path/to/your-proposal-adapter"],
  "timeout_seconds": 180
}
```

The executable reads one JSON object from stdin containing `prompt`, `operation`, `candidate_id`, `proposal_seed`, `baseline`, `parents`, `archive`, `context` and `training_seconds`. Repairs also include `diagnostic`. It must exit successfully and write exactly one JSON object to stdout:

```json
{
  "source": "complete Python training source",
  "description": "what changed and why it may improve validation BPB",
  "usage": {"cost_usd": 0.02, "input_tokens": 5000, "output_tokens": 7000}
}
```

Return the complete `train.py`, preserving the final-evaluation suffix, not a patch or a fragment. Usage is optional; omitted cost remains unknown. Send diagnostics to stderr. Adapter commands run on the host with its environment and must be trusted. Multiple processes may run concurrently, so avoid shared temporary filenames or other unsafe shared state. Paths in the command should be absolute; environment-variable references are not expanded as shell expressions.

## Budgets and reproducibility

- Every reserved evaluation consumes a slot, including the baseline, crashes, timeouts, runtime repairs and interrupted executions. Failed slots are not refunded. AST duplicates and syntax errors consume proposal calls but no training slot.
- Every proposal call, including a repair, counts toward `max_proposals`. Proposal exhaustion can end a search before all evaluation slots are used.
- Allocation is `evaluations × training_seconds`; actual wall time also includes compilation, validation, preparation and proposal waits. Equal slot allocations do not guarantee equal realized compute or API costs. Reports expose reserved training time and observed evaluator wall time separately.
- The engine stores hypotheses, parent IDs, canonical AST hashes, source commits, logs, metrics and provider usage. Formatting/comment/docstring-only changes are treated as AST duplicates.
- The frozen manifest records benchmark hashes and upstream Git revision when available, Docker image ID, GPU/driver identity and mounted-data hashes. Study manifests additionally enforce the paired experiment configuration.
- State is checkpointed before external work. Provider responses are persisted as soon as they complete, including while another candidate trains. A durable response preserves any usage returned by the provider. Requests without a durable response, and responses without billing information, retain unknown cost.
- Reports are derived from durable state. Preserve complete run directories for audit and recovery. `best_train.py` is source code, not trained model weights.

## Troubleshooting

| Symptom | What to check or change |
| --- | --- |
| `docker` permission denied | Verify group membership after logging back in and that `docker info` works as your user. Do not run the engine as root to mask the problem. |
| No GPUs in the container | Get host `nvidia-smi` working, configure NVIDIA Container Toolkit, restart Docker, then rerun the container smoke test. |
| CUDA/driver or attention-kernel error | Verify the driver supports the image runtime and the pinned kernel supports the device. Run both `check` and the full baseline. A smaller batch will not fix unsupported hardware. |
| Missing tokenizer/shards or offline kernel cache | Complete both cache-preparation commands with network access on the target GPU. Confirm configuration mount sources match those directories. |
| Image dependency mismatch | Build with the pinned upstream checkout as Docker context. Its `pyproject.toml` and `uv.lock` must match the benchmark snapshot. |
| Frozen files, cache, image or GPU changed | Restore the original environment to resume, or begin a fresh experiment with a new output directory. Do not edit manifests to bypass identity checks. |
| CUDA out of memory | Reduce upstream `DEVICE_BATCH_SIZE` and/or `DEPTH` before a fresh baseline; preserve batch divisibility. Clear competing GPU workloads. |
| Training-duration rejection | Match `training_seconds` to frozen `TIME_BUDGET`. Check whether very slow steps exceed the configured tolerance; choose a feasible fixed protocol before comparing runs. |
| Wall-clock timeout | Inspect `evaluation.log` and `evaluation.training.log`. Compilation plus validation may exceed the remaining wall budget. Increase `timeout_seconds` in a new configuration/run if appropriate; use the same value across comparisons. |
| Provider key missing | Export the configured environment variable in the shell running `eet`. `.env` is not loaded automatically. |
| Provider request or JSON failure | Check model access, balance, JSON-output support, context limits and request timeout. Increase output allowance if full source was truncated. Failures can still be billed. |
| Search ends below evaluation limit | Check exhausted proposal budget, syntax errors, repeated AST duplicates and repair failures in the report. |
| Output directory already exists | Use `resume` for that run or choose a fresh directory. A check/baseline directory is not a partially initialized search to reuse. |
| Run is locked | Stop the existing writer before `resume`, `report` or `finalize`. Do not launch two processes against the same directory. |
| No test result | Expected for the default configuration. A separate evaluator and independent data must have been configured before search. |

## CPU demonstration

After installing the engine, this needs no GPU, Docker, API key or network calls:

```bash
.venv/bin/eet demo --output runs/my-demo --evaluations 16 --finalize
.venv/bin/eet study --demo --output runs/demo-study \
  --seeds 0 1 2 --evaluations 16 --finalize
```

It trains small byte n-gram models on disjoint procedural train/validation/test sentences, with deterministic scripted proposals and real bits-per-byte scores. It demonstrates search, recovery, lineage and finalization; it does not establish NVIDIA performance or the benefit of model-generated evolutionary proposals.

## Development and repository layout

```bash
uv pip install ruff
.venv/bin/ruff check src tests examples
.venv/bin/ruff format --check src tests examples
.venv/bin/python -m unittest discover -s tests -v
uv build
```

GitHub Actions runs the checks and CPU smoke experiments on Python 3.11, 3.12 and 3.13. CUDA execution requires separate hardware. Build outputs are written to `dist/`; local environments, local configurations, caches and experiment outputs are excluded from Git.

```text
configs/                         Example JSON configurations
Dockerfile.autoresearch          Runtime built against the upstream dependency lock
src/evolution_elite_table/
  cli.py                        Command-line entry points
  config.py                     Configuration and path validation
  engine.py                     Search, proposal scheduling and repairs
  providers.py                  OpenRouter, command and scripted providers
  runner.py                     Preflight, Docker execution and metric checks
  storage.py                    Snapshots, locks, checkpoints and Git lineage
  programs.py                   Source validation, AST hashing and archive selection
  process.py                    Subprocess execution and interruption cleanup
  report.py                     Reports and best-source export
  study.py                      Paired multi-variant experiments
  assets/                       Frozen NVIDIA worker and packaged CPU benchmark
examples/tiny_lm/                 Standalone CPU fixture
tests/                          Automated regression tests
docs/                           GPU notes, experiment protocol and review findings
```

## Execution boundaries

Candidate containers use read-only benchmark/data mounts and root filesystem, no network, dropped Linux capabilities, a process limit, writable temporary space, a dedicated log-output mount, and a wall timeout. Normal interruption removes the active container. The frozen worker recomputes scores using snapshotted `prepare.py`; candidate-printed scores are not accepted as results. The source validator freezes the original final-evaluation suffix.

Generated Python remains cooperative research code. Code executing alongside the evaluator can monkey-patch objects, inspect mounted validation data, misreport timing or exploit GPU access. These controls do not provide an adversarially secure scoring oracle. Review candidates before treating results as scientific evidence. The generic local backend has no host sandbox.

The current engine handles one GPU serially per run. Concurrent independent runs need different devices and separate output directories; there is no cluster scheduler or cross-run GPU lock. Hardware load, provider routing and training nondeterminism remain practical sources of variation.
