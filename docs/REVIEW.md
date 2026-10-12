# Implementation review — 12 October 2026

Reviewed experiment design, search transitions, provider concurrency, recovery, resource accounting, storage, Docker execution, the frozen worker, configuration, packaging and GPU setup.

## Verified corrections

| Area | Problem | Correction and verification |
| --- | --- | --- |
| Recovery | Responses completed during training remained only in futures. Interruption lost their source and known billing and caused replacement calls. | Persist each response immediately; resume consumes it. A regression interrupts training and verifies that no replacement proposals are made and costs remain known. |
| Configuration | Strings passed as commands, invalid numbers and normalized-path aliases could escape validation. | Validate argument arrays, finite numeric types, normalized paths and provider settings before external work. Regressions originally failed and now pass. |
| Test separation | A search mount could include the held-out directory as a descendant. | Reject overlapping resolved source paths and ambiguous mount targets. This does not prove semantic dataset independence or detect every symlink-based overlap. |
| Search efficiency | Even zero diversity weighting parsed candidate ASTs repeatedly while ranking. | Direct top-K score selection when weighting is zero; cache structural features within diversity selection. A test prohibits unnecessary parsing. |
| Concurrency | Synchronous repairs could add a model call beyond the background-worker limit. | Repairs and background calls share a semaphore; a concurrent regression verifies the limit. |
| Resume budgets | A partially submitted queue could exceed the remaining proposal budget after recovered repairs. | Bound every submission batch against current remaining proposal and evaluation slots. Preserve each proposal's original archive context. |
| Scientific comparison | Different study variants could silently use changed files, data or Docker images. | A locked study manifest pins configuration, seed panel and environment identity; tests reject changed benchmark/image/seed panel. |
| Python execution | The worker's ad hoc globals dictionary did not represent the candidate's actual module. Libraries resolving annotations could inspect the wrong globals. | Execute in a registered temporary module and restore it afterward. The real worker passes a dataclass/ClassVar regression with a fake CUDA interface. |
| GPU setup | Image dependencies and configured training duration could disagree with the frozen benchmark. | Preflight compares lock hashes and TIME_BUDGET before training; negative tests confirm rejection without consuming an evaluation. |
| GPU diagnostics | A timed-out container could discard its training output. | Preserve a separate training log through a dedicated output mount. A timeout regression verifies that the last training output survives cleanup. |
| Infrastructure | Docker launch failures could be treated as bad programs and trigger paid repair proposals. | Docker launch failure stops execution as an infrastructure error. |
| Resource accounting | Duration/memory rejection discarded otherwise valid resource metrics; cost summation differed between Python versions. | Preserve measured resources for rejected candidates and use accurate floating-point summation for known costs. |
| Provider protocol | Non-object JSON and empty fenced responses could escape the provider error contract or lose billing context. | Normalize these failures as provider errors with preserved reported usage. |
| CLI/reporting | Demo dry runs created files; report writes did not consistently hold the run lock. | Dry runs avoid filesystem writes; reports use the lock and no longer write twice on successful completion. |

## NVIDIA verification additions

`eet check CONFIG --output RUN` verifies CUDA access, compilation, cached attention kernels, tokenizer/data loading and image dependency identity without API calls. `eet baseline CONFIG --output RUN` performs one full frozen benchmark evaluation without a model or API key. Both accept an unchanged provider-model placeholder. See [GPU setup](GPU_SETUP.md).

## Validation completed

- 52 automated tests pass on Python 3.11 and Python 3.12.
- Lint and formatting checks pass, including unused-import checks.
- A fresh study completed 12 CPU searches, 144 training evaluations and 12 held-out evaluations; resuming the completed study succeeded.
- The built wheel ran a separate end-to-end demo outside the source checkout.
- Source and wheel distributions were rebuilt with the revised code and instructions.

The CPU study uses scripted proposals and a small n-gram language model. Docker command construction, preflight failure handling and worker module behavior are tested with controlled substitutes. CUDA kernels, GPU memory use, real throughput and the full NVIDIA image still require validation on the target GPU. No paid model calls were made during this review.

## Remaining boundaries

The evaluator assumes cooperative generated Python; it is not an adversarially secure metric oracle. Test-dataset independence must be established externally. Equal evaluation-slot allocations do not guarantee equal realized compute or API cost. One GPU is processed serially per search; there is no cross-run GPU scheduler. Normal interruption can wait for submitted provider requests, while forced termination can leave unknown billing. These boundaries are documented rather than reported as completed capabilities.
