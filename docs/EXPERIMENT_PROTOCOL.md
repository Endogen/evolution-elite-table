# Comparison protocol

The question is whether population exploration and remixing discover better training programs than a sequential baseline under a controlled resource allocation. Integration alone cannot answer it.

1. Run `eet check` and `eet baseline` to verify the GPU environment without any proposal calls. Fix the upstream revision, dataset/tokenizer, image, GPU, provider/model, training duration, timeout, failure/repair limits, and evaluation-slot limit before any run.
2. Reserve an independent test dataset. Keep its evaluator fixed and its data out of search containers. Report validation separately from test results.
3. Run A (sequential), B (memory), C (branching edits), and D (branching edits plus remix) for each search seed. The built-in study uses a fixed A–D order; counterbalance order externally when temporal provider drift or GPU load is a concern.
4. Keep diversity weighting zero for the primary ablation. Study nonzero diversity as a separate extension. Preserve the 20% new-proposal rate across C/D.
5. Include every run and failure. Baseline and all execution retries consume evaluation slots. Record slots used, allocated training time, measured evaluator wall time, memory, known proposal cost and unknown-billing counts. Proposal exhaustion can leave unused slots; report this rather than presenting it as equal realized compute.
6. Preselect the endpoint by validation score and test that source once. Finalization retrains the source; it does not reuse a saved weight checkpoint. Treat this additional run as a separate endpoint cost.
7. Compare paired endpoints across several independent search seeds. Publish all seeds, variability and resource differences. Search randomness and training randomness are different: this harness retains the original training seed while varying the search seed. For training-seed robustness, explicitly freeze and repeat a separate protocol.

The included study summary gives descriptive means and sample standard deviations, plus per-run records. It does not establish significance or attribute gains to mechanisms automatically. A/B have sequential updates; C/D use generations. These are intentionally different search procedures and should be described that way.

The current implementation supports one GPU per run. Concurrent runs must be assigned distinct devices and separate output directories. There is no cluster scheduler, cross-run GPU lock, strategy-level evolution, semantic architecture classifier, or adversarially secure Python evaluator in this version.

`study-manifest.json` enforces the configuration and environment identity across variants and seeds. Resume validates existing snapshots before continuing; it refuses changed inputs rather than silently combining incomparable runs.
