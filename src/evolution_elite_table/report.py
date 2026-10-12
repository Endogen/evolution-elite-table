import csv
import json
import math
import statistics
from collections import Counter

from .storage import atomic_json


def summary(state):
    rows = state["candidates"]
    winner = min(
        (c for c in rows if c["status"] == "success"),
        key=lambda c: (c["metrics"]["val_bpb"], len(c["source"]), c["id"]),
        default=None,
    )
    calls = [c for c in rows if c["operation"] != "baseline" and c["status"] != "queued"]
    known_costs = [c.get("usage", {}).get("cost_usd") for c in calls]
    cfg = state["config"]
    return {
        "mode": cfg["mode"],
        "seed": cfg["seed"],
        "status": state["status"],
        "best_candidate": winner["id"] if winner else None,
        "best_val_bpb": winner["metrics"]["val_bpb"] if winner else None,
        "baseline_val_bpb": rows[0].get("metrics", {}).get("val_bpb") if rows else None,
        "evaluation_attempts": state["evaluation_count"],
        "evaluation_limit": cfg["evaluations"],
        "proposal_calls": state["proposal_count"],
        "archive_size": len(state["archive"]),
        "reserved_training_hours": state["evaluation_count"] * cfg["training_seconds"] / 3600,
        "observed_evaluation_wall_hours": sum(c.get("wall_seconds", 0) for c in rows) / 3600,
        "evaluation_time_complete": not any(
            c["status"] in ("interrupted", "evaluating") for c in rows
        ),
        "known_api_cost_usd": math.fsum(v for v in known_costs if v is not None),
        "unknown_cost_calls": sum(v is None for v in known_costs),
        "peak_vram_mb": max((c.get("metrics", {}).get("peak_vram_mb", 0) for c in rows), default=0),
        "outcomes": dict(Counter(c["status"] for c in rows)),
        "heldout": state["heldout"],
        "demo": cfg["provider"]["kind"] == "demo",
    }


def write_report(store, state):
    result = summary(state)
    atomic_json(store.root / "summary.json", result)
    with (store.root / "results.tsv").open("w", newline="") as handle:
        columns = [
            "id",
            "generation",
            "operation",
            "parents",
            "status",
            "val_bpb",
            "training_seconds",
            "peak_vram_mb",
            "wall_seconds",
            "cost_usd",
            "commit",
            "description",
        ]
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in state["candidates"]:
            writer.writerow(
                dict(
                    row,
                    parents=",".join(map(str, row["parents"])),
                    cost_usd=row.get("usage", {}).get("cost_usd"),
                    **row.get("metrics", {}),
                )
            )
    if result["best_candidate"] is not None:
        winner = state["candidates"][result["best_candidate"] - 1]
        (store.root / "best_train.py").write_text(winner["source"])
    lines = [
        "# Evolutionary autoresearch run",
        "",
        "CPU demo with scripted proposals." if result["demo"] else "Training-program evaluation.",
        "",
        f"- Search: {result['mode']}; seed {result['seed']}; status {result['status']}",
        f"- Baseline val_bpb: {result['baseline_val_bpb']}",
        f"- Best val_bpb: {result['best_val_bpb']}",
        f"- Evaluation attempts: {result['evaluation_attempts']} / {result['evaluation_limit']}",
        f"- Proposal calls: {result['proposal_calls']}",
        f"- Reserved training hours: {result['reserved_training_hours']:.4f}",
        f"- Observed evaluation wall hours: {result['observed_evaluation_wall_hours']:.4f}",
        f"- Known proposal cost: ${result['known_api_cost_usd']:.6f}; unknown-cost calls: {result['unknown_cost_calls']}",
        "",
        "## Held-out evaluation",
        "",
        json.dumps(result["heldout"], indent=2)
        if result["heldout"]
        else "Not run. Validation is not an untouched test result.",
        "",
        "Sources, parent IDs and Git commits are in results.tsv and state.json. Failed evaluations consume a slot. "
        "Reserved training time is an allocation, not measured GPU use; wall time includes startup and evaluation. "
        "Interrupted wall time and unknown API billing are not imputed.",
    ]
    (store.root / "REPORT.md").write_text("\n".join(lines) + "\n")
    return result


def study_summary(run_summaries):
    grouped = {}
    for item in run_summaries:
        grouped.setdefault(item["mode"], []).append(item)
    result = {"runs": run_summaries, "by_mode": {}}
    for mode, rows in grouped.items():
        values = [r["best_val_bpb"] for r in rows if r["best_val_bpb"] is not None]
        tests = [
            r["heldout"]["metrics"]["test_bpb"]
            for r in rows
            if r["heldout"] and r["heldout"]["status"] == "success"
        ]
        result["by_mode"][mode] = {
            "runs": len(rows),
            "validation_mean": statistics.mean(values) if values else None,
            "validation_stdev": statistics.stdev(values) if len(values) > 1 else None,
            "test_mean": statistics.mean(tests) if tests else None,
            "test_runs": len(tests),
            "all_evaluation_slots_used": all(
                r["evaluation_attempts"] == r["evaluation_limit"] for r in rows
            ),
        }
    result["interpretation"] = (
        "Descriptive results only. Equal slot limits are not proof of equal realized compute or superiority."
    )
    return result
