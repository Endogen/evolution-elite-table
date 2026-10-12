"""Syntax validation, canonical identities, and an explicitly heuristic diversity metric."""

import ast
import hashlib
import math


def canonical(source):
    tree = ast.parse(source)
    compile(tree, "train.py", "exec")
    # Comments and formatting do not buy another evaluation. Docstrings are ignored too.
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr):
            if isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
                body.pop(0)
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()


def features(source):
    result = set()
    for node in ast.walk(ast.parse(source)):
        result.add(type(node).__name__)
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            result.add(f"definition:{node.name}")
        elif isinstance(node, ast.Call):
            result.add("call:" + ast.dump(node.func, include_attributes=False))
    return result


def validate(source, baseline, adapter):
    if not isinstance(source, str) or not source.strip() or len(source.encode()) > 1_000_000:
        raise ValueError("source must be nonempty Python smaller than 1 MB")
    fingerprint = canonical(source)
    if adapter == "autoresearch":
        marker = "# Final eval\n"
        if source.count(marker) != 1 or baseline.count(marker) != 1:
            raise ValueError("preserve the upstream '# Final eval' marker exactly once")
        if canonical(source.split(marker)[1]) != canonical(baseline.split(marker)[1]):
            raise ValueError(
                "the final evaluation and summary are frozen; edit only the training prefix"
            )
    return fingerprint


def select_archive(candidates, size, diversity_weight=0.0):
    pool = sorted(
        (c for c in candidates if c["status"] == "success"),
        key=lambda c: (c["metrics"]["val_bpb"], len(c["source"]), c["id"]),
    )
    if not pool:
        return []
    if diversity_weight == 0:
        return [c["id"] for c in pool[:size]]
    structural = {c["id"]: features(c["source"]) for c in pool}

    def distance(a, b):
        left, right = structural[a["id"]], structural[b["id"]]
        return 1 - len(left & right) / max(1, len(left | right))

    # Always preserve the global validation winner. Other slots trade rank for diversity.
    selected = [pool.pop(0)]
    while pool and len(selected) < size:
        ranks = {c["id"]: i / max(1, len(pool) - 1) for i, c in enumerate(pool)}
        best = min(
            pool,
            key=lambda c: ranks[c["id"]] - diversity_weight * min(distance(c, s) for s in selected),
        )
        selected.append(best)
        pool.remove(best)
    return [c["id"] for c in selected]


def parse_metrics(text, metric="val_bpb"):
    fields = {}
    for line in text.splitlines():
        key, sep, value = line.partition(":")
        if sep and key.strip() in (metric, "training_seconds", "peak_vram_mb", "total_seconds"):
            key = key.strip()
            if key in fields:
                raise ValueError(f"duplicate metric: {key}")
            try:
                fields[key] = float(value.strip())
            except ValueError:
                raise ValueError(f"invalid metric: {key}") from None
    for key in (metric, "training_seconds", "peak_vram_mb"):
        if key not in fields or not math.isfinite(fields[key]) or fields[key] < 0:
            raise ValueError(f"missing, negative, or nonfinite metric: {key}")
    if any(not math.isfinite(x) or x < 0 for x in fields.values()):
        raise ValueError("nonfinite or negative resource metric")
    return fields
