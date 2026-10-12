"""A repeated study pins configuration and environment before evaluating candidates."""

import json

from .engine import Engine
from .report import study_summary, write_report
from .storage import Store, atomic_json


def environment_identity(state):
    provenance = state["provenance"]
    return {
        "benchmark_files": state["manifest"],
        **{
            key: provenance.get(key)
            for key in ("image_id", "gpu_hardware", "mounted_data", "python", "host")
        },
    }


class StudyGuard:
    def __init__(self, path, plan, resume=False):
        self.path = path
        if path.exists():
            self.manifest = json.loads(path.read_text())
            if not resume:
                raise ValueError("study already exists; use --resume")
            if self.manifest["plan"] != plan:
                raise ValueError("study configuration or seed panel changed")
        else:
            self.manifest = {"schema": 1, "plan": plan, "environment": None}
            atomic_json(path, self.manifest)

    def __call__(self, state):
        identity = environment_identity(state)
        if self.manifest["environment"] is None:
            self.manifest["environment"] = identity
            atomic_json(self.path, self.manifest)
        elif identity != self.manifest["environment"]:
            raise ValueError(
                "study benchmark, image, GPU, mounted data, or host environment changed"
            )


def execute_study(configs, output, *, resume, finalize, run):
    output.mkdir(parents=True, exist_ok=True)
    with Store(output).lock():
        paths = [output / f"{cfg.mode}-seed-{cfg.seed}" for cfg in configs]
        if not resume and any(path.exists() for path in paths):
            raise ValueError("a study run already exists; use --resume")
        # Legacy studies may be adopted only if all existing snapshots agree.
        guard = StudyGuard(
            output / "study-manifest.json", [cfg.to_dict() for cfg in configs], resume
        )
        for path in paths:
            if path.exists():
                existing = Store(path).load()
                if existing["provenance"].get("image_id") or existing["candidates"]:
                    guard(existing)
        records = []
        for cfg, folder in zip(configs, paths):
            engine = Engine.resume(folder) if folder.exists() else Engine.create(cfg, folder)
            if engine.config.to_dict() != cfg.to_dict():
                raise ValueError("study configuration differs from saved run")
            if engine.state["status"] == "sealed":
                with engine.store.lock():
                    state = engine.store.load()
                    guard(state)
                    result = write_report(engine.store, state)
            else:
                result = run(engine, finalize, environment_guard=guard)
            records.append(result)
            atomic_json(output / "study.json", study_summary(records))
    return records
