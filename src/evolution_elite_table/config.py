import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

MODES = ("sequential", "memory", "edit", "remix")


@dataclass
class Config:
    benchmark: str
    files: list[str] = field(
        default_factory=lambda: ["train.py", "prepare.py", "pyproject.toml", "uv.lock"]
    )
    mode: str = "remix"
    seed: int = 0
    evaluations: int = 30
    max_proposals: int = 120
    population: int = 10
    elite_size: int = 5
    proposal_workers: int = 3
    max_repairs: int = 2
    diversity_weight: float = 0.0
    training_seconds: float = 300
    timeout_seconds: float = 600
    training_tolerance_seconds: float = 30
    max_memory_mb: float | None = None
    adapter: str = "autoresearch"
    backend: str = "docker"
    command: list[str] = field(default_factory=lambda: ["python", "/bench/_eet_worker.py"])
    image: str = "eet-autoresearch:local"
    gpu: str = "0"
    mounts: list[dict] = field(default_factory=list)
    heldout_command: list[str] = field(default_factory=list)
    heldout_mounts: list[dict] = field(default_factory=list)
    provider: dict = field(
        default_factory=lambda: {"kind": "openrouter", "model": "REPLACE_WITH_MODEL_ID"}
    )
    context: str = "Optimize validation bits per byte under the fixed training budget."

    def validate(self):
        if not isinstance(self.benchmark, str) or not self.benchmark.strip():
            raise ValueError("benchmark must be a directory path")
        if not isinstance(self.context, str):
            raise ValueError("context must be a string")
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        for name in (
            "evaluations",
            "max_proposals",
            "population",
            "elite_size",
            "proposal_workers",
        ):
            _integer(name, getattr(self, name), minimum=1)
        _integer("seed", self.seed)
        _integer("max_repairs", self.max_repairs)
        for name in (
            "training_seconds",
            "timeout_seconds",
            "training_tolerance_seconds",
            "diversity_weight",
        ):
            _number(name, getattr(self, name))
        if self.timeout_seconds <= self.training_seconds:
            raise ValueError("timeout_seconds must exceed training_seconds")
        if self.backend not in ("local", "docker") or self.adapter not in (
            "generic",
            "autoresearch",
        ):
            raise ValueError("unknown backend or adapter")
        if self.adapter == "autoresearch" and self.backend != "docker":
            raise ValueError("the autoresearch adapter requires Docker")
        if self.max_memory_mb is not None:
            _number("max_memory_mb", self.max_memory_mb, positive=True)
        _argv("command", self.command)
        _argv("heldout_command", self.heldout_command, optional=True)
        if not isinstance(self.files, list) or not all(isinstance(x, str) for x in self.files):
            raise ValueError("files must be a list of relative paths")
        if "train.py" not in self.files:
            raise ValueError("files must include train.py")
        for name in self.files:
            p = Path(name)
            if (
                not name
                or p.as_posix() != name
                or p.is_absolute()
                or ".." in p.parts
                or name.startswith("_eet_")
            ):
                raise ValueError(f"invalid benchmark file: {name}")
        if len(set(self.files)) != len(self.files):
            raise ValueError("duplicate benchmark files")
        if self.adapter == "autoresearch":
            if "prepare.py" not in self.files:
                raise ValueError("the autoresearch adapter requires prepare.py")
            if self.command != ["python", "/bench/_eet_worker.py"]:
                raise ValueError("the autoresearch adapter must use its frozen worker command")
            if self.heldout_command and self.heldout_command != [
                "python",
                "/bench/_eet_worker.py",
                "--heldout",
            ]:
                raise ValueError("use the frozen worker --heldout command or the generic adapter")
            if self.heldout_command and "heldout_evaluator.py" not in self.files:
                raise ValueError("files must include heldout_evaluator.py before finalization")
        for name in ("image", "gpu"):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or not value
                or any(c in value for c in ("\n", "\0", ","))
            ):
                raise ValueError(f"{name} must be one nonempty identifier")
        if self.gpu == "all":
            raise ValueError("gpu must select one device, not all devices")
        targets = []
        for mounts in (self.mounts, self.heldout_mounts):
            if not isinstance(mounts, list):
                raise ValueError("mounts must be lists")
            for mount in mounts:
                if not isinstance(mount, dict) or set(mount) != {"source", "target"}:
                    raise ValueError("mounts require source and target")
                if not all(
                    isinstance(x, str) and x and not any(c in x for c in (",", "\n", "\0"))
                    for x in mount.values()
                ):
                    raise ValueError("invalid bind-mount path")
                target = Path(mount["target"])
                if (
                    not target.is_absolute()
                    or target.as_posix() != mount["target"]
                    or ".." in target.parts
                ):
                    raise ValueError("mount target must be a normalized absolute path")
                for reserved in ("/bench", "/work", "/logs", "/tmp", "/proc", "/sys", "/dev"):
                    if _overlaps(target, Path(reserved)):
                        raise ValueError(
                            "mount must not shadow benchmark, system or temporary directories"
                        )
                if any(_overlaps(target, other) for other in targets):
                    raise ValueError("mount targets must not overlap")
                targets.append(target)
        search_sources = [Path(m["source"]).expanduser().resolve() for m in self.mounts]
        search_sources.append(Path(self.benchmark).expanduser().resolve())
        for mount in self.heldout_mounts:
            heldout = Path(mount["source"]).expanduser().resolve()
            if any(_overlaps(heldout, source) for source in search_sources):
                raise ValueError(
                    "held-out data must not overlap search mounts or the benchmark directory"
                )
        if not isinstance(self.provider, dict):
            raise ValueError("provider must be an object")
        kind = self.provider.get("kind")
        allowed = {
            "none": {"kind"},
            "demo": {"kind"},
            "command": {"kind", "command", "timeout_seconds"},
            "openrouter": {
                "kind",
                "model",
                "api_key_env",
                "temperature",
                "max_tokens",
                "timeout_seconds",
            },
        }
        if kind not in allowed:
            raise ValueError("unknown provider kind")
        if set(self.provider) - allowed[kind]:
            raise ValueError(
                "unknown provider setting: " + ", ".join(sorted(set(self.provider) - allowed[kind]))
            )
        if kind == "none" and self.evaluations != 1:
            raise ValueError("provider none is only valid for a single baseline evaluation")
        if kind == "command":
            _argv("provider.command", self.provider.get("command"))
        if kind == "openrouter":
            model = self.provider.get("model")
            if not isinstance(model, str) or not model.strip() or "REPLACE" in model:
                raise ValueError("set provider.model to an explicit OpenRouter model ID")
            _integer("provider.max_tokens", self.provider.get("max_tokens", 16000), minimum=1)
            temperature = self.provider.get("temperature", 0.8)
            _number("provider.temperature", temperature)
            if temperature > 2:
                raise ValueError("provider.temperature must be between 0 and 2")
            key_env = self.provider.get("api_key_env", "OPENROUTER_API_KEY")
            if not isinstance(key_env, str) or not key_env or "=" in key_env or "\0" in key_env:
                raise ValueError("provider.api_key_env must be an environment variable name")
        if kind in ("openrouter", "command"):
            _number(
                "provider.timeout_seconds", self.provider.get("timeout_seconds", 180), positive=True
            )
        return self

    def to_dict(self):
        return asdict(self)

    @classmethod
    def load(cls, path, *, baseline_only=False):
        path = Path(path).resolve()
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            raise ValueError("configuration must be a JSON object")
        if baseline_only:
            data.update(evaluations=1, provider={"kind": "none"})
        try:
            config = cls(**data)
        except TypeError as exc:
            raise ValueError(f"invalid configuration fields: {exc}") from exc
        # Validate types before interpreting paths; validate overlap after resolution.
        if isinstance(config.benchmark, str):
            config.benchmark = str((path.parent / Path(config.benchmark).expanduser()).resolve())
        for mounts in (config.mounts, config.heldout_mounts):
            if isinstance(mounts, list):
                for mount in mounts:
                    if isinstance(mount, dict) and isinstance(mount.get("source"), str):
                        mount["source"] = str(
                            (path.parent / Path(mount["source"]).expanduser()).resolve()
                        )
        return config.validate()


def _integer(name, value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _number(name, value, positive=False):
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or value < 0
        or (positive and value == 0)
    ):
        raise ValueError(
            f"{name} must be a finite {'positive' if positive else 'nonnegative'} number"
        )


def _argv(name, value, optional=False):
    if (
        not isinstance(value, list)
        or (not value and not optional)
        or not all(isinstance(x, str) and x and "\0" not in x for x in value)
    ):
        raise ValueError(f"{name} must be {'an' if optional else 'a nonempty'} argv list")


def _overlaps(a, b):
    return a.is_relative_to(b) or b.is_relative_to(a)
