import ast
import json
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

from .process import run_process, tail
from .programs import parse_metrics
from .storage import mount_identity


class Runner:
    def __init__(self, config, store, state):
        self.config, self.store, self.state = config, store, state

    def preflight(self):
        self.store.verify(self.state)
        cfg = self.config
        if cfg.adapter == "autoresearch":
            snapshot = self.store.root / "benchmark"
            from .programs import validate

            baseline = (snapshot / "train.py").read_text()
            validate(baseline, baseline, "autoresearch")
            constants = {}
            for node in ast.parse((snapshot / "prepare.py").read_text()).body:
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id == "TIME_BUDGET":
                            constants[target.id] = ast.literal_eval(node.value)
            if constants.get("TIME_BUDGET") != cfg.training_seconds:
                raise ValueError("training_seconds must match the frozen prepare.py TIME_BUDGET")
        if cfg.backend == "docker":
            result = subprocess.run(
                ["docker", "image", "inspect", cfg.image],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            if result.returncode:
                raise ValueError(
                    f"Docker image {cfg.image!r} is unavailable; build it before starting"
                )
            info = json.loads(result.stdout)[0]
            image_id = info["Id"]
            previous = self.state["provenance"].get("image_id")
            if previous and previous != image_id:
                raise ValueError("Docker image changed since this run began")
            self.state["provenance"]["image_id"] = image_id
            for mount in cfg.mounts + cfg.heldout_mounts:
                if not Path(mount["source"]).exists():
                    raise ValueError(f"mount source does not exist: {mount['source']}")
            data_identity = mount_identity(cfg.mounts + cfg.heldout_mounts)
            previous = self.state["provenance"].get("mounted_data")
            if previous is not None and previous != data_identity:
                raise ValueError("mounted dataset or kernel cache changed since the run began")
            self.state["provenance"]["mounted_data"] = data_identity
            # Fail before spending a model call if the selected GPU is unavailable.
            probe_script = (
                (
                    "import hashlib,json,pathlib,subprocess; "
                    "gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,memory.total,driver_version','--format=csv,noheader'],text=True); "
                    "print(json.dumps({'gpu':gpu,'dependencies':{name:hashlib.sha256((pathlib.Path('/opt/autoresearch')/name).read_bytes()).hexdigest() "
                    "for name in ['pyproject.toml','uv.lock']}}))"
                )
                if cfg.adapter == "autoresearch"
                else (
                    "import json,subprocess; print(json.dumps({'gpu':subprocess.check_output("
                    "['nvidia-smi','--query-gpu=uuid,name,memory.total,driver_version','--format=csv,noheader'],text=True)}))"
                )
            )
            name = "eet-check-" + uuid.uuid4().hex
            try:
                code, stdout, stderr, _, _ = run_process(
                    [
                        "docker",
                        "run",
                        "--rm",
                        "--name",
                        name,
                        "--network=none",
                        "--gpus",
                        f"device={cfg.gpu}",
                        image_id,
                        "python",
                        "-c",
                        probe_script,
                    ],
                    timeout=60,
                )
            finally:
                self._remove_container(name)
            if code:
                raise ValueError("Docker cannot access the selected NVIDIA GPU: " + stderr[-1000:])
            diagnostic = json.loads(stdout)
            for filename, checksum in diagnostic.get("dependencies", {}).items():
                if self.state["manifest"].get(filename) != checksum:
                    raise ValueError(
                        f"Docker image dependencies differ from frozen benchmark: {filename}"
                    )
            hardware = diagnostic["gpu"].strip()
            if not hardware or len(hardware.splitlines()) != 1:
                raise ValueError("the experiment requires exactly one visible NVIDIA GPU")
            previous = self.state["provenance"].get("gpu_hardware")
            if previous and previous != hardware:
                raise ValueError("GPU identity or driver changed since the run began")
            self.state["provenance"]["gpu_hardware"] = hardware
        else:
            executable = cfg.command[0]
            if not shutil.which(executable):
                raise ValueError(f"training executable not found: {executable}")
        self.store.save(self.state)

    def probe(self):
        self.preflight()
        if self.config.adapter != "autoresearch":
            return {"status": "ready", "backend": self.config.backend}
        source = (self.store.root / "benchmark" / "train.py").read_text()
        log = self.store.root / "check.log"
        result = self._execute(
            {"source": source}, ["python", "/bench/_eet_worker.py", "--check"], log
        )
        if result["returncode"]:
            raise ValueError("CUDA environment check failed: " + tail(log))
        payloads = [
            line.removeprefix("eet_check: ")
            for line in log.read_text().splitlines()
            if line.startswith("eet_check: ")
        ]
        if len(payloads) != 1:
            raise ValueError("CUDA check did not return one diagnostic result; inspect check.log")
        details = json.loads(payloads[0])
        for name, value in details["dependency_hashes"].items():
            if self.state["manifest"].get(name) != value:
                raise ValueError(f"Docker image dependencies do not match the benchmark: {name}")
        self.state["provenance"]["cuda_check"] = details
        self.store.save(self.state)
        return {"status": "ready", "wall_seconds": result["wall_seconds"], **details}

    @staticmethod
    def _remove_container(name):
        subprocess.run(
            ["docker", "rm", "--force", name], capture_output=True, timeout=30, check=False
        )

    def _execute(self, row, command, log, heldout=False):
        cfg = self.config
        self.store.verify(self.state)
        name = "eet-" + uuid.uuid4().hex
        with tempfile.TemporaryDirectory(prefix="eet-") as temporary:
            work = Path(temporary)
            snapshot = self.store.root / "benchmark"
            if cfg.backend == "local":
                # The local backend is for trusted fixtures only; it is not a sandbox.
                for filename in self.state["manifest"]:
                    dest = work / filename
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(snapshot / filename, dest)
                (work / "train.py").write_text(row["source"])
                argv, cwd = command, work
                env = {
                    key: os.environ[key]
                    for key in ("PATH", "LANG", "SYSTEMROOT")
                    if key in os.environ
                }
                env.update(HOME=str(work), PYTHONDONTWRITEBYTECODE="1", EET_SEED=str(cfg.seed))
            else:
                candidate = work / "train.py"
                logs = work / "logs"
                logs.mkdir()
                # Recreate from the checksum-verified checkpoint, never trust an edited artifact.
                candidate.write_text(row["source"])
                argv = [
                    "docker",
                    "run",
                    "--rm",
                    "--name",
                    name,
                    "--network=none",
                    "--read-only",
                    "--cap-drop=ALL",
                    "--security-opt=no-new-privileges",
                    "--pids-limit=256",
                    "--gpus",
                    f"device={cfg.gpu}",
                    "--ipc=private",
                    "--shm-size=2g",
                    "--tmpfs",
                    "/tmp:rw,exec,nosuid,size=8g",
                    "--tmpfs",
                    "/work:rw,exec,nosuid,size=8g",
                    "--workdir",
                    "/work",
                    "--env",
                    "HOME=/root",
                    "--env",
                    "HF_HUB_OFFLINE=1",
                    "--env",
                    "PYTHONDONTWRITEBYTECODE=1",
                    "--env",
                    "CUDA_CACHE_PATH=/tmp/cuda",
                    "--env",
                    "PYTORCH_ALLOC_CONF=expandable_segments:True",
                    "--env",
                    "TRITON_CACHE_DIR=/tmp/triton",
                    "--env",
                    "TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor",
                    "--env",
                    f"EET_SEED={cfg.seed}",
                    "--env",
                    "PYTHONPATH=/bench",
                    "--mount",
                    f"type=bind,src={snapshot},dst=/bench,readonly",
                    "--mount",
                    f"type=bind,src={candidate},dst=/bench/train.py,readonly",
                    "--mount",
                    f"type=bind,src={logs},dst=/logs",
                ]
                for mount in cfg.mounts + (cfg.heldout_mounts if heldout else []):
                    argv += [
                        "--mount",
                        f"type=bind,src={mount['source']},dst={mount['target']},readonly",
                    ]
                argv += [self.state["provenance"]["image_id"], *command]
                cwd, env = None, None
            try:
                code, _, _, elapsed, timed_out = run_process(
                    argv, cwd=cwd, env=env, timeout=cfg.timeout_seconds, log=log
                )
            finally:
                if cfg.backend == "docker":
                    # Killing the client does not necessarily stop its container.
                    try:
                        self._remove_container(name)
                    finally:
                        training_log = logs / "training.log"
                        if training_log.is_file() and not training_log.is_symlink():
                            shutil.copyfile(training_log, log.with_suffix(".training.log"))
            if cfg.backend == "docker" and code == 125:
                raise RuntimeError("Docker could not launch the evaluation container: " + tail(log))
            return {
                "wall_seconds": elapsed,
                "log": str(log.relative_to(self.store.root)),
                "returncode": code,
                "timed_out": timed_out,
            }

    def evaluate(self, row, heldout=False):
        cfg = self.config
        log = self.store.candidate_dir(row["id"]) / ("heldout.log" if heldout else "evaluation.log")
        command = cfg.heldout_command if heldout else cfg.command
        if not command:
            raise ValueError("no held-out evaluation command configured")
        result = self._execute(row, command, log, heldout)
        if result["timed_out"] or result["returncode"]:
            error = "wall-clock timeout" if result["timed_out"] else f"exit {result['returncode']}"
            return dict(result, status="failed", error=error + "\n" + tail(log))
        try:
            metrics = parse_metrics(
                log.read_text(errors="replace"), "test_bpb" if heldout else "val_bpb"
            )
        except ValueError as exc:
            return dict(result, status="failed", error=str(exc))
        # Keep measured resources even when duration or memory rejects a candidate.
        result["metrics"] = metrics
        duration = metrics["training_seconds"]
        if (
            duration < cfg.training_seconds - 0.5
            or duration > cfg.training_seconds + cfg.training_tolerance_seconds
        ):
            return dict(
                result,
                status="failed",
                error=f"training duration {duration} violates the configured budget",
            )
        if cfg.max_memory_mb is not None and metrics["peak_vram_mb"] > cfg.max_memory_mb:
            return dict(
                result, status="failed", error="peak GPU memory exceeds the configured limit"
            )
        return dict(result, status="success")
