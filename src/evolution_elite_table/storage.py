import contextlib
import fcntl
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def mount_identity(mounts):
    """Hash mounted data once per startup/resume, including relative names."""
    identities = {}
    for mount in mounts:
        root = Path(mount["source"])
        paths = sorted(p for p in root.rglob("*") if p.is_file()) if root.is_dir() else [root]
        hasher = hashlib.sha256()
        for path in paths:
            name = str(path.relative_to(root)) if root.is_dir() else root.name
            hasher.update(name.encode() + b"\0" + digest(path).encode() + b"\0")
        identities[mount["target"]] = {"sha256": hasher.hexdigest(), "files": len(paths)}
    return identities


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def git(args, *, cwd=None, data=None, git_dir=None):
    command = ["git"] + (["--git-dir", str(git_dir)] if git_dir else []) + args
    env = dict(
        os.environ,
        GIT_AUTHOR_NAME="Evolution Elite Table",
        GIT_AUTHOR_EMAIL="eet@localhost",
        GIT_COMMITTER_NAME="Evolution Elite Table",
        GIT_COMMITTER_EMAIL="eet@localhost",
    )
    return subprocess.run(
        command, cwd=cwd, input=data, text=True, capture_output=True, check=True, env=env
    ).stdout.strip()


class Store:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.path = self.root / "state.json"

    @contextlib.contextmanager
    def lock(self):
        if not self.root.is_dir():
            raise ValueError(f"run does not exist: {self.root}")
        with (self.root / ".lock").open("a") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError("another process is using this run") from None
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def create(self, config):
        source = Path(config.benchmark).resolve()
        for name in config.files:
            path = source / name
            if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(source):
                raise ValueError(f"missing or unsafe benchmark file: {path}")
        self.root.mkdir(parents=True, exist_ok=False)
        snapshot = self.root / "benchmark"
        snapshot.mkdir()
        for name in config.files:
            dest = snapshot / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / name, dest)
        if config.adapter == "autoresearch":
            shutil.copyfile(
                Path(__file__).parent / "assets" / "autoresearch_worker.py",
                snapshot / "_eet_worker.py",
            )
        manifest = {
            str(p.relative_to(snapshot)): digest(p)
            for p in sorted(snapshot.rglob("*"))
            if p.is_file()
        }
        revision = None
        try:
            revision = git(["rev-parse", "HEAD"], cwd=source)
        except subprocess.CalledProcessError:
            pass
        state = {
            "schema": 1,
            "config": config.to_dict(),
            "manifest": manifest,
            "provenance": {
                "source_revision": revision,
                "host": platform.platform(),
                "python": sys.version,
                "created_at": time.time(),
            },
            "status": "ready",
            "generation": 0,
            "evaluation_count": 0,
            "proposal_count": 0,
            "archive": [],
            "candidates": [],
            "heldout": None,
        }
        git(["init", "--bare", str(self.root / "provenance.git")])
        self.save(state)
        return state

    def load(self):
        state = json.loads(self.path.read_text())
        if state.get("schema") != 1:
            raise ValueError("unsupported checkpoint schema")
        self.verify(state)
        return state

    def verify(self, state):
        snapshot = self.root / "benchmark"
        actual = {str(p.relative_to(snapshot)) for p in snapshot.rglob("*") if p.is_file()}
        if actual != set(state["manifest"]):
            raise ValueError("frozen benchmark file set was modified")
        for name, expected in state["manifest"].items():
            p = snapshot / name
            if not p.is_file() or p.is_symlink() or digest(p) != expected:
                raise ValueError(f"frozen benchmark was modified: {name}")
        for row in state["candidates"]:
            if (
                row.get("source_sha256")
                and hashlib.sha256(row["source"].encode()).hexdigest() != row["source_sha256"]
            ):
                raise ValueError(f"candidate source checksum mismatch: {row['id']}")

    def save(self, state):
        atomic_json(self.path, state)

    def candidate_dir(self, candidate_id):
        path = self.root / "candidates" / f"{candidate_id:05d}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def record_source(self, row, candidates):
        path = self.candidate_dir(row["id"]) / "train.py"
        path.write_text(row["source"])
        row["source_sha256"] = digest(path)
        repo = self.root / "provenance.git"
        blob = git(["hash-object", "-w", "--stdin"], data=row["source"], git_dir=repo)
        tree = git(["mktree"], data=f"100644 blob {blob}\ttrain.py\n", git_dir=repo)
        parents = []
        for parent_id in row["parents"]:
            parent = next(c for c in candidates if c["id"] == parent_id)
            if parent.get("commit"):
                parents += ["-p", parent["commit"]]
        commit = git(
            ["commit-tree", tree, *parents],
            data=f"{row['id']}: {row['description']}\n",
            git_dir=repo,
        )
        git(["update-ref", f"refs/heads/candidate-{row['id']:05d}", commit], git_dir=repo)
        row["commit"] = commit
