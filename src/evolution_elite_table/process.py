"""Bounded subprocesses with process-group cleanup on timeout and interruption."""

import os
import signal
import subprocess
import time
from pathlib import Path


def run_process(argv, *, cwd=None, env=None, timeout=600, input_text=None, log=None):
    start = time.monotonic()
    output = open(log, "w") if log else subprocess.PIPE
    try:
        proc = subprocess.Popen(
            argv,
            cwd=cwd,
            env=env,
            stdin=subprocess.PIPE if input_text is not None else subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT if log else subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            stdout, stderr = proc.communicate(input_text, timeout=timeout)
            return proc.returncode, stdout or "", stderr or "", time.monotonic() - start, False
        except (subprocess.TimeoutExpired, KeyboardInterrupt, SystemExit) as exc:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.communicate()
            if not isinstance(exc, subprocess.TimeoutExpired):
                raise
            return -9, "", "process exceeded wall-clock limit", time.monotonic() - start, True
    finally:
        if log:
            output.close()


def tail(path, limit=6000):
    with Path(path).open("rb") as f:
        f.seek(0, 2)
        f.seek(max(0, f.tell() - limit))
        return f.read().decode("utf-8", errors="replace")
