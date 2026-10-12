import argparse
import importlib.resources
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from .config import MODES, Config
from .engine import Engine
from .report import write_report
from .storage import Store, atomic_json
from .study import execute_study


def demo_config(output, seed, evaluations, mode, *, write=True):
    # Put assets beside the run, and package them so the installed CLI works anywhere.
    folder = Path(output).resolve().parent / "tiny-lm-benchmark"
    if write:
        folder.mkdir(parents=True, exist_ok=True)
    assets = importlib.resources.files("evolution_elite_table") / "assets"
    for src, dest in (("tiny_train.py", "train.py"), ("tiny_benchmark.py", "benchmark.py")):
        text = (assets / src).read_text()
        target = folder / dest
        if target.exists() and target.read_text() != text:
            raise ValueError(f"demo fixture was modified: {target}")
        if write:
            target.write_text(text)
    return Config(
        benchmark=str(folder),
        files=["train.py", "benchmark.py"],
        mode=mode,
        seed=seed,
        evaluations=evaluations,
        max_proposals=evaluations * 5,
        population=5,
        elite_size=4,
        training_seconds=0,
        timeout_seconds=30,
        training_tolerance_seconds=30,
        adapter="generic",
        backend="local",
        command=[sys.executable, "train.py"],
        heldout_command=[sys.executable, "train.py", "--heldout"],
        provider={"kind": "demo"},
        context="CPU demonstration: optimize a byte n-gram model. Scripted proposals; no LLM or GPU.",
    )


def parser():
    root = argparse.ArgumentParser(
        description="Evolutionary autoresearch with an elite archive and explicit experiment budgets"
    )
    commands = root.add_subparsers(dest="action", required=True)
    demo = commands.add_parser(
        "demo", help="run a real tiny CPU language-model search with scripted proposals"
    )
    demo.add_argument("--output", default="runs/demo")
    demo.add_argument("--seed", type=int, default=0)
    demo.add_argument("--evaluations", type=int, default=16)
    demo.add_argument("--mode", choices=MODES, default="remix")
    demo.add_argument("--finalize", action="store_true")
    run = commands.add_parser("run", help="start a search from a JSON configuration")
    run.add_argument("config")
    run.add_argument("--output", required=True)
    run.add_argument("--dry-run", action="store_true")
    for name in ("check", "baseline"):
        sub = commands.add_parser(
            name,
            help="check the GPU environment"
            if name == "check"
            else "evaluate only the baseline, without model calls",
        )
        sub.add_argument("config")
        sub.add_argument("--output", required=True)
    for name in ("resume", "finalize", "report"):
        sub = commands.add_parser(name)
        sub.add_argument("run")
    study = commands.add_parser(
        "study", help="run all four variants across paired independent search seeds"
    )
    study.add_argument("config", nargs="?")
    study.add_argument("--demo", action="store_true")
    study.add_argument("--output", required=True)
    study.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    study.add_argument("--evaluations", type=int, default=None)
    study.add_argument("--finalize", action="store_true")
    study.add_argument("--dry-run", action="store_true")
    study.add_argument("--resume", action="store_true")
    return root


def _report(store):
    with store.lock():
        return write_report(store, store.load())


def _run(engine, finalize=False, environment_guard=None):
    try:
        engine.run(environment_guard=environment_guard)
        if finalize:
            engine.finalize()
    except BaseException:
        # Never write into another process's run or hide the original error.
        try:
            _report(engine.store)
        except (ValueError, OSError) as report_error:
            print(f"Could not refresh report: {report_error}", file=sys.stderr)
        raise
    result = _report(engine.store)
    print(json.dumps(result, indent=2))
    return result


def _plan(cfg):
    return {
        "mode": cfg.mode,
        "seed": cfg.seed,
        "evaluation_limit": cfg.evaluations,
        "maximum_training_hours": cfg.evaluations * cfg.training_seconds / 3600,
        "maximum_evaluation_wall_hours": cfg.evaluations * cfg.timeout_seconds / 3600,
        "max_proposal_calls": cfg.max_proposals,
        "model": cfg.provider.get("model", cfg.provider["kind"]),
        "backend": cfg.backend,
        "image": cfg.image if cfg.backend == "docker" else None,
    }


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.action == "demo":
            cfg = demo_config(args.output, args.seed, args.evaluations, args.mode)
            _run(Engine.create(cfg, args.output), args.finalize)
        elif args.action == "run":
            cfg = Config.load(args.config)
            if args.dry_run:
                print(json.dumps(_plan(cfg), indent=2))
            else:
                _run(Engine.create(cfg, args.output))
        elif args.action in ("check", "baseline"):
            cfg = Config.load(args.config, baseline_only=True)
            engine = Engine.create(cfg, args.output)
            if args.action == "baseline":
                _run(engine)
            else:
                with engine.store.lock():
                    result = engine.runner.probe()
                    atomic_json(engine.store.root / "check.json", result)
                print(json.dumps(result, indent=2))
        elif args.action == "resume":
            _run(Engine.resume(args.run))
        elif args.action == "finalize":
            engine = Engine.resume(args.run)
            try:
                result = engine.finalize()
            finally:
                _report(engine.store)
            print(json.dumps(result, indent=2))
        elif args.action == "report":
            store = Store(args.run)
            with store.lock():
                print(json.dumps(write_report(store, store.load()), indent=2))
        elif args.action == "study":
            if bool(args.config) == args.demo:
                raise ValueError("choose either a configuration file or --demo")
            if len(set(args.seeds)) != len(args.seeds):
                raise ValueError("study seeds must be unique")
            output = Path(args.output).resolve()
            cfg = (
                demo_config(
                    output / "fixture",
                    0,
                    16 if args.evaluations is None else args.evaluations,
                    "remix",
                    write=not args.dry_run,
                )
                if args.demo
                else Config.load(args.config)
            )
            if args.evaluations is not None:
                cfg = replace(cfg, evaluations=args.evaluations)
            configs = [
                replace(cfg, mode=mode, seed=seed).validate()
                for seed in args.seeds
                for mode in MODES
            ]
            if args.finalize and not cfg.heldout_command:
                raise ValueError("configure heldout_command before a study with --finalize")
            if args.dry_run:
                print(json.dumps([_plan(c) for c in configs], indent=2))
                return
            execute_study(configs, output, resume=args.resume, finalize=args.finalize, run=_run)
            print(f"Study saved to {output / 'study.json'}")
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    except KeyboardInterrupt:
        print("Interrupted. Reservations are preserved; use eet resume RUN.", file=sys.stderr)
        raise SystemExit(130) from None
