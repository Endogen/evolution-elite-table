import json
import os
import random
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from .config import Config
from .programs import select_archive, validate
from .providers import propose
from .runner import Runner
from .storage import Store, atomic_json


class Engine:
    """Generational archive selection with concurrent proposals and a serial GPU queue.

    State is reserved before external work. Interrupted evaluations remain charged;
    interrupted model calls retain unknown billing instead of silently repeating.
    """

    def __init__(self, store, state, provider=None, runner=None):
        self.store, self.state = store, state
        self.config = Config(**state["config"]).validate()
        self._provider_slots = threading.BoundedSemaphore(self.config.proposal_workers)
        self.provider = provider or (lambda request: propose(self.config.provider, request))
        self.runner = runner or Runner(self.config, store, state)
        self.baseline = (store.root / "benchmark" / "train.py").read_text()

    @classmethod
    def create(cls, config, output):
        config.validate()
        store = Store(output)
        return cls(store, store.create(config))

    @classmethod
    def resume(cls, output):
        store = Store(output)
        return cls(store, store.load())

    def save(self):
        self.store.save(self.state)

    def recover(self):
        for row in self.state["candidates"]:
            if row["status"] == "evaluating":
                row.update(
                    status="interrupted",
                    error="evaluation interrupted; reservation remains charged",
                )
            elif row["status"] == "proposing":
                response = self.store.candidate_dir(row["id"]) / "response.json"
                if response.exists():
                    self._consume(row, json.loads(response.read_text()))
                else:
                    row.update(
                        status="proposal_interrupted",
                        error="model call interrupted; billing unknown",
                        usage={"cost_usd": None},
                    )
        self.save()

    def _row(self, operation, parents, repair_depth=0):
        row = {
            "id": len(self.state["candidates"]) + 1,
            "generation": self.state["generation"],
            "operation": operation,
            "parents": parents,
            "repair_depth": repair_depth,
            "status": "queued",
            "description": "",
            "source": "",
            "created_at": time.time(),
            "archive_context": list(self.state["archive"]),
        }
        self.state["candidates"].append(row)
        return row

    def _by_id(self, candidate_id):
        return self.state["candidates"][candidate_id - 1]

    def _request(self, row):
        cfg = self.config
        archive = [self._by_id(i) for i in row.get("archive_context", self.state["archive"])]
        # No held-out results, commands, mount paths, or test data enter a proposal.
        visible = archive if cfg.mode != "sequential" else archive[:1]

        def public(candidate):
            return {
                k: candidate[k]
                for k in ("id", "source", "description", "metrics")
                if k in candidate
            }

        request = {
            "operation": row["operation"],
            "candidate_id": row["id"],
            "baseline": self.baseline,
            "context": cfg.context,
            "training_seconds": cfg.training_seconds,
            "proposal_seed": cfg.seed * 1_000_003 + row["id"],
            "parents": [public(self._by_id(i)) for i in row["parents"]],
            "archive": [public(c) for c in visible],
        }
        if row["operation"] == "repair":
            request["diagnostic"] = self._by_id(row["parents"][0]).get("error", "invalid program")
        return request

    def _generate(self, row):
        request = self._request(row)
        self.state["proposal_count"] += 1
        row["status"] = "proposing"
        self.store.candidate_dir(row["id"]).joinpath("request.json").write_text(
            json.dumps(request, indent=2)
        )
        self.save()
        return request

    def _invoke(self, request):
        """Worker threads write only their own response file, never shared state.

        Persist billing and source as soon as the call returns, even while the
        main thread is busy training an earlier candidate.
        """
        try:
            with self._provider_slots:
                response = {"result": self.provider(request)}
        except Exception as exc:
            response = {"error": str(exc), "usage": getattr(exc, "usage", {"cost_usd": None})}
        atomic_json(self.store.candidate_dir(request["candidate_id"]) / "response.json", response)
        return response

    def _consume(self, row, response):
        if "error" in response:
            row.update(status="proposal_failed", error=response["error"], usage=response["usage"])
            self.save()
        else:
            self._accept(row, result=response["result"])

    def _accept(self, row, result):
        row.update(
            source=result["source"], description=result["description"], usage=result["usage"]
        )
        self.store.record_source(row, self.state["candidates"])
        try:
            fingerprint = validate(row["source"], self.baseline, self.config.adapter)
            if any(
                c.get("fingerprint") == fingerprint
                for c in self.state["candidates"]
                if c is not row
            ):
                row.update(status="duplicate", error="AST-equivalent candidate already attempted")
            else:
                row.update(status="generated", fingerprint=fingerprint)
        except (SyntaxError, ValueError) as exc:
            row.update(status="invalid", error=str(exc))
        self.save()

    def _evaluate(self, row):
        if self.state["evaluation_count"] >= self.config.evaluations:
            row["status"] = "budget_skipped"
            self.save()
            return
        row["status"] = "evaluating"
        self.state["evaluation_count"] += 1
        row["evaluation_index"] = self.state["evaluation_count"]
        self.save()
        row.update(self.runner.evaluate(row))
        self.save()
        value = row.get("metrics", {}).get("val_bpb")
        print(
            f"  {row['id']:04d} {row['operation']:8} {row['status']:16}"
            + (f" val_bpb={value:.6f}" if value is not None else ""),
            flush=True,
        )

    def _repair(self, row):
        while (
            row["status"] in ("invalid", "failed")
            and row["repair_depth"] < self.config.max_repairs
            and self.state["proposal_count"] < self.config.max_proposals
            and self.state["evaluation_count"] < self.config.evaluations
        ):
            parent = row
            row = self._row("repair", [parent["id"]], parent["repair_depth"] + 1)
            row["archive_context"] = list(parent.get("archive_context", self.state["archive"]))
            row["generation"] = parent["generation"]
            request = self._generate(row)
            self._consume(row, self._invoke(request))
            if row["status"] == "generated":
                self._evaluate(row)

    def _select(self):
        size = 1 if self.config.mode == "sequential" else self.config.elite_size
        self.state["archive"] = select_archive(
            self.state["candidates"], size, self.config.diversity_weight
        )
        self.save()

    def _plan(self, count):
        cfg = self.config
        rng = random.Random(cfg.seed * 1_000_003 + self.state["generation"])
        archive = self.state["archive"]
        for _ in range(count):
            if not archive:
                operation, parents = "new", []
            elif cfg.mode in ("sequential", "memory"):
                operation, parents = "edit", archive[:1]
            else:
                roll = rng.random()
                if roll < 0.2:
                    operation, parents = "new", []
                elif cfg.mode == "edit" or roll < 0.6 or len(archive) < 2:
                    operation, parents = "edit", [rng.choice(archive)]
                else:
                    operation, parents = "remix", rng.sample(archive, min(3, len(archive)))
            self._row(operation, parents)
        self.save()

    def _batch(self, rows):
        with ThreadPoolExecutor(max_workers=self.config.proposal_workers) as pool:
            futures = []
            # Requests see the same archive, independent of provider completion order.
            for row in rows:
                request = self._generate(row)
                futures.append((row, pool.submit(self._invoke, request)))
            for row, future in futures:
                self._consume(row, future.result())
                if row["status"] == "generated":
                    self._evaluate(row)
                self._repair(row)

    def run(self, environment_guard=None):
        with self.store.lock():
            # Load under lock so two resumed processes cannot use stale checkpoints.
            self.state.clear()
            self.state.update(self.store.load())
            if self.state["heldout"] is not None:
                raise ValueError("run is sealed for held-out evaluation; search cannot resume")
            if self.config.to_dict() != self.state["config"]:
                raise ValueError("configuration changed after this engine was loaded")
            if self.state["status"] in ("completed", "proposal_limit"):
                if environment_guard:
                    environment_guard(self.state)
                return self.state
            settings = self.config.provider
            if settings["kind"] == "openrouter" and not os.environ.get(
                settings.get("api_key_env", "OPENROUTER_API_KEY")
            ):
                raise ValueError("set the proposal-provider API key before launching training")
            if settings["kind"] == "command" and not shutil.which(settings["command"][0]):
                raise ValueError("proposal-provider executable was not found")
            self.runner.preflight()
            if environment_guard:
                environment_guard(self.state)
            self.recover()
            cfg = self.config
            self.state["status"] = "running"
            self.save()
            if not self.state["candidates"]:
                baseline = self._row("baseline", [])
                baseline.update(
                    source=self.baseline, description="Unmodified baseline", usage={"cost_usd": 0.0}
                )
                self.store.record_source(baseline, self.state["candidates"])
                baseline["fingerprint"] = validate(self.baseline, self.baseline, cfg.adapter)
                self._evaluate(baseline)
                if baseline["status"] != "success":
                    self.state["status"] = "baseline_failed"
                    self.save()
                    raise ValueError(
                        "baseline failed; inspect its evaluation.log and start a corrected run"
                    )
                self._select()
            if self.state["candidates"][0]["status"] != "success":
                raise ValueError("baseline did not complete successfully; start a new run")
            # Complete already-generated work without repeating a model call.
            for row in list(self.state["candidates"]):
                if row["status"] == "generated":
                    self._evaluate(row)
                if row["status"] in ("invalid", "failed") and row["operation"] != "baseline":
                    if not any(
                        c["operation"] == "repair" and c["parents"] == [row["id"]]
                        for c in self.state["candidates"]
                    ):
                        self._repair(row)
            self._select()
            while (
                self.state["evaluation_count"] < cfg.evaluations
                and self.state["proposal_count"] < cfg.max_proposals
            ):
                queued = [c for c in self.state["candidates"] if c["status"] == "queued"]
                if not queued:
                    self.state["generation"] += 1
                    size = 1 if cfg.mode in ("sequential", "memory") else cfg.population
                    count = min(
                        size,
                        cfg.max_proposals - self.state["proposal_count"],
                        cfg.evaluations - self.state["evaluation_count"],
                    )
                    self._plan(count)
                    queued = [c for c in self.state["candidates"] if c["status"] == "queued"]
                print(
                    f"Generation {self.state['generation']} · {cfg.mode} · {len(queued)} proposals",
                    flush=True,
                )
                queued = queued[
                    : min(
                        cfg.max_proposals - self.state["proposal_count"],
                        cfg.evaluations - self.state["evaluation_count"],
                    )
                ]
                self._batch(queued)
                self._select()
            self.state["status"] = (
                "completed"
                if self.state["evaluation_count"] >= cfg.evaluations
                else "proposal_limit"
            )
            self.save()
            return self.state

    def finalize(self):
        with self.store.lock():
            self.state.clear()
            self.state.update(self.store.load())
            if self.state["heldout"] is not None:
                raise ValueError(
                    "held-out evaluation already reserved; no automatic re-selection or retry"
                )
            if (
                self.state["status"] not in ("completed", "proposal_limit")
                or not self.state["archive"]
            ):
                raise ValueError("finish the search before held-out evaluation")
            if not self.config.heldout_command:
                raise ValueError(
                    "configure an independent heldout_command before starting the search"
                )
            self.runner.preflight()
            winner = self._by_id(self.state["archive"][0])
            self.state["heldout"] = {
                "candidate_id": winner["id"],
                "source_sha256": winner["source_sha256"],
                "status": "reserved",
                "reserved_at": time.time(),
            }
            self.save()
            self.state["heldout"].update(self.runner.evaluate(winner, heldout=True))
            self.state["status"] = "sealed"
            self.save()
            return self.state["heldout"]
