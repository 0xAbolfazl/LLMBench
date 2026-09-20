"""Background execution of a benchmark run, managed as a single worker.

A run blocks for as long as every configuration takes to answer every prompt,
so it is driven from a worker thread and the browser polls for the outcome
instead of holding a request open. One :class:`RunManager` serves every run
shape — a single model, several configurations of one model, several models
under one configuration, or a tournament — because the executor already
treats them as slices of one matrix.

Only one run is kept at a time. Two at once would contend for the same GPU
and make every timing in both meaningless, so a second request is refused
rather than queued.

A run that finishes successfully is archived automatically, so a reload or
restart does not erase what it measured. Archiving is best-effort: a store
failure is logged and dropped, since the live result is what the page is
showing anyway.
"""

from __future__ import annotations

from datetime import datetime
import threading
import time
import uuid

from ..core.engines.registry import EngineRegistry
from ..core.eventlog import record_event
from ..core.interrupt import RunInterrupted, StopSignal
from ..core.matrix import execute_matrix
from ..core.plan import build_plan
from ..core.telemetry import machine_profile
from .archive import RunArchive

COMPONENT = "bench.worker"

STATE_RUNNING = "running"


class RunManager:
    """Owns the one benchmark run the app keeps at a time."""

    def __init__(self, archive: RunArchive, engines: EngineRegistry) -> None:
        """Create a manager with no run in flight.

        Args:
            archive: The store finished runs are archived into.
            engines: The engines runs can be launched against.
        """
        self._archive = archive
        self._engines = engines
        self._lock = threading.Lock()
        self._run: dict | None = None

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def launch(
        self,
        models: list[str],
        prompts: list[str],
        engine_id: str,
        configurations: list[dict] | None = None,
        model_configs: dict[str, dict] | None = None,
        include_output: bool = False,
        repetitions: int = 1,
    ) -> dict:
        """Begin a benchmark run in the background, replacing any finished one.

        Args:
            models: Model names to run, in run order (one or more).
            engine_id: Identifier of the engine the run talks to.
            prompts: Prompt strings sent to every model and configuration.
            configurations: Optional shared configurations, each
                ``{name, options}``.
            model_configs: Optional per-model overrides, keyed by model name.
                A model named here runs under its own options; the rest keep
                the shared one.
            include_output: Keep the generated text in the results.
            repetitions: How many times every prompt runs per configuration,
                from 1.

        Returns:
            dict: Snapshot of the newly started run.

        Raises:
            RuntimeError: If a run is already running.
        """
        lanes = build_plan(models, configurations, model_configs)

        with self._lock:
            if self._run is not None and self._run["status"] == STATE_RUNNING:
                raise RuntimeError(
                    "A benchmark is already running. Wait for it to finish "
                    "before starting another."
                )

            # One entry per model-configuration leaf, so the results and the
            # plan can show exactly what will run and under which options.
            leaf_configs = [
                {
                    "name": config["name"],
                    "options": dict(config["options"]),
                    "model": lane["model"],
                }
                for lane in lanes
                for config in lane["configurations"]
            ]

            planned_steps = sum(
                len(lane["configurations"]) for lane in lanes
            ) * len(prompts) * repetitions

            self._run = {
                "id": uuid.uuid4().hex[:12],
                "status": STATE_RUNNING,
                "engine": self._engines.get(engine_id).id,
                "models": models,
                "prompts": prompts,
                "lanes": lanes,
                "configurations": leaf_configs,
                "include_output": include_output,
                "repetitions": repetitions,
                "planned_runs": planned_steps,
                "progress": {
                    "phase": "starting",
                    "percent": 0.0,
                    "steps_done": 0,
                    "steps_total": planned_steps,
                    "model": None,
                    "model_index": 0,
                    "model_count": len(models),
                    "configuration": None,
                    "configuration_index": 0,
                    "configuration_count": len(leaf_configs),
                    "prompt_index": 0,
                    "prompt_count": len(prompts),
                    "repetition": 0,
                    "repetition_count": repetitions,
                },
                "started_monotonic": time.monotonic(),
                "started_at": datetime.now().strftime("%H:%M:%S"),
                "finished_at": None,
                "elapsed_seconds": 0.0,
                "error": None,
                "result": None,
                "history_id": None,
                "signal": StopSignal(),
            }

            run = self._run

        # Daemon, so a run in flight never keeps the dev server alive.
        threading.Thread(
            target=self._execute,
            args=(run,),
            name=f"benchmark-{run['id']}",
            daemon=True,
        ).start()

        with self._lock:
            return self._snapshot(run)

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #

    def inspect(self) -> dict | None:
        """Return the current run snapshot.

        Returns:
            dict | None: Snapshot, or None when no run has been started.
        """
        with self._lock:
            return self._snapshot(self._run) if self._run is not None else None

    def is_busy(self) -> bool:
        """Report whether a run is holding the GPU right now.

        Returns:
            bool: True while a run is in flight.
        """
        with self._lock:
            return self._run is not None and self._run["status"] == STATE_RUNNING

    # ------------------------------------------------------------------ #
    # Control
    # ------------------------------------------------------------------ #

    def request_stop(self) -> bool:
        """Request cancellation of the running run.

        The executor stops at the next safe point — between prompts or
        mid-generation — discards partial results and unloads the model it
        loaded. The run then settles into the ``cancelled`` status the poller
        reports.

        Returns:
            bool: Whether a running run was found to cancel.
        """
        with self._lock:
            if self._run is None or self._run["status"] != STATE_RUNNING:
                return False

            self._run["signal"].request_stop("cancelled from the dashboard")
            return True

    def discard(self) -> bool:
        """Forget a finished run so the page can start from a clean slate.

        Returns:
            bool: Whether a run was discarded. A running run is never
            discarded; stop it first.
        """
        with self._lock:
            if self._run is None or self._run["status"] == STATE_RUNNING:
                return False

            self._run = None
            return True

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _snapshot(self, run: dict) -> dict:
        """Build the browser-facing view of a run.

        Elapsed time is derived on read rather than stored, so a running run
        reports a live duration without the worker having to tick anything.

        Args:
            run: Internal run record.

        Returns:
            dict: Run state safe to serialise.
        """
        if run["status"] == STATE_RUNNING:
            elapsed = time.monotonic() - run["started_monotonic"]
        else:
            elapsed = run["elapsed_seconds"]

        cross_model = len(run["models"]) > 1

        return {
            "id": run["id"],
            "status": run["status"],
            "engine": run["engine"],
            "cross_model": cross_model,
            "models": run["models"],
            "model": None if cross_model else run["models"][0],
            "prompts": run["prompts"],
            "configurations": run["configurations"],
            "include_output": run["include_output"],
            "repetitions": run["repetitions"],
            "planned_runs": run["planned_runs"],
            "progress": run.get("progress"),
            "history_id": run.get("history_id"),
            "elapsed_seconds": elapsed,
            "started_at": run["started_at"],
            "finished_at": run["finished_at"],
            "error": run["error"],
            "result": run["result"],
        }

    def _apply_progress(self, run: dict, step: dict) -> None:
        """Turn one executor progress step into the run's browser-facing
        progress.

        The executor reports which model and configuration a step belongs to,
        which prompt and repetition, and how many steps the whole comparison
        has finished out of how many it will run; this adds the share those
        steps make of the total, which is what a progress bar draws.

        Args:
            run: The running run the comparison belongs to.
            step: One progress step as the executor's callback delivered it.
        """
        steps_total = step["total"]
        steps_done = step["completed"]

        with self._lock:
            run["progress"] = {
                "phase": step["phase"],
                "percent": round(steps_done / steps_total * 100, 1) if steps_total else 0.0,
                "steps_done": steps_done,
                "steps_total": steps_total,
                "model": step.get("model"),
                "model_index": step.get("model_index"),
                "model_count": step.get("model_count"),
                "configuration": step.get("configuration"),
                "configuration_index": step.get("configuration_index"),
                "configuration_count": step.get("configuration_count"),
                "prompt_index": step.get("prompt_index"),
                "prompt_count": step.get("prompt_count"),
                "repetition": step.get("repetition"),
                "repetition_count": step.get("repetition_count"),
            }

    def _execute(self, run: dict) -> None:
        """Run one comparison to completion and record its outcome.

        Args:
            run: Internal run record to fill in.
        """
        engine = self._engines.get(run["engine"])

        try:
            result = execute_matrix(
                engine=engine,
                lanes=run["lanes"],
                shared_prompts=run["prompts"],
                keep_output=run["include_output"],
                interrupt=run["signal"],
                repetitions=run["repetitions"],
                on_progress=lambda step: self._apply_progress(run, step),
            )

            # The profile is what makes a saved run reproducible and
            # comparable months later: which engine, at which version, on
            # which machine.
            result["profile"] = {
                **engine.identity(),
                "machine": machine_profile(),
                "captured_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            }

            # The executor returns the full significance matrix (by_model and
            # across_models); the page shows the half its own run shape needs.
            outcome = {"status": "done", "error": None, "result": result}
        except RunInterrupted:
            # The executor discards partial results and unloads the model it
            # loaded, so a cancelled run leaves nothing behind but its log
            # entry.
            outcome = {"status": "cancelled", "error": None, "result": None}
        except Exception as error:
            record_event(
                level="ERROR",
                component=COMPONENT,
                action="run",
                message="Benchmark run failed",
                details={"id": run["id"], "error": f"{type(error).__name__}: {error}"},
            )
            outcome = {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "result": None,
            }

        with self._lock:
            run.update(outcome)
            run["elapsed_seconds"] = time.monotonic() - run["started_monotonic"]
            run["finished_at"] = datetime.now().strftime("%H:%M:%S")

            if outcome["status"] == "done" and run.get("progress"):
                # The comparison completed, so the bar reads full rather than
                # stopping at whatever the last progress step landed on.
                run["progress"] = {
                    **run["progress"],
                    "percent": 100.0,
                    "steps_done": run["progress"]["steps_total"],
                }

        if outcome["status"] == "done":
            # Outside the lock: the archive touches the filesystem, and the
            # run is already visible as finished to any poller.
            archived_id = None

            try:
                archived_id = self._archive.archive_run(outcome["result"])
            except Exception as error:
                record_event(
                    level="WARNING",
                    component=COMPONENT,
                    action="history_save",
                    message="Saving the finished run to history failed",
                    details={"error": str(error)},
                )

            with self._lock:
                run["history_id"] = archived_id
