"""Persisted benchmark history — every saved run is one JSON file.

A comparison run is the unit of history: the models that were benchmarked,
the prompts their configurations answered, and one result per
model-configuration pair. The pieces belong together — a configuration's
numbers only mean something against the others it was compared with — so a
run is stored whole in a single file rather than split per configuration or
per model.

Layout under the app's data root (see :mod:`llmbench.core.storage`)::

    %LOCALAPPDATA%\\LLMBench\\
    └── runs/
        ├── index.json                  one small record per saved run
        └── 20260905T211530_a1b2c3.json the full result of one run

The index exists so listing runs never opens the result files: one record per
run with the identity fields a list shows (model, time, winner, noise
verdict), kept in the same write as the result so the two cannot disagree. A
run dropped from the index is dropped from disk with it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import time
import uuid

from ..core.eventlog import record_event
from ..core.storage import make_directory, runs_directory

COMPONENT = "bench.archive"

INDEX_NAME = "index.json"

# Runs kept on disk. Oldest beyond the cap are removed — with their index
# records — so a long-running machine does not accumulate files without
# bound.
RETENTION_LIMIT = 100

# A benchmark id is minted here and used as a file name, so anything arriving
# from outside is checked before it touches the filesystem.
_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


class RunNotFoundError(FileNotFoundError):
    """Raised when no benchmark with the requested identifier is stored."""


class RunArchive:
    """The on-disk store of finished benchmark runs."""

    def __init__(self) -> None:
        """Point the archive at the data root's runs directory."""
        self._directory = make_directory(runs_directory())

    # ------------------------------------------------------------------ #
    # Storage primitives
    # ------------------------------------------------------------------ #

    def _index_path(self) -> Path:
        """Return the index file's path, beside the result files."""
        return self._directory / INDEX_NAME

    def _record_path(self, run_id: str) -> Path:
        """Return the file a run id is stored under.

        Args:
            run_id: Identifier of a saved run.
        """
        return self._directory / f"{run_id}.json"

    @staticmethod
    def _write_atomically(path: Path, payload) -> None:
        """Write JSON to a path through a temporary file and an atomic
        replace.

        A reader either sees the previous file or the new one, never a partial
        write left behind by a crash between the two.

        Args:
            path: Destination file.
            payload: JSON-serialisable value.

        Raises:
            OSError: If the write or the replace fails.
        """
        temporary = path.with_name(f"{path.name}.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )

        os.replace(temporary, path)

    @staticmethod
    def _read_json(path: Path):
        """Parse a JSON file, tolerating absence and corruption alike.

        A result file or index that cannot be read is treated as absent
        rather than failing a listing: the store's job is to serve what
        survived, and a damaged record is not worth taking the whole history
        down for.

        Args:
            path: File to read.

        Returns:
            The parsed value, or None when the file is missing or unreadable.
        """
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def _read_index(self) -> list[dict]:
        """Read the index records, newest first.

        Returns:
            list[dict]: Index records, or an empty list when there is none
            yet.
        """
        records = self._read_json(self._index_path())

        if not isinstance(records, list):
            return []

        return [record for record in records if isinstance(record, dict)]

    def _write_index(self, records: list[dict]) -> None:
        """Replace the whole index with the given records, newest first.

        Args:
            records: Index records in the order they should be listed.
        """
        self._write_atomically(self._index_path(), records)

    # ------------------------------------------------------------------ #
    # Record shaping
    # ------------------------------------------------------------------ #

    @staticmethod
    def _multi_model(cases: list) -> bool:
        """Tell whether a comparison's cases span more than one model.

        Args:
            cases: One case result per comparison step.

        Returns:
            bool: True when the cases carry more than one distinct model
            name.
        """
        return (
            len({
                case.get("model")
                for case in cases
                if isinstance(case, dict) and case.get("model")
            })
            > 1
        )

    @staticmethod
    def _case_label(case: dict, multi_model: bool) -> str:
        """Name one case result the way a history record lists it.

        A comparison that spans several models can carry two configurations
        of the same name — two models' defaults, say — so there the model's
        name prefixes the configuration's, keeping every entry in the record
        distinct. A case named after its own model needs no prefix.

        Args:
            case: One case result.
            multi_model: Whether the result's cases span more than one model.

        Returns:
            str: The label the case is listed under.
        """
        name = case.get("name", "unnamed")
        model = case.get("model")

        if multi_model and model and model != name:
            return f"{model} / {name}"

        return name

    @staticmethod
    def _distinct_prompts(cases: list) -> list:
        """Collect the distinct prompts a comparison's cases answered.

        Configurations answer the same shared prompt list, but a run can
        still reach the store with rows missing or reordered between cases,
        so the header takes the union across every case in first-seen order.

        Args:
            cases: One case result per comparison step.

        Returns:
            list: The distinct prompts, in the order they were first
            answered.
        """
        prompts: list = []
        seen: set = set()

        for case in cases:
            if not isinstance(case, dict):
                continue

            for row in case.get("results", []):
                if not isinstance(row, dict):
                    continue

                prompt = row.get("prompt")

                if isinstance(prompt, str) and prompt not in seen:
                    seen.add(prompt)
                    prompts.append(prompt)

        return prompts

    @staticmethod
    def _verdict_flag(significance) -> bool | None:
        """Read the one flag a history list shows from any verdict shape.

        The matrix verdict holds a judgement per model and one across models,
        and the list shows the cross-model one when there is, falling back to
        the first within-model judgement the matrix could make.

        Args:
            significance: The 'significance' of a comparison result.

        Returns:
            bool | None: The flag, or None when nothing measurable was
            judged.
        """
        if not isinstance(significance, dict):
            return None

        if "significant" in significance:
            return significance.get("significant")

        across = significance.get("across_models")

        if isinstance(across, dict):
            return across.get("significant")

        for assessment in significance.get("by_model", {}).values():
            if isinstance(assessment, dict) and assessment.get("significant") is not None:
                return assessment.get("significant")

        return None

    @classmethod
    def _summarize_result(cls, result: dict) -> dict | None:
        """Read the winner's figures a list needs out of a comparison result.

        Args:
            result: A comparison result as produced by the executor.

        Returns:
            dict | None: Per-configuration averages keyed by label, the
            fastest configuration's label, and the significance flag when one
            was made; None when the result holds no measurable configuration.
        """
        cases = [case for case in result.get("tests", []) if isinstance(case, dict)]

        multi_model = cls._multi_model(cases)
        averages: dict = {}

        for case in cases:
            rate = case.get("summary", {}).get("average_output_tokens_per_second")

            if rate is not None:
                averages[cls._case_label(case, multi_model)] = rate

        if not averages:
            return None

        best = max(averages.values())
        tied = [label for label, rate in averages.items() if rate == best]

        # A dead heat has no fastest; the list shows a dash for the winner.
        winner = tied[0] if len(tied) == 1 else None

        return {
            "average_output_tokens_per_second": averages,
            "winner": winner,
            "significant": cls._verdict_flag(result.get("significance")),
        }

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def archive_run(self, result: dict) -> str:
        """Store a comparison result as one history entry.

        The result is kept exactly as given — cases, per-prompt rows, noise
        spreads, significance — under a header naming what was compared. A
        cross-model run names what it measured under 'models', and a
        single-model one carries its model under 'model'; both are accepted.
        Two runs saved in the same second are told apart by a short random
        suffix, so an identifier is never reused.

        Args:
            result: A comparison result as produced by
                :func:`llmbench.core.matrix.execute_matrix`, carrying 'tests'
                plus either 'model' or a 'models' list, and optionally
                'significance'.

        Returns:
            str: The identifier the run was saved under.

        Raises:
            TypeError: If result is not a dictionary.
            ValueError: If the result names no model(s) or holds no test
                results.
            OSError: If the run cannot be written to disk.
        """
        if not isinstance(result, dict):
            raise TypeError("result must be a dictionary")

        model = result.get("model")
        models = result.get("models")

        if models is None and isinstance(model, str) and model.strip():
            models = [model]

        if (
            not isinstance(models, list)
            or not models
            or not all(isinstance(name, str) and name.strip() for name in models)
        ):
            raise ValueError(
                "result must carry what it benchmarked: a 'model' name or a "
                "non-empty 'models' list"
            )

        model = model if isinstance(model, str) and model.strip() else None

        cases = result.get("tests")

        if not isinstance(cases, list) or not cases:
            raise ValueError("result must carry at least one test result")

        multi_model = self._multi_model(cases)

        saved_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        run_id = time.strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:6]

        configurations = [
            {
                "name": self._case_label(case, multi_model),
                "options": case.get("configuration", {}),
            }
            for case in cases
            if isinstance(case, dict)
        ]

        profile = result.get("profile") if isinstance(result.get("profile"), dict) else None

        record = {
            "id": run_id,
            "saved_at": saved_at,
            "engine": result.get("profile", {}).get("engine") if isinstance(result.get("profile"), dict) else None,
            "model": model,
            "models": models,
            "prompts": self._distinct_prompts(cases),
            "configurations": configurations,
            "repetitions": cases[0].get("repetitions", 1) if isinstance(cases[0], dict) else 1,
            "summary": self._summarize_result(result),
            "profile": profile,
            "result": result,
        }

        self._write_atomically(self._record_path(run_id), record)

        records = self._read_index()
        records.insert(
            0,
            {
                "id": run_id,
                "saved_at": saved_at,
                "engine": record["engine"],
                "model": model,
                "models": models,
                "repetitions": record["repetitions"],
                "configuration_count": len(record["configurations"]),
                "prompt_count": len(record["prompts"]),
                "winner": record["summary"]["winner"] if record["summary"] else None,
                "significant": record["summary"]["significant"] if record["summary"] else None,
            },
        )
        records = records[:RETENTION_LIMIT]
        self._write_index(records)

        # The cap applies to files as well: results that fell off the index
        # are deleted, so disk and index always agree.
        kept_ids = {entry["id"] for entry in records}

        for path in sorted(self._directory.glob("*.json")):
            if path.name == INDEX_NAME:
                continue

            if path.stem not in kept_ids:
                try:
                    path.unlink()
                except OSError:
                    continue

        record_event(
            level="INFO",
            component=COMPONENT,
            action="save",
            message="Benchmark saved to history",
            details={
                "id": run_id,
                "model": model,
                "models": models,
                "configuration_count": len(record["configurations"]),
                "prompt_count": len(record["prompts"]),
            },
        )

        return run_id

    def list_runs(self) -> list[dict]:
        """List the saved benchmark runs, newest first.

        The listing comes from the index, so it costs one small file however
        many runs are stored. A record whose result file has gone missing on
        its own — deleted outside this store, say — is left out rather than
        listed and then failing to open.

        Returns:
            list[dict]: One record per saved run, each carrying 'id',
            'saved_at', 'model' (or 'models' for a cross-model run),
            'repetitions', 'configuration_count', 'prompt_count', 'winner'
            and 'significant'.
        """
        records = self._read_index()
        available = {
            path.stem
            for path in self._directory.glob("*.json")
            if path.name != INDEX_NAME
        }

        return [record for record in records if record.get("id") in available]

    def fetch(self, run_id: str) -> dict:
        """Read one saved benchmark run in full.

        Args:
            run_id: Identifier as returned by :meth:`archive_run` or listed
                by :meth:`list_runs`.

        Returns:
            dict: The stored record — the header, the summary for lists, and
            the complete comparison result under 'result'.

        Raises:
            ValueError: If the identifier is not a plain id.
            RunNotFoundError: If no run is stored under that identifier.
        """
        if not run_id or not _ID_PATTERN.match(run_id):
            raise ValueError(f"Not a benchmark id: '{run_id}'")

        record = self._read_json(self._record_path(run_id))

        if record is None:
            record_event(
                level="WARNING",
                component=COMPONENT,
                action="load",
                message="Benchmark record not found",
                details={"id": run_id},
            )
            raise RunNotFoundError(f"No benchmark is stored under '{run_id}'.")

        return record

    def remove(self, run_id: str) -> bool:
        """Remove one saved benchmark run from the history.

        Args:
            run_id: Identifier of the run to remove.

        Returns:
            bool: True when a run was removed, False when nothing was stored
            under that identifier — removing a missing run is the state the
            caller asked for, not a failure.

        Raises:
            ValueError: If the identifier is not a plain id.
        """
        if not run_id or not _ID_PATTERN.match(run_id):
            raise ValueError(f"Not a benchmark id: '{run_id}'")

        removed = False
        path = self._record_path(run_id)

        if path.exists():
            try:
                path.unlink()
                removed = True
            except OSError:
                return False

        records = self._read_index()
        remaining = [record for record in records if record.get("id") != run_id]

        if len(remaining) != len(records):
            removed = True

            try:
                self._write_index(remaining)
            except OSError:
                return False

        if removed:
            record_event(
                level="INFO",
                component=COMPONENT,
                action="delete",
                message="Benchmark removed from history",
                details={"id": run_id},
            )

        return removed
