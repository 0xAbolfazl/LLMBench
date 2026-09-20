"""API routes for the benchmark workbench, backed by the run manager.

    GET  /api/benchmark/schema         option table the manual form is built from
    POST /api/benchmark/parse          validate a pasted or uploaded config file
    POST /api/benchmark/run            start the benchmark in the background
    GET  /api/benchmark/status         poll the running or finished run
    POST /api/benchmark/cancel         ask the running comparison to stop
    POST /api/benchmark/clear          discard a finished run
    POST /api/benchmark/export         download a run as a reusable config file
    POST /api/benchmark/results-json   download a run's result as one JSON document
    POST /api/benchmark/results-csv    download a run's summary table as CSV

A single run endpoint serves every shape: one model, several configurations
of one model, several models under one shared configuration, or a tournament
with per-model configurations. Which one it is follows from the counts in the
body.
"""

from __future__ import annotations

import csv
import io

from flask import Blueprint, Response, current_app, jsonify, request

from ..core.limits import MAX_MODELS, MAX_UPLOAD_BYTES
from ..core.options import (
    OptionError,
    decode_run_file,
    normalize_configurations,
    normalize_options,
    normalize_prompts,
    normalize_repetitions,
)
from ..services.run_manager import RunManager
from .envelope import failure, request_payload, success

blueprint = Blueprint("runs", __name__, url_prefix="/api/benchmark")


def _run_manager() -> RunManager:
    """Return the app's run manager."""
    return current_app.extensions["llmbench"]["run_manager"]


def _registry():
    """Return the app's engine registry."""
    return current_app.extensions["llmbench"]["engines"]


def _read_engine(payload: dict):
    """Resolve the engine a request names, defaulting to the first one.

    Raises:
        ValueError: With a message when the engine is unknown.
    """
    requested = (payload.get("engine") or "").strip()
    registry = _registry()

    try:
        return registry.get(requested or registry.default_id())
    except KeyError:
        known = ", ".join(engine.id for engine in registry)
        raise ValueError(f"Unknown engine '{requested}'. Known engines: {known}")


def _configured_host() -> str:
    """Return the configured Ollama host for a new run."""
    return current_app.config["OLLAMA_HOST"]


@blueprint.route("/schema")
def schema():
    """Return the option table the manual configuration form is built from.

    The table belongs to the engine named by the ``engine`` query parameter;
    every engine carries its own.
    """
    try:
        engine = _read_engine(request.args)
    except ValueError as error:
        return failure(str(error), 400)

    return success({"engine": engine.id, "options": engine.option_catalog()})


@blueprint.route("/parse", methods=["POST"])
def parse():
    """Validate configuration file contents without starting a run.

    Accepts either a multipart upload under ``file`` or a JSON body with a
    ``text`` key, so the same endpoint serves the file picker and a paste box.
    """
    try:
        engine = _read_engine(request_payload() if request.files.get("file") is None else request.form)
    except ValueError as error:
        return failure(str(error), 400)

    upload = request.files.get("file")

    if upload is not None:
        raw = upload.read(MAX_UPLOAD_BYTES + 1)

        if len(raw) > MAX_UPLOAD_BYTES:
            return failure("The configuration file is too large (limit 256 KB)", 400)

        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            return failure("The configuration file must be UTF-8 text", 400)

        source = upload.filename or "configuration.json"
    else:
        text = request_payload().get("text") or ""
        source = "pasted text"

    try:
        document = decode_run_file(text, engine.option_catalog())
    except OptionError as error:
        return failure(str(error), 400)

    if not document["configurations"]:
        return failure(
            "The file holds no configurations. Include a 'configurations' "
            "list, or an object of Ollama options.",
            400,
        )

    return success({**document, "source": source, "engine": engine.id})


def _read_models(payload: dict) -> list[str]:
    """Read and validate the model list from a run body.

    Raises:
        ValueError: With a message when the list is absent, too short, too
        long, or holds an invalid entry.
    """
    raw_models = payload.get("models")

    if not isinstance(raw_models, list):
        raise ValueError("A list of models is required")

    models: list[str] = []

    for position, model in enumerate(raw_models, start=1):
        if not isinstance(model, str) or not model.strip():
            raise ValueError(f"Model {position} must be a non-empty string")

        models.append(model.strip())

    if not models:
        raise ValueError("Pick at least one model to benchmark")

    if len(models) > MAX_MODELS:
        raise ValueError(f"At most {MAX_MODELS} models can be benchmarked in one run")

    return models


def _read_options(payload: dict) -> tuple[list[dict], dict[str, dict]]:
    """Read the shared and per-model configurations from a run body.

    Returns:
        tuple: Normalized shared configurations (possibly empty) and a
        model-name-to-options map for a tournament.

    Raises:
        ValueError: With a message when the configurations are inconsistent.
    """
    raw_models = payload.get("models") or []
    models = [m.strip() for m in raw_models if isinstance(m, str) and m.strip()]

    raw_configs = payload.get("configurations")
    configurations = normalize_configurations(raw_configs) if raw_configs else []

    # Several models only run under one shared configuration (or none); a
    # per-model assignment is the tournament's job, sent as model_configs.
    if len(models) > 1 and len(configurations) > 1:
        raise ValueError(
            "With several models, at most one shared configuration can be set. "
            "For per-model configurations, send them as 'model_configs'."
        )

    raw_model_configs = payload.get("model_configs") or {}

    if not isinstance(raw_model_configs, dict):
        raise ValueError("model_configs must be an object keyed by model name")

    model_configs: dict[str, dict] = {}

    for name, entry in raw_model_configs.items():
        if name not in models:
            raise ValueError(
                f"model_configs names '{name}', which is not among the models "
                "being benchmarked"
            )

        model_configs[name] = normalize_options(entry or {})

    return configurations, model_configs


@blueprint.route("/run", methods=["POST"])
def start_run():
    """Start a benchmark over every supplied model and configuration."""
    payload = request_payload()

    try:
        engine = _read_engine(payload)
        models = _read_models(payload)
        prompts = normalize_prompts(payload.get("prompts"))
        configurations, model_configs = _read_options(payload)
        repetitions = normalize_repetitions(payload.get("repetitions"))
    except (OptionError, ValueError) as error:
        return failure(str(error), 400)

    catalog = engine.option_catalog()

    try:
        configurations = [
            {**config, "options": normalize_options(config["options"], catalog)}
            for config in configurations
        ]
        model_configs = {
            name: normalize_options(options, catalog)
            for name, options in model_configs.items()
        }
    except OptionError as error:
        return failure(str(error), 400)

    try:
        snapshot = _run_manager().launch(
            models=models,
            prompts=prompts,
            engine_id=engine.id,
            configurations=configurations,
            model_configs=model_configs,
            include_output=bool(payload.get("include_output")),
            repetitions=repetitions,
        )
    except RuntimeError as error:
        # 409: the request is well-formed, the server is just already busy.
        return failure(str(error), 409)

    return success(snapshot)


@blueprint.route("/status")
def run_status():
    """Return the current run, or null when none has run."""
    return success(_run_manager().inspect())


@blueprint.route("/cancel", methods=["POST"])
def cancel_run():
    """Ask the running comparison to stop.

    The executor discards partial results and unloads the model it loaded;
    the run then reports the ``cancelled`` status the page already polls for.
    """
    if not _run_manager().request_stop():
        return failure("No benchmark is running", 409)

    return success(None)


@blueprint.route("/clear", methods=["POST"])
def clear_run():
    """Discard a finished run so the page starts clean.

    Nothing to discard is the state the caller asked for, not a failure.
    """
    manager = _run_manager()

    if manager.inspect() is None:
        return success(None)

    if not manager.discard():
        return failure("A benchmark is still running", 409)

    return success(None)


def _attach_download(response: Response, filename: str) -> Response:
    """Attach the download headers the page saves files through.

    Args:
        response: The response body to offer for download.
        filename: File name suggested to the browser.

    Returns:
        Response: The same response with Content-Disposition attached.
    """
    response.headers["Content-Disposition"] = f'attachment; filename="{filename}"'

    return response


@blueprint.route("/export", methods=["POST"])
def export_setup():
    """Return the current setup as a configuration file the page can
    re-upload."""
    payload = request_payload()

    try:
        engine = _read_engine(payload)
        models = _read_models(payload)
        prompts = (
            normalize_prompts(payload.get("prompts"))
            if payload.get("prompts")
            else []
        )
        configurations, model_configs = _read_options(payload)
        repetitions = normalize_repetitions(payload.get("repetitions"))
    except (OptionError, ValueError) as error:
        return failure(str(error), 400)

    catalog = engine.option_catalog()

    try:
        configurations = [
            {**config, "options": normalize_options(config["options"], catalog)}
            for config in configurations
        ]
        model_configs = {
            name: normalize_options(options, catalog)
            for name, options in model_configs.items()
        }
    except OptionError as error:
        return failure(str(error), 400)

    document = {
        "engine": engine.id,
        "models": models,
        "prompts": prompts,
        "include_output": bool(payload.get("include_output")),
        "repetitions": repetitions,
        "configurations": configurations,
        "model_configs": model_configs or None,
    }

    return _attach_download(
        jsonify(document),
        "benchmark-configurations.json",
    )


@blueprint.route("/results-json", methods=["POST"])
def download_results_json():
    """Download the finished run's result as one JSON document."""
    payload = request_payload()
    result = payload.get("result")

    if not isinstance(result, dict):
        return failure("No benchmark result was supplied", 400)

    return _attach_download(
        jsonify(result),
        f"benchmark-{payload.get('id') or 'results'}.json",
    )


@blueprint.route("/results-csv", methods=["POST"])
def download_results_csv():
    """Download the finished run's summary table as CSV.

    One row per case: the metrics the results table shows, so the file drops
    straight into a spreadsheet.
    """
    payload = request_payload()
    result = payload.get("result")

    if not isinstance(result, dict):
        return failure("No benchmark result was supplied", 400)

    cases = result.get("tests")

    if not isinstance(cases, list) or not cases:
        return failure("No benchmark result was supplied", 400)

    columns = [
        ("average_output_tokens_per_second", "output tok/s"),
        ("average_prompt_tokens_per_second", "prompt tok/s"),
        ("average_duration_seconds", "seconds"),
        ("average_ttft_seconds", "ttft s"),
        ("output_tokens_per_second_stddev", "output noise"),
        ("total_output_tokens", "output tokens"),
        ("vram_used_mb", "vram MB"),
        ("gpu_temperature_c", "gpu C"),
    ]

    # The per-case summary nests inside each case; flatten it into rows first.
    rows = []

    for case in cases:
        summary = case.get("summary") or {}
        row = {"configuration": case.get("name")}

        for key, header in columns:
            row[header] = summary.get(key)

        case_rows = case.get("results") or []
        row["prompts"] = len(case_rows)
        row["failed"] = sum(1 for entry in case_rows if not entry.get("success"))
        rows.append(row)

    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=["configuration"] + [header for _, header in columns] + ["prompts", "failed"],
    )
    writer.writeheader()
    writer.writerows(rows)

    return _attach_download(
        Response(output.getvalue(), mimetype="text/csv"),
        f"benchmark-{payload.get('id') or 'results'}.csv",
    )
