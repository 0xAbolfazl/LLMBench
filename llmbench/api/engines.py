"""API routes for the engine registry and each engine's server and models.

    GET  /api/engines                        the engines, in display order
    GET  /api/engines/<id>/status            installed, running, version, host
    POST /api/engines/<id>/start             start the engine's server
    POST /api/engines/<id>/stop              stop the engine's server
    GET  /api/engines/<id>/models            the installed models
    GET  /api/engines/<id>/models/running    the models resident in memory
    POST /api/engines/<id>/models/unload     unload one resident model
    POST /api/engines/<id>/models/remove     delete one installed model

Every engine answers the same shape, whatever it is behind: the benchmark
interface and the dashboard are engine-agnostic and only ever name an engine
by its id.
"""

from __future__ import annotations

from flask import Blueprint, current_app

from ..core.engines.base import EngineError, EngineUnsupported
from .envelope import explain, failure, request_payload, success

blueprint = Blueprint("engines", __name__, url_prefix="/api/engines")


def _registry():
    """Return the app's engine registry."""
    return current_app.extensions["llmbench"]["engines"]


def _resolve(engine_id: str):
    """Return one engine, or a failure envelope for an unknown id.

    Returns:
        BenchEngine | tuple: The engine, or ``(response, status)`` when the
        id names nothing registered.
    """
    try:
        return _registry().get(engine_id)
    except KeyError:
        known = ", ".join(engine.id for engine in _registry())
        return failure(f"Unknown engine '{engine_id}'. Known engines: {known}", 404)


@blueprint.route("")
def list_engines():
    """Return the registered engines in display order."""
    return success(_registry().catalog())


@blueprint.route("/<engine_id>/status")
def engine_status(engine_id: str):
    """Return whether the engine is installed, running, and at which
    version."""
    resolved = _resolve(engine_id)

    if isinstance(resolved, tuple):
        return resolved

    try:
        return success(resolved.probe())
    except Exception as error:
        return failure(explain(error))


@blueprint.route("/<engine_id>/start", methods=["POST"])
def engine_start(engine_id: str):
    """Start the engine's server and wait for it to answer."""
    resolved = _resolve(engine_id)

    if isinstance(resolved, tuple):
        return resolved

    payload = request_payload()

    try:
        timeout = float(payload.get("timeout") or 0) or None
    except (TypeError, ValueError):
        return failure("timeout must be a number", 400)

    try:
        if timeout is not None:
            resolved.bring_up(timeout)
        else:
            resolved.bring_up()
    except EngineError as error:
        return failure(str(error), 502)
    except Exception as error:
        return failure(explain(error))

    return success(resolved.probe(), started=True)


@blueprint.route("/<engine_id>/stop", methods=["POST"])
def engine_stop(engine_id: str):
    """Stop the engine's server and wait for it to go quiet."""
    resolved = _resolve(engine_id)

    if isinstance(resolved, tuple):
        return resolved

    try:
        stopped = resolved.shut_down()
    except EngineError as error:
        return failure(str(error), 502)
    except Exception as error:
        return failure(explain(error))

    status = resolved.probe()

    # A supervised server process can be respawned by its parent after a
    # kill, so a successful stop can still leave the API reachable. Report
    # that rather than claiming a stop that did not stick.
    return success(status, stopped=stopped, restarted=status.get("running", False))


@blueprint.route("/<engine_id>/models")
def engine_models(engine_id: str):
    """Return the models the engine can run."""
    resolved = _resolve(engine_id)

    if isinstance(resolved, tuple):
        return resolved

    try:
        return success(resolved.installed_models())
    except EngineError as error:
        return failure(str(error), 502)
    except Exception as error:
        return failure(explain(error))


@blueprint.route("/<engine_id>/models/running")
def engine_models_running(engine_id: str):
    """Return the models currently resident in memory."""
    resolved = _resolve(engine_id)

    if isinstance(resolved, tuple):
        return resolved

    try:
        return success(resolved.loaded_models())
    except EngineError as error:
        return failure(str(error), 502)
    except Exception as error:
        return failure(explain(error))


@blueprint.route("/<engine_id>/models/unload", methods=["POST"])
def engine_model_unload(engine_id: str):
    """Unload one model resident in memory."""
    resolved = _resolve(engine_id)

    if isinstance(resolved, tuple):
        return resolved

    name = (request_payload().get("model") or "").strip()

    if not name:
        return failure("A model name is required", 400)

    try:
        unloaded = resolved.unload(name)
    except Exception as error:
        return failure(explain(error))

    if not unloaded:
        return failure(f"Could not unload {name}", 502)

    return success(None)


@blueprint.route("/<engine_id>/models/remove", methods=["POST"])
def engine_model_remove(engine_id: str):
    """Delete one installed model from the engine's storage."""
    resolved = _resolve(engine_id)

    if isinstance(resolved, tuple):
        return resolved

    name = (request_payload().get("model") or "").strip()

    if not name:
        return failure("A model name is required", 400)

    try:
        resolved.delete_model(name)
    except EngineUnsupported as error:
        return failure(str(error), 501)
    except EngineError as error:
        return failure(str(error), 502)
    except Exception as error:
        return failure(explain(error))

    return success(None)
