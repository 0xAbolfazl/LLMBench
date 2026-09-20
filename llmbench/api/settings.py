"""API routes for the persisted engine settings.

    GET /api/settings              every configurable engine's current values
    PUT /api/settings/<engine_id>  validate and save one engine's values

The settings are the few facts only the person at the machine knows — where
llama.cpp is installed, which folders hold the gguf files. They persist in
``settings.json`` under the app's data root, and a value saved here wins over
the matching environment variable.
"""

from __future__ import annotations

from flask import Blueprint, current_app

from ..core.engines.base import EngineError
from .envelope import failure, request_payload, success

blueprint = Blueprint("settings", __name__, url_prefix="/api/settings")


def _registry():
    """Return the app's engine registry."""
    return current_app.extensions["llmbench"]["engines"]


@blueprint.route("")
def list_settings():
    """Return every configurable engine's configuration view."""
    views = {}

    for engine in _registry():
        if not engine.configurable:
            continue

        views[engine.id] = engine.config_view()

    return success(views)


@blueprint.route("/<engine_id>", methods=["POST"])
def save_settings(engine_id: str):
    """Validate and save one engine's configuration.

    Like every mutating endpoint here, this is a POST. The engine validates
    the patch itself — a folder must exist, a home must hold its executable —
    so a bad path is refused before it is persisted.
    """
    try:
        engine = _registry().get(engine_id)
    except KeyError:
        known = ", ".join(item.id for item in _registry())
        return failure(f"Unknown engine '{engine_id}'. Known engines: {known}", 404)

    if not engine.configurable:
        return failure(f"The {engine.label} engine has no settings to edit.", 400)

    patch = request_payload()

    if not isinstance(patch, dict) or not patch:
        return failure("A settings object is required", 400)

    try:
        view = engine.configure(patch)
    except EngineError as error:
        return failure(str(error), 400)
    except OSError as error:
        return failure(f"The settings could not be saved: {error}", 500)

    return success(view)