"""API routes for the benchmark history, backed by :class:`RunArchive`.

Finished runs are archived automatically (see the run manager); this
blueprint only serves the store back out:

    GET    /api/history            the index, newest first
    GET    /api/history/<id>       one run in full
    DELETE /api/history/<id>       remove one run
"""

from __future__ import annotations

from flask import Blueprint, current_app

from ..services.archive import RunNotFoundError, RunArchive
from .envelope import failure, success

blueprint = Blueprint("history", __name__, url_prefix="/api/history")


def _archive() -> RunArchive:
    """Return the app's run archive."""
    return current_app.extensions["llmbench"]["archive"]


@blueprint.route("")
def list_history():
    """List the saved benchmark runs, newest first."""
    return success(_archive().list_runs())


@blueprint.route("/<run_id>")
def get_history(run_id: str):
    """Return one saved run in full, header and result alike."""
    try:
        return success(_archive().fetch(run_id))
    except ValueError as error:
        return failure(str(error), 400)
    except RunNotFoundError as error:
        return failure(str(error), 404)


@blueprint.route("/<run_id>", methods=["DELETE"])
def delete_history(run_id: str):
    """Remove one saved run from the history."""
    try:
        deleted = _archive().remove(run_id)
    except ValueError as error:
        return failure(str(error), 400)

    return success({"deleted": deleted})
