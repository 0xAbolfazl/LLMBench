"""API routes for the execution log.

    GET /api/logs     the log entries, newest last, with optional filters
    DELETE /api/logs  wipe the log file

The log is written by every benchmark step as it happens; this blueprint only
reads it back out, parsed and filtered by level, component, action, or a cap
on how many of the newest entries come back.
"""

from __future__ import annotations

from pathlib import Path

from flask import Blueprint, request

from ..core.eventlog import read_events, wipe_events
from ..core.storage import log_file_path
from .envelope import failure, success

blueprint = Blueprint("events", __name__, url_prefix="/api/logs")

# The page shows a table, not an archive, so the default cap keeps the
# payload small on a machine that has been running benchmarks for weeks.
DEFAULT_LIMIT = 200
MAX_LIMIT = 2000


@blueprint.route("")
def list_events():
    """Return the execution log with optional level/component/action filters.

    Query parameters:
        level:     Only entries at this severity (INFO, WARNING, ERROR).
        component: Only entries from this component.
        action:    Only entries for this action.
        limit:     Cap on how many of the newest matching entries come back.
    """
    # A limit that does not parse as an integer is dropped rather than
    # rejected: the query then simply answers uncapped, which is a sensible
    # reading of a mistyped filter.
    limit = request.args.get("limit", type=int)

    if limit is not None:
        if limit < 1:
            return failure("limit must be 1 or greater", 400)

        limit = min(limit, MAX_LIMIT)

    entries = read_events(
        level=request.args.get("level") or None,
        component=request.args.get("component") or None,
        action=request.args.get("action") or None,
        tail=limit,
    )

    file_path = log_file_path()

    return success(
        {
            "entries": entries,
            "file": {
                "path": str(file_path),
                "exists": file_path.exists(),
                "size_bytes": (
                    Path(file_path).stat().st_size if file_path.exists() else 0
                ),
            },
        }
    )


@blueprint.route("", methods=["DELETE"])
def clear_events():
    """Wipe the log file so the station starts from an empty table."""
    removed = wipe_events()

    return success({"removed": removed})
