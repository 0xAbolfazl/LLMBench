"""Execution event log: record and read back the app's own activity.

Each event is one pipe-delimited line with the structured details encoded as
trailing JSON::

    2026-09-19 12:04:11 | INFO | bench.executor | cell | Cell started | {"name": "hot"}

Logging is best-effort: an entry that cannot be written — a full disk, a
locked file, a profile directory that cannot be created — is dropped rather
than raised, because failing the operation being logged would be the larger
loss. Every long-running step logs, so this module is never the reason one of
them fails.
"""

from datetime import datetime
import json

from .storage import log_file_path


def record_event(
    level: str,
    component: str,
    action: str,
    message: str,
    details: dict | None = None,
) -> None:
    """Append a new execution event to the log file.

    Args:
        level: Log severity (e.g., 'INFO', 'WARNING', 'ERROR').
        component: System component originating the event.
        action: Operation being performed.
        message: Human-readable description of the log event.
        details: Optional dictionary of additional event metadata.
    """
    if details is None:
        details = {}

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    line = (
        f"{timestamp} | "
        f"{level} | "
        f"{component} | "
        f"{action} | "
        f"{message} | "
        f"{json.dumps(details, ensure_ascii=False)}\n"
    )

    try:
        with open(log_file_path(), "a", encoding="utf-8") as file:
            file.write(line)
    except OSError:
        # Includes a PermissionError on the log directory or the file itself.
        return


def wipe_events() -> bool:
    """Delete the log file, wiping every recorded event.

    Returns:
        bool: True when a file existed and was removed, False otherwise.
    """
    try:
        path = log_file_path()
    except OSError:
        return False

    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return False
    except OSError:
        # The file is locked or the directory refuses the delete; the log
        # keeps working, the entries simply stay.
        return False


def _parse_line(line: str) -> dict | None:
    """Parse one raw log line into its fields.

    Args:
        line: Raw line as read from the log file.

    Returns:
        dict | None: Parsed entry, or None if the line is malformed or its
        details are not valid JSON.
    """
    # Only the four leading fields are delimiter-free, so the split stops there
    # and the message and details are separated afterwards.
    parts = line.strip().split(" | ", 4)

    if len(parts) != 5:
        return None

    # The message and the JSON details may both contain " | ", so the boundary
    # between them is found by taking the longest trailing segment that parses
    # as JSON. Everything before it is the message.
    segments = parts[4].split(" | ")

    for position in range(1, len(segments)):
        try:
            details = json.loads(" | ".join(segments[position:]))
        except json.JSONDecodeError:
            continue

        return {
            "timestamp": parts[0],
            "level": parts[1],
            "component": parts[2],
            "action": parts[3],
            "message": " | ".join(segments[:position]),
            "details": details,
        }

    return None


def read_events(
    level: str | None = None,
    component: str | None = None,
    action: str | None = None,
    tail: int | None = None,
) -> list[dict]:
    """Read recorded events with optional filtering.

    The three value filters select which entries match; ``tail`` then caps how
    many of those matches come back, keeping the most recent ones, since the
    log is appended in chronological order.

    Args:
        level: Optional log severity to filter by (e.g., 'ERROR').
        component: Optional component name to filter by.
        action: Optional action name to filter by.
        tail: Optional maximum number of entries to return, counted back from
            the newest match.

    Returns:
        list[dict]: Parsed log entries matching the criteria, oldest first.

    Raises:
        ValueError: If tail is given but is not 1 or greater.
    """
    if tail is not None and tail < 1:
        raise ValueError(f"tail must be 1 or greater, got {tail}")

    try:
        path = log_file_path()
    except OSError:
        return []

    if not path.exists():
        return []

    matches: list[dict] = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            entry = _parse_line(line)

            if entry is None:
                continue

            if level and entry["level"] != level:
                continue

            if component and entry["component"] != component:
                continue

            if action and entry["action"] != action:
                continue

            matches.append(entry)

            if tail is not None and len(matches) > tail:
                matches.pop(0)

    return matches
