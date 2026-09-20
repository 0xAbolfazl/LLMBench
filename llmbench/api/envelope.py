"""JSON transport envelope shared by every API route.

Every endpoint answers with the same shape so the browser has one code path
for success and failure::

    {"ok": bool, "error": str | None, "data": Any, ...extra}
"""

from __future__ import annotations

import json
import re
import urllib.error

from flask import jsonify, request

# The Ollama CLI writes spinner frames and colour codes to its output even when
# it is not attached to a terminal, so strip them before anything is displayed.
ANSI_ESCAPE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]|\x1B[@-Z\\-_]")
SPINNER_CHARS = re.compile(r"[⠁-⣿]")


def strip_terminal_noise(value):
    """Strip terminal control sequences and spinner frames from CLI output.

    Non-string values pass through untouched so JSON payloads from the Ollama
    HTTP API are not mangled.

    Args:
        value: Text from a CLI command, or any other value.

    Returns:
        The cleaned string, or the original value when it is not a string.
    """
    if not isinstance(value, str):
        return value

    text = ANSI_ESCAPE.sub("", value)
    text = SPINNER_CHARS.sub("", text)

    # Progress output redraws one line with carriage returns; only the final
    # state of each line is meaningful once the sequences are gone.
    lines = [line.split("\r")[-1].rstrip() for line in text.splitlines()]

    return "\n".join(line for line in lines if line.strip()).strip()


def success(data=None, **extra):
    """Build a success envelope.

    Args:
        data: Payload for the caller.
        **extra: Additional top-level keys.

    Returns:
        flask.Response: JSON response with HTTP 200.
    """
    return jsonify({"ok": True, "error": None, "data": strip_terminal_noise(data), **extra})


def failure(error: str, status: int = 500):
    """Build a failure envelope with an HTTP status.

    Args:
        error: Message to show the user.
        status: HTTP status code. Defaults to 500.

    Returns:
        tuple: JSON response and status code.
    """
    return jsonify({"ok": False, "error": strip_terminal_noise(error), "data": None}), status


def explain(error: Exception) -> str:
    """Build the most specific message available for an exception.

    The Ollama client re-raises HTTP failures as a generic RuntimeError, so
    the Ollama API's own explanation only survives on the chained cause. Read
    it back when it is there.

    Args:
        error: Exception raised by a core call.

    Returns:
        str: Error text, including the upstream Ollama message when
        available.
    """
    message = f"{type(error).__name__}: {error}"
    cause = error.__cause__

    if isinstance(cause, urllib.error.HTTPError):
        try:
            detail = json.loads(cause.read().decode("utf-8")).get("error")
        except Exception:
            detail = None

        if detail:
            return f"{message} — {detail}"

    return message


def request_payload() -> dict:
    """Return the request JSON body, tolerating an empty payload.

    Returns:
        dict: Parsed body, or an empty dict when none was sent.
    """
    return request.get_json(silent=True) or {}
