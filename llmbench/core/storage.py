"""Filesystem locations LLMBench reads and writes.

Everything the app persists lives under one root in the current user's local
application data::

    %LOCALAPPDATA%\\LLMBench
    ├── benchmarks/         saved benchmark runs (one JSON file each)
    └── logs/
        └── llmbench.log    the execution log

Collecting the locations here keeps the archive store and the event log from
drifting apart, and gives any caller a single place to learn where a written
artifact actually landed.

The root is deliberately per-user rather than machine-wide: ``%LOCALAPPDATA%``
is writable without elevation, so nothing depends on the app having been
started as an administrator. On a platform with no ``%LOCALAPPDATA%`` the root
falls back to ``~/.llmbench``.
"""

from __future__ import annotations

import os
from pathlib import Path

APP_NAME = "LLMBench"


def data_root() -> Path:
    """Return the root directory everything the app stores lives under.

    Resolved on each call rather than at import, so a process that adjusts
    ``%LOCALAPPDATA%`` — a test harness, or a service running under a
    different profile — is not left writing to the profile that happened to be
    current when the module was first imported.

    Returns:
        Path: ``%LOCALAPPDATA%\\LLMBench`` when that variable is set,
        otherwise ``~/.llmbench``.
    """
    local_app_data = os.environ.get("LOCALAPPDATA")

    if local_app_data:
        return Path(local_app_data) / APP_NAME

    return Path.home() / f".{APP_NAME.lower()}"


def runs_directory() -> Path:
    """Return where saved benchmark runs live.

    The on-disk name predates the redesign and is kept, so histories saved
    by earlier versions stay readable.

    Returns:
        Path: ``<data root>/benchmarks``. Not created; pass it to
        :func:`make_directory` when it has to exist.
    """
    return data_root() / "benchmarks"


def logs_directory() -> Path:
    """Return the directory the execution log is written to.

    Returns:
        Path: ``<data root>/logs``. Not created; pass it to
        :func:`make_directory` when it has to exist.
    """
    return data_root() / "logs"


LOG_FILE_NAME = "llmbench.log"
SETTINGS_FILE_NAME = "settings.json"


def log_file_path() -> Path:
    """Return the execution log's path, creating its directory when missing.

    Returns:
        Path: Absolute path to the execution log file.
    """
    return make_directory(logs_directory()) / LOG_FILE_NAME


def settings_file_path() -> Path:
    """Return the persisted settings file's path, without creating it.

    Returns:
        Path: <data root>/settings.json.
    """
    return data_root() / SETTINGS_FILE_NAME


def make_directory(path: str | Path) -> Path:
    """Create a directory and its parents, and return it.

    Args:
        path: Directory to create. Missing parents are created with it, and an
            existing directory is left as it is.

    Returns:
        Path: The directory, which now exists.

    Raises:
        PermissionError: If the process may not create the directory.
        OSError: If the directory could not be created for any other reason.
    """
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)

    return directory
