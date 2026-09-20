"""Persisted application settings — the few values a user configures.

Most of what an engine needs to run is discovered or fixed, but a few facts
can only come from the person at the machine: where llama.cpp is installed,
and which folders hold their ``.gguf`` files. Those live here, in one small
JSON file under the app's data root, so a choice made once in the interface
survives a restart.

Precedence is deliberate: a value saved through the interface wins, and the
environment variable is the fallback. That way a fresh checkout runs from the
environment alone, and a user who sets a path in the interface is not
surprised by an environment variable overriding it.

Every write goes through a temporary file and an atomic replace, so a crash
mid-save leaves the previous settings intact rather than a truncated file.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import threading

from .storage import make_directory, settings_file_path

SETTINGS_FILE_NAME = "settings.json"


class Settings:
    """A small, thread-safe, file-backed settings document.

    The document is a mapping of sections; each engine owns one section named
    after it. Reads take a fresh snapshot of the section, so a caller never
    holds a reference into the store.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        """Point the store at its file.

        Args:
            path: Settings file to read and write. Defaults to
                ``<app data>/settings.json``.
        """
        self._path = Path(path) if path is not None else settings_file_path()
        self._lock = threading.Lock()
        self._document: dict = self._read()

    # ------------------------------------------------------------------ #
    # Reading
    # ------------------------------------------------------------------ #

    def _read(self) -> dict:
        """Load the document, tolerating absence and corruption."""
        try:
            loaded = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

        return loaded if isinstance(loaded, dict) else {}

    def snapshot(self) -> dict:
        """Return the whole document.

        Returns:
            dict: A copy-safe snapshot of every saved section.
        """
        with self._lock:
            return json.loads(json.dumps(self._document))

    def section(self, name: str) -> dict:
        """Return one section's saved values.

        Args:
            name: Section name, such as an engine id.

        Returns:
            dict: The section, or an empty mapping when it was never saved.
        """
        with self._lock:
            values = self._document.get(name)

            return dict(values) if isinstance(values, dict) else {}

    # ------------------------------------------------------------------ #
    # Writing
    # ------------------------------------------------------------------ #

    def replace_section(self, name: str, values: dict) -> dict:
        """Replace one section wholesale and persist the document.

        Replacing rather than merging is what lets a field be cleared: an
        empty value removes the saved setting, and the environment fallback
        takes over again.

        Args:
            name: Section name, such as an engine id.
            values: The section's new values.

        Returns:
            dict: The stored section.

        Raises:
            OSError: If the document could not be written.
        """
        stored = {key: value for key, value in values.items() if value not in (None, "", [])}

        with self._lock:
            self._document[name] = stored
            self._write(self._document)

            return dict(stored)

    def _write(self, document: dict) -> None:
        """Write the document through a temporary file and an atomic replace.

        Args:
            document: The whole document to store.

        Raises:
            OSError: If the write or the replace fails.
        """
        make_directory(self._path.parent)

        temporary = self._path.with_name(f"{self._path.name}.tmp")

        temporary.write_text(
            json.dumps(document, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        os.replace(temporary, self._path)


#: The sections a fresh install starts with, so the interface has a shape to
#: render before anything has been saved.
DEFAULT_SECTIONS = ("llama_cpp",)
