"""Starting and stopping the Ollama server process itself.

The Ollama HTTP API (see :mod:`llmbench.core.daemon_client`) can be *observed*
from any host, but *controlling* the server is a local act: it is this
machine's own ``ollama`` binary that owns the process. These helpers give the
dashboard the process lifecycle — spawn ``ollama serve`` detached and wait for
the API to answer, or kill the process and wait for it to go quiet — so a
machine without a running server can be brought up without leaving the app.

Only the local CLI is involved, so a machine that benchmarks a *remote*
``OLLAMA_HOST`` has no local process to control: the functions report that
rather than pretending, and a missing CLI is answered as "not installed"
rather than as a crash.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time

import requests

from .eventlog import record_event

COMPONENT = "ollama/process"

# How long a start or stop may take before it is declared a failure.
START_WAIT_SECONDS = 15.0
STOP_WAIT_SECONDS = 10.0

# How often the API is probed while waiting for a state change.
PROBE_INTERVAL = 0.25

# Probe timeout for a single "is it up?" check: short, since it runs in a loop.
PROBE_TIMEOUT = 2


def cli_present() -> bool:
    """Return whether the ``ollama`` binary is on this machine's PATH.

    Returns:
        bool: True when the CLI is callable from this process.
    """
    return shutil.which("ollama") is not None


def read_cli_version() -> dict | None:
    """Read the version(s) the local CLI reports about itself.

    ``ollama --version`` prints a client line and, when a server answers, a
    server line; both are captured because a remote ``OLLAMA_HOST`` can run a
    different build than the local binary. Terminal decoration is stripped
    before the lines are parsed.

    Returns:
        dict | None: ``{"client": str | None, "server": str | None}``, or None
        when the CLI is not installed.
    """
    if not cli_present():
        return None

    try:
        result = subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=5,
            # The app may run inside a tool whose stdin carries a protocol
            # stream; a child that inherited it could consume those bytes.
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    from ..api.envelope import strip_terminal_noise

    output = strip_terminal_noise(result.stdout + result.stderr)

    client: str | None = None
    server: str | None = None

    for line in output.splitlines():
        if "client version is" in line:
            client = line.split("client version is", 1)[1].strip()
        elif line.startswith("ollama version is"):
            server = line.split("ollama version is", 1)[1].strip()

    if client is None and server is None:
        return None

    return {"client": client, "server": server}


def server_responds(base_url: str) -> bool:
    """Return whether the Ollama API at ``base_url`` is answering right now.

    Args:
        base_url: Root of the Ollama HTTP API to probe.

    Returns:
        bool: True when ``/api/version`` answers OK.
    """
    try:
        response = requests.get(
            f"{base_url.rstrip('/')}/api/version",
            timeout=(PROBE_TIMEOUT, PROBE_TIMEOUT),
        )
        return response.ok
    except requests.RequestException:
        return False


def _await_state(base_url: str, expected: bool, timeout: float) -> bool:
    """Poll the API until it reaches the expected running state.

    Args:
        base_url: Root of the Ollama HTTP API to poll.
        expected: The state to wait for (True for running, False for stopped).
        timeout: Maximum seconds to wait.

    Returns:
        bool: Whether the expected state was reached in time.
    """
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        if server_responds(base_url) == expected:
            return True

        time.sleep(PROBE_INTERVAL)

    return server_responds(base_url) == expected


def bring_server_up(base_url: str, timeout: float = START_WAIT_SECONDS) -> None:
    """Start the local Ollama server and wait until its API answers.

    The server is spawned detached with no window, then the API is probed
    until it responds — Ollama takes a moment to come up, and a benchmark
    started against a half-started server would fail on its first request.

    Args:
        base_url: Root of the Ollama HTTP API to wait on.
        timeout: Maximum seconds to wait for readiness.

    Raises:
        RuntimeError: If Ollama is not installed, could not be spawned, or did
        not answer within the timeout.
    """
    if not cli_present():
        raise RuntimeError(
            "Ollama is not installed on this machine, so there is no local "
            "server to start."
        )

    if server_responds(base_url):
        record_event(
            level="INFO",
            component=COMPONENT,
            action="start",
            message="Ollama is already running",
        )
        return

    record_event(
        level="INFO",
        component=COMPONENT,
        action="start",
        message="Starting Ollama server",
        details={"base_url": base_url},
    )

    try:
        subprocess.Popen(
            ["ollama", "serve"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=(
                subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            ),
        )
    except Exception as error:
        record_event(
            level="ERROR",
            component=COMPONENT,
            action="start",
            message="Failed to start Ollama",
            details={"error": str(error)},
        )
        raise RuntimeError(f"Failed to start Ollama: {error}") from error

    if not _await_state(base_url, expected=True, timeout=timeout):
        record_event(
            level="ERROR",
            component=COMPONENT,
            action="start",
            message="Ollama did not reach a running state",
            details={"timeout": timeout},
        )
        raise RuntimeError(f"Ollama did not start within {timeout:.0f} seconds")

    record_event(
        level="INFO",
        component=COMPONENT,
        action="start",
        message="Ollama server is running",
    )


def shut_server_down(base_url: str, timeout: float = STOP_WAIT_SECONDS) -> bool:
    """Terminate the local Ollama server and wait until its API goes quiet.

    Args:
        base_url: Root of the Ollama HTTP API to wait on as it goes down.
        timeout: Maximum seconds to wait for the shutdown.

    Returns:
        bool: True when the server was stopped, False when there was nothing
        installed or running to stop.

    Raises:
        RuntimeError: If the process could not be killed, or it was still
        answering after the timeout.
    """
    if not cli_present():
        record_event(
            level="WARNING",
            component=COMPONENT,
            action="stop",
            message="Ollama is not installed; nothing to stop",
        )
        return False

    if not server_responds(base_url):
        record_event(
            level="INFO",
            component=COMPONENT,
            action="stop",
            message="Ollama is not running; nothing to stop",
        )
        return False

    record_event(
        level="INFO",
        component=COMPONENT,
        action="stop",
        message="Stopping Ollama server",
    )

    try:
        if os.name == "nt":
            command = ["taskkill", "/F", "/IM", "ollama.exe"]
        else:
            command = ["pkill", "-TERM", "-x", "ollama"]

        result = subprocess.run(
            command,
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        if result.returncode not in (0, 1):
            raise RuntimeError(f"Failed to stop Ollama (exit code {result.returncode})")
    except Exception as error:
        record_event(
            level="ERROR",
            component=COMPONENT,
            action="stop",
            message="Failed to stop Ollama",
            details={"error": str(error)},
        )
        raise

    if not _await_state(base_url, expected=False, timeout=timeout):
        record_event(
            level="ERROR",
            component=COMPONENT,
            action="stop",
            message="Ollama did not stop",
            details={"timeout": timeout},
        )
        raise RuntimeError(f"Ollama did not stop within {timeout:.0f} seconds")

    record_event(
        level="INFO",
        component=COMPONENT,
        action="stop",
        message="Ollama server stopped",
    )

    return True
