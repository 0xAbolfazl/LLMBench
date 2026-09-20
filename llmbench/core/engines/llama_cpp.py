"""Running a llama.cpp ``llama-server`` process for the benchmark.

Unlike Ollama, llama.cpp has no resident daemon holding every model: the
engine is one ``llama-server`` process per model, spawned with the model file
and the session-scoped options as command-line flags, and answering the
OpenAI-style native API over HTTP (``/health``, ``/props``, ``/completion``).
So the engine's unit of residency is the whole server process:

- ``ensure_ready`` spawns (or reuses, when model and server options did not
  change) a server for the model and waits for ``/health`` to answer;
- ``release`` stops that server, freeing its VRAM entirely;
- ``generate`` posts one streamed ``/completion`` request; the sampling
  options ride the request, and the server's final event carries the token
  counts and millisecond timings that get folded into the statistics.

The installation and the model library are configured from the environment:

- ``LLAMA_CPP_HOME``   directory holding ``llama-server.exe``;
- ``LLAMA_CPP_MODELS`` directory scanned for ``.gguf`` model files;
- ``LLAMA_CPP_HOST``   the address the spawned server listens on.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time

import requests

from ..interrupt import POLL_SECONDS, StopSignal
from ..eventlog import record_event
from ..settings import Settings
from .base import BenchEngine, EngineError, EngineUnsupported

COMPONENT = "engine/llama-cpp"

# The options that shape the loaded model and must be on the server command
# line; everything else in a configuration is sampling, sent per request.
# Each entry maps the option key to the server flag.
SESSION_FLAGS = {
    "n_gpu_layers": "--n-gpu-layers",
    "n_ctx": "--ctx-size",
    "n_batch": "--batch-size",
    "n_ubatch": "--ubatch-size",
    "n_threads": "--threads",
    "n_threads_batch": "--threads-batch",
    "flash_attn": "--flash-attn",
    "no_mmap": "--no-mmap",
    "mlock": "--mlock",
}

# How long a spawned server may take to load its model before it is declared
# failed. Large models offloading to a GPU can legitimately take a while.
START_TIMEOUT = 180.0

# Probe timeout for a single health check: short, since it runs in a loop.
PROBE_TIMEOUT = 2

DEFAULT_HOST = "http://127.0.0.1:8082"

# A version line looks like: version: 0.4.1-dev (build 11056, commit ...)
_VERSION_PATTERN = re.compile(r"version:\s*(\S+)\s+\(build\s+(\d+)")


def _version_tuple(value: str):
    """Parse a dotted version string into comparable parts.

    Args:
        value: Dotted version such as ``0.4.1-dev``.

    Returns:
        tuple: Numeric parts, for comparing two versions.
    """
    return tuple(int(part) for part in re.findall(r"\d+", value)[:3])


class LlamaCppEngine(BenchEngine):
    """Benchmarks models served by a llama.cpp ``llama-server`` process."""

    id = "llama-cpp"
    label = "llama.cpp"
    can_delete_models = False
    configurable = True
    #: The server is spawned per model inside a run, so none must be pre-started.
    requires_server = False

    #: The settings section this engine reads its paths from.
    SETTINGS_SECTION = "llama_cpp"

    def __init__(self, settings: Settings) -> None:
        """Configure the engine from the saved settings.

        Args:
            settings: The persisted settings store; the engine reads its own
                section on every access, so a path saved through the interface
                takes effect without restarting the app.
        """
        self._settings = settings
        self._process: subprocess.Popen | None = None
        self._serving: str | None = None
        self._session_args: dict | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Configuration: saved settings first, environment second
    # ------------------------------------------------------------------ #

    def _saved(self) -> dict:
        """Return this engine's saved settings section."""
        return self._settings.section(self.SETTINGS_SECTION)

    @property
    def home(self) -> str:
        """Directory holding ``llama-server``, saved or from the environment."""
        return str(self._saved().get("home") or os.environ.get("LLAMA_CPP_HOME") or "")

    @property
    def model_dirs(self) -> list[str]:
        """Directories scanned for ``.gguf`` files, saved or from the environment.

        ``LLAMA_CPP_MODELS`` may name several directories separated by the
        platform's path separator.
        """
        saved = self._saved().get("model_dirs")

        if isinstance(saved, list) and saved:
            return [str(entry) for entry in saved if str(entry).strip()]

        raw = os.environ.get("LLAMA_CPP_MODELS") or ""

        return [part for part in raw.split(os.pathsep) if part.strip()]

    @property
    def base_url(self) -> str:
        """The address the spawned server listens on."""
        saved = str(self._saved().get("host") or "")

        return (saved or os.environ.get("LLAMA_CPP_HOST") or DEFAULT_HOST).rstrip("/")

    def config_view(self) -> dict:
        """Describe the engine's configuration for the interface.

        Returns:
            dict: The saved and effective values, where the executable was
            found, and whether each side came from the environment, so the
            station can show what is actually in force.
        """
        executable = self.server_executable()

        return {
            "home": self.home or None,
            "model_dirs": self.model_dirs,
            "host": self.base_url,
            "executable": executable,
            "installed": executable is not None,
            "version": self._read_version(executable) if executable else None,
            "env_home": os.environ.get("LLAMA_CPP_HOME") or None,
            "env_model_dirs": [
                part
                for part in (os.environ.get("LLAMA_CPP_MODELS") or "").split(os.pathsep)
                if part.strip()
            ],
            "env_host": os.environ.get("LLAMA_CPP_HOST") or None,
        }

    def configure(self, patch: dict) -> dict:
        """Validate and save a configuration patch.

        A path is accepted only when it is what it claims to be: a folder that
        exists for the home, and folders that exist for the model library. An
        empty value clears the saved setting, which hands the decision back to
        the environment variable.

        Args:
            patch: ``home``, ``model_dirs`` and/or ``host`` values to save.

        Returns:
            dict: The configuration view after saving.

        Raises:
            EngineError: If a supplied path does not exist or the home holds no
                llama-server executable.
        """
        current = self._saved()
        updated = dict(current)

        if "home" in patch:
            home = str(patch.get("home") or "").strip()

            if home:
                if not os.path.isdir(home):
                    raise EngineError(f"There is no folder at {home}.")

                if not self._find_executable(home):
                    raise EngineError(
                        f"No llama-server executable was found in {home}. "
                        f"Pick the folder that holds llama-server.exe."
                    )

            updated["home"] = home

        if "model_dirs" in patch:
            raw = patch.get("model_dirs")

            if isinstance(raw, str):
                entries = [part.strip() for part in raw.splitlines()]
            elif isinstance(raw, list):
                entries = [str(part).strip() for part in raw]
            else:
                raise EngineError("model_dirs must be a list of folders.")

            entries = [entry for entry in entries if entry]

            for entry in entries:
                if not os.path.isdir(entry):
                    raise EngineError(f"There is no folder at {entry}.")

            updated["model_dirs"] = entries

        if "host" in patch:
            host = str(patch.get("host") or "").strip().rstrip("/")

            if host and not host.startswith(("http://", "https://")):
                raise EngineError("The host must start with http:// or https://.")

            updated["host"] = host

        self._settings.replace_section(self.SETTINGS_SECTION, updated)

        return self.config_view()

    # ------------------------------------------------------------------ #
    # Installation facts
    # ------------------------------------------------------------------ #

    @staticmethod
    def _find_executable(home: str) -> str | None:
        """Return the llama-server executable inside a folder, if there.

        Args:
            home: Folder to look in.

        Returns:
            str | None: Path to the executable, or None when it is not there.
        """
        for name in ("llama-server.exe", "llama-server"):
            candidate = os.path.join(home, name)

            if os.path.isfile(candidate):
                return candidate

        return None

    def server_executable(self) -> str | None:
        """Return the llama-server executable's path, or None.

        The configured home is tried first, so a saved folder wins over
        whatever happens to be on the PATH.

        Returns:
            str | None: Path to ``llama-server.exe`` (or ``llama-server``
            elsewhere), when the configured home holds it or it is on PATH.
        """
        if self.home:
            found = self._find_executable(self.home)

            if found:
                return found

        return shutil.which("llama-server")

    # ------------------------------------------------------------------ #
    # Model library: a recursive scan of the configured folders
    # ------------------------------------------------------------------ #

    @staticmethod
    def _is_continuation_shard(name: str) -> bool:
        """Tell whether a gguf file is a later part of a split model.

        A model split across files is named ``...-00001-of-00003.gguf`` and
        up. Only the first part is a model a user picks; the rest are pulled
        in by llama.cpp itself.

        Args:
            name: gguf file name.

        Returns:
            bool: True for any part other than the first.
        """
        match = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", name, re.IGNORECASE)

        return bool(match) and match.group(1) != "00001"

    def _scan_models(self) -> list[dict]:
        """Scan every configured folder — and its subfolders — for gguf files.

        Returns:
            list[dict]: One entry per model: ``name`` (file name), ``path``,
            ``size``, ``modified`` and ``details`` carrying the folder it was
            found under.
        """
        models: list[dict] = []
        seen: set[str] = set()

        for folder in self.model_dirs:
            if not os.path.isdir(folder):
                continue

            for root, _dirs, files in os.walk(folder):
                for name in files:
                    if not name.lower().endswith(".gguf"):
                        continue

                    if self._is_continuation_shard(name):
                        continue

                    path = os.path.join(root, name)

                    if path in seen:
                        continue

                    seen.add(path)

                    try:
                        stat = os.stat(path)
                    except OSError:
                        continue

                    try:
                        relative = os.path.relpath(root, folder)
                    except ValueError:
                        relative = ""

                    models.append({
                        "name": name,
                        "path": path,
                        "size": stat.st_size,
                        "modified": int(stat.st_mtime),
                        "details": {
                            "folder": folder,
                            "relative_folder": "" if relative == "." else relative,
                        },
                    })

        models.sort(key=lambda entry: entry["name"].lower())

        return models

    def _model_path(self, model: str) -> str | None:
        """Resolve a model name to a gguf file in the model library.

        The name is the file's own name; a bare stem matches the ``.gguf`` it
        names, and a path relative to a configured folder is honoured, so the
        interface can speak in short names while the engine spawns with full
        paths. A model given as an absolute path is taken as-is.

        Args:
            model: Model name, file stem, relative or absolute path.

        Returns:
            str | None: The file path, or None when it cannot be resolved.
        """
        candidate = model.strip()

        if os.path.isfile(candidate):
            return candidate

        for folder in self.model_dirs:
            for suffix in (".gguf", ""):
                path = os.path.join(folder, candidate + suffix)

                if os.path.isfile(path):
                    return path

        # A name without its folder — "Qwen3-14B-Q4_K_M.gguf" for a file one
        # level down — still resolves, as long as it is unique in the library.
        matches = [
            entry["path"]
            for entry in self._scan_models()
            if entry["name"] == candidate
            or entry["name"] == f"{candidate}.gguf"
        ]

        if len(matches) == 1:
            return matches[0]

        return None

    # ------------------------------------------------------------------ #
    # Server lifecycle and inventory
    # ------------------------------------------------------------------ #

    def _health_ok(self) -> bool:
        """Return whether the server's ``/health`` answers ready right now."""
        try:
            response = requests.get(
                f"{self.base_url}/health",
                timeout=(PROBE_TIMEOUT, PROBE_TIMEOUT),
            )
            return response.ok
        except requests.RequestException:
            return False

    def probe(self) -> dict:
        """Report whether llama.cpp is installed and its server is running.

        The version is read once from the executable, since the server may
        not be up to ask over HTTP.

        Returns:
            dict: ``installed``, ``running``, ``version``, ``host``, the
            configured ``home`` and ``model_dirs``, and how many models the
            library scan currently finds.
        """
        executable = self.server_executable()

        version = None

        if executable:
            version = self._read_version(executable)

        return {
            "installed": executable is not None,
            "running": self._health_ok(),
            "version": version,
            "host": self.base_url,
            "home": self.home or None,
            "executable": executable,
            "model_dirs": self.model_dirs,
            "model_count": len(self._scan_models()),
        }

    def _read_version(self, executable: str) -> str | None:
        """Read the build version the server executable reports.

        Args:
            executable: Path to the llama-server executable.

        Returns:
            str | None: Version such as ``0.4.1-dev (build 11056)``, or None
            when it could not be read.
        """
        try:
            result = subprocess.run(
                [executable, "--version"],
                capture_output=True,
                text=True,
                errors="replace",
                check=False,
                timeout=10,
                stdin=subprocess.DEVNULL,
            )
        except (OSError, subprocess.SubprocessError):
            return None

        match = _VERSION_PATTERN.search((result.stdout or "") + (result.stderr or ""))

        if not match:
            return None

        return f"{match.group(1)} (build {match.group(2)})"

    def bring_up(self, timeout: float = START_TIMEOUT) -> None:
        """Nothing to bring up without a model.

        A llama.cpp server exists per model, so there is no model-independent
        server to start; the benchmark's ``ensure_ready`` spawns it.

        Raises:
            EngineError: Always — the caller is told to start a run instead.
        """
        raise EngineError(
            "llama.cpp has no server to start on its own: it is started per "
            "model when a benchmark run needs it."
        )

    def shut_down(self, timeout: float = 10.0) -> bool:
        """Stop the spawned server, if one is running.

        Args:
            timeout: Maximum seconds to wait for the process to exit.

        Returns:
            bool: True when a server was actually stopped.
        """
        with self._lock:
            return self._stop_process(timeout)

    def installed_models(self) -> list[dict]:
        """Return every model the configured folders hold, recursively.

        Returns:
            list[dict]: One entry per model: ``name`` (file name), ``path``,
            ``size`` (bytes), ``modified`` (epoch seconds) and ``details``
            carrying the folder it was found under.

        Raises:
            EngineError: If no model folder is configured at all.
        """
        if not self.model_dirs:
            raise EngineError(
                "No llama.cpp model folder is configured. Set one in the "
                "engine station, or set LLAMA_CPP_MODELS."
            )

        return self._scan_models()

    def loaded_models(self) -> list[dict]:
        """Return the model the spawned server is currently serving.

        Returns:
            list[dict]: One entry when the server is up and serving, with
            ``name`` and the reported ``size_vram`` when available.
        """
        if not self._health_ok():
            return []

        try:
            response = requests.get(
                f"{self.base_url}/props",
                timeout=(PROBE_TIMEOUT, PROBE_TIMEOUT),
            )
            response.raise_for_status()
            props = response.json()
        except (requests.RequestException, ValueError):
            return []

        default_generation = props.get("default_generation_settings") or {}
        model_path = (
            props.get("model_path")
            or default_generation.get("model")
            or self._serving
        )

        if not model_path:
            return []

        entry = {
            "name": os.path.basename(str(model_path)),
            "size_vram": default_generation.get("model_vram")
            or props.get("model_vram"),
            "expires_at": None,
        }

        return [entry]

    def unload(self, model: str) -> bool:
        """Stop the server, which is what unloading means here.

        Args:
            model: Model name; the server is stopped whatever it serves.

        Returns:
            bool: True when a server was stopped.
        """
        return self.shut_down()

    def delete_model(self, model: str) -> None:
        """Not supported: llama.cpp models are plain files this app does not
        own.

        Raises:
            EngineUnsupported: Always.
        """
        raise EngineUnsupported(
            "llama.cpp models are plain files; remove them from "
            "LLAMA_CPP_MODELS yourself."
        )

    # ------------------------------------------------------------------ #
    # Benchmark interface
    # ------------------------------------------------------------------ #

    def _session_options(self, options: dict) -> dict:
        """Split a configuration into its session-scoped part.

        Args:
            options: The case's full option set.

        Returns:
            dict: The options that must be on the server command line.
        """
        return {
            key: options[key]
            for key in SESSION_FLAGS
            if key in options
        }

    def _command(self, model_path: str, session_options: dict) -> list[str]:
        """Build the server command line for one model and configuration.

        Args:
            model_path: Path to the gguf file to serve.
            session_options: Session-scoped options, already validated.

        Returns:
            list[str]: The argv to spawn.
        """
        from urllib.parse import urlparse

        parsed = urlparse(self.base_url)
        port = parsed.port or 8082

        command = [
            self.server_executable(),
            "-m", model_path,
            "--host", parsed.hostname or "127.0.0.1",
            "--port", str(port),
        ]

        for key, flag in SESSION_FLAGS.items():
            if key not in session_options:
                continue

            value = session_options[key]

            if isinstance(value, bool):
                # Boolean flags exist only in their affirmative form.
                if value:
                    command.append(flag)
                continue

            command.extend([flag, str(value)])

        return command

    def _stop_process(self, timeout: float = 10.0) -> bool:
        """Terminate the spawned server, if any, and wait for it to exit.

        Args:
            timeout: Maximum seconds to wait for the process to exit.

        Returns:
            bool: True when a process was actually stopped.
        """
        if self._process is None:
            return False

        process = self._process
        self._process = None
        self._serving = None
        self._session_args = None

        if process.poll() is not None:
            return False

        process.terminate()

        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()

        record_event(
            level="INFO",
            component=COMPONENT,
            action="stop",
            message="llama-server stopped",
        )

        return True

    def _await_health(self, timeout: float) -> None:
        """Poll ``/health`` until the server answers ready.

        Args:
            timeout: Maximum seconds to wait.

        Raises:
            EngineError: If the server never became ready, or died first.
        """
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            if self._process is not None and self._process.poll() is not None:
                raise EngineError(
                    "llama-server exited during startup (exit code "
                    f"{self._process.returncode})"
                )

            if self._health_ok():
                return

            time.sleep(0.25)

        raise EngineError(
            f"llama-server did not become ready within {timeout:.0f} seconds"
        )

    def ensure_ready(self, model: str, options: dict, interrupt: StopSignal | None = None) -> None:
        """Spawn — or reuse — a server for this model and configuration.

        A server that is already serving the same model under the same
        session-scoped options is reused untouched; anything else stops the
        old server first, because llama.cpp serves one model per process and
        session options only take effect at startup.

        Args:
            model: Model name, stem or gguf path to serve.
            options: The case's full option set; its session-scoped part
                becomes server flags.
            interrupt: Optional signal consulted while waiting for startup.

        Raises:
            EngineError: If the model or the executable cannot be found, or
            the server fails to become ready.
        """
        model_path = self._model_path(model)

        if not model_path:
            raise EngineError(
                f"Model '{model}' was not found. Set LLAMA_CPP_MODELS to the "
                "directory holding your .gguf files, or pass a full path."
            )

        session_options = self._session_options(options)

        with self._lock:
            if (
                self._process is not None
                and self._process.poll() is None
                and self._serving == model_path
                and self._session_args == session_options
                and self._health_ok()
            ):
                # Same model, same flags, already answering: keep it.
                return

            self._stop_process()

            executable = self.server_executable()

            if not executable:
                raise EngineError(
                    "llama-server was not found. Set LLAMA_CPP_HOME to the "
                    "directory holding it."
                )

            command = self._command(model_path, session_options)

            record_event(
                level="INFO",
                component=COMPONENT,
                action="start",
                message="Starting llama-server",
                details={"model": os.path.basename(model_path)},
            )

            try:
                self._process = subprocess.Popen(
                    command,
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
                    message="Failed to start llama-server",
                    details={"error": str(error)},
                )
                raise EngineError(f"Failed to start llama-server: {error}") from error

            self._serving = model_path
            self._session_args = session_options

        try:
            self._await_health(START_TIMEOUT)
        except EngineError:
            self._stop_process()
            raise

        record_event(
            level="INFO",
            component=COMPONENT,
            action="start",
            message="llama-server is ready",
            details={"model": os.path.basename(model_path)},
        )

    def release(self, model: str) -> None:
        """Stop the server, freeing the VRAM the served model holds."""
        with self._lock:
            self._stop_process()

    def generate(
        self,
        model: str,
        prompt: str,
        options: dict,
        interrupt: StopSignal | None = None,
    ) -> dict:
        """Stream one ``/completion`` request and return the measured event.

        Args:
            model: Model name; the running server already serves it.
            prompt: The prompt text.
            options: Request-scoped options (sampling) for this generation.
            interrupt: Optional signal that stops the generation part-way.

        Returns:
            dict: The final event in the Ollama field names the statistics
            expect: ``response``, ``prompt_eval_count``, ``eval_count``,
            ``prompt_eval_duration``, ``eval_duration``, ``ttft_seconds``
            and ``done``.

        Raises:
            EngineError: If the request fails or the stream breaks.
        """
        signal = interrupt if interrupt is not None else StopSignal()

        signal.raise_if_stopped()

        payload: dict = {"prompt": prompt, "stream": True}
        payload.update({
            key: value
            for key, value in options.items()
            if key not in SESSION_FLAGS
        })

        request_started = time.perf_counter()

        try:
            response = requests.post(
                f"{self.base_url}/completion",
                json=payload,
                stream=True,
                timeout=(10, 600),
            )
            response.raise_for_status()
        except requests.RequestException as error:
            raise EngineError(f"Failed to run model: {error}") from error

        # Cancellation closes the socket out from under the reader, from a
        # watcher thread, exactly as the Ollama client does.
        from ..daemon_client import _StopWatcher

        watcher = _StopWatcher(response=response, signal=signal)
        watcher.start()

        chunks: list[str] = []
        final: dict = {}
        ttft: float | None = None

        try:
            for line in response.iter_lines():
                signal.raise_if_stopped()

                if not line:
                    continue

                # The stream is server-sent events: JSON payloads prefixed
                # with "data: ", separated by blank lines and keep-alives.
                text = line.decode("utf-8")

                if text.startswith("data:"):
                    text = text[len("data:"):].lstrip()
                else:
                    continue

                try:
                    event = json.loads(text)
                except json.JSONDecodeError as error:
                    raise EngineError("llama-server returned invalid JSON") from error

                if "error" in event:
                    raise EngineError(event["error"])

                content = event.get("content")
                if content:
                    if ttft is None:
                        ttft = time.perf_counter() - request_started

                    chunks.append(content)

                if event.get("timings") or any(
                    event.get(key) for key in event if key.startswith("stopped_")
                ):
                    final = event
        except EngineError:
            raise
        except Exception as error:
            # The socket may have been closed by a cancellation; consult the
            # signal before believing the error.
            signal.raise_if_stopped()
            raise EngineError(f"Failed to run model: {error}") from error
        finally:
            watcher.stop()
            try:
                response.close()
            except Exception:
                pass

        if not final:
            signal.raise_if_stopped()
            raise EngineError("llama-server closed the stream before finishing")

        timings = final.get("timings") or {}
        prompt_ms = timings.get("prompt_ms") or 0
        predicted_ms = timings.get("predicted_ms") or 0

        result = {
            "response": "".join(chunks),
            # The statistics read Ollama's field names, so the counts and the
            # nanosecond durations are carried under them.
            "prompt_eval_count": timings.get("prompt_n", 0),
            "eval_count": timings.get("predicted_n", 0),
            "prompt_eval_duration": int(prompt_ms * 1_000_000),
            "eval_duration": int(predicted_ms * 1_000_000),
            "ttft_seconds": ttft,
            "done": True,
        }

        return result

    # ------------------------------------------------------------------ #
    # Option catalog
    # ------------------------------------------------------------------ #

    def option_catalog(self) -> list[dict]:
        """Return the llama.cpp options the editor renders.

        Session-scoped entries become server flags and only take effect when
        the server (re)starts; the rest ride every completion request.
        """
        return [
            {
                "key": "n_gpu_layers",
                "label": "n_gpu_layers",
                "type": "int",
                "group": "Session",
                "min": 0,
                "step": 1,
                "placeholder": "999",
                "scope": "session",
                "hint": "Model layers offloaded to the GPU. 0 forces CPU only; a "
                    "large number offloads everything that fits.",
            },
            {
                "key": "n_ctx",
                "label": "n_ctx",
                "type": "int",
                "group": "Session",
                "min": 1,
                "step": 1,
                "placeholder": "4096",
                "scope": "session",
                "hint": "Context window size in tokens.",
            },
            {
                "key": "n_batch",
                "label": "n_batch",
                "type": "int",
                "group": "Session",
                "min": 1,
                "step": 1,
                "placeholder": "2048",
                "scope": "session",
                "hint": "Logical prompt batch size.",
            },
            {
                "key": "n_ubatch",
                "label": "n_ubatch",
                "type": "int",
                "group": "Session",
                "min": 1,
                "step": 1,
                "placeholder": "512",
                "scope": "session",
                "hint": "Physical prompt batch size.",
            },
            {
                "key": "n_threads",
                "label": "n_threads",
                "type": "int",
                "group": "Session",
                "min": 1,
                "step": 1,
                "placeholder": "",
                "scope": "session",
                "hint": "CPU threads used for generation. Blank uses the server default.",
            },
            {
                "key": "n_threads_batch",
                "label": "n_threads_batch",
                "type": "int",
                "group": "Session",
                "min": 1,
                "step": 1,
                "placeholder": "",
                "scope": "session",
                "hint": "CPU threads used for prompt processing.",
            },
            {
                "key": "flash_attn",
                "label": "flash_attn",
                "type": "bool",
                "group": "Session",
                "scope": "session",
                "hint": "Enable flash attention.",
            },
            {
                "key": "no_mmap",
                "label": "no_mmap",
                "type": "bool",
                "group": "Session",
                "scope": "session",
                "hint": "Do not memory-map the weights; read them in fully.",
            },
            {
                "key": "mlock",
                "label": "mlock",
                "type": "bool",
                "group": "Session",
                "scope": "session",
                "hint": "Lock the weights in RAM so they are never swapped out.",
            },
            {
                "key": "temperature",
                "label": "temperature",
                "type": "float",
                "group": "Sampling",
                "min": 0,
                "max": 2,
                "step": 0.05,
                "placeholder": "0.8",
                "hint": "Higher values make the output more random.",
            },
            {
                "key": "top_p",
                "label": "top_p",
                "type": "float",
                "group": "Sampling",
                "min": 0,
                "max": 1,
                "step": 0.05,
                "placeholder": "0.9",
                "hint": "Nucleus sampling cutoff.",
            },
            {
                "key": "top_k",
                "label": "top_k",
                "type": "int",
                "group": "Sampling",
                "min": 1,
                "step": 1,
                "placeholder": "40",
                "hint": "How many candidate tokens to consider.",
            },
            {
                "key": "min_p",
                "label": "min_p",
                "type": "float",
                "group": "Sampling",
                "min": 0,
                "max": 1,
                "step": 0.01,
                "placeholder": "0.05",
                "hint": "Minimum probability relative to the best token.",
            },
            {
                "key": "typical_p",
                "label": "typical_p",
                "type": "float",
                "group": "Sampling",
                "min": 0,
                "max": 1,
                "step": 0.05,
                "placeholder": "1.0",
                "hint": "Locally typical sampling cutoff.",
            },
            {
                "key": "seed",
                "label": "seed",
                "type": "int",
                "group": "Sampling",
                "step": 1,
                "placeholder": "42",
                "hint": "Fixing the seed makes a run repeatable.",
            },
            {
                "key": "n_predict",
                "label": "n_predict",
                "type": "int",
                "group": "Context and output",
                "step": 1,
                "placeholder": "256",
                "hint": "Maximum tokens to generate. -1 means unlimited.",
            },
            {
                "key": "stop",
                "label": "stop",
                "type": "list",
                "group": "Context and output",
                "placeholder": "</s>, User:",
                "hint": "Comma-separated stop sequences.",
            },
            {
                "key": "repeat_penalty",
                "label": "repeat_penalty",
                "type": "float",
                "group": "Repetition",
                "min": 0,
                "step": 0.05,
                "placeholder": "1.1",
                "hint": "Penalty applied to tokens already seen.",
            },
            {
                "key": "presence_penalty",
                "label": "presence_penalty",
                "type": "float",
                "group": "Repetition",
                "step": 0.05,
                "placeholder": "0",
                "hint": "Flat penalty for tokens that already appeared.",
            },
            {
                "key": "frequency_penalty",
                "label": "frequency_penalty",
                "type": "float",
                "group": "Repetition",
                "step": 0.05,
                "placeholder": "0",
                "hint": "Penalty scaled by how often a token appeared.",
            },
            {
                "key": "mirostat",
                "label": "mirostat",
                "type": "int",
                "group": "Mirostat",
                "min": 0,
                "max": 2,
                "step": 1,
                "placeholder": "0",
                "hint": "0 disables it, 1 is Mirostat, 2 is Mirostat 2.0.",
            },
            {
                "key": "mirostat_tau",
                "label": "mirostat_tau",
                "type": "float",
                "group": "Mirostat",
                "min": 0,
                "step": 0.5,
                "placeholder": "5.0",
                "hint": "Target entropy. Lower is more focused.",
            },
            {
                "key": "mirostat_eta",
                "label": "mirostat_eta",
                "type": "float",
                "group": "Mirostat",
                "min": 0,
                "step": 0.01,
                "placeholder": "0.1",
                "hint": "How fast Mirostat adapts.",
            },
        ]
