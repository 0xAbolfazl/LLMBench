"""A client for a running Ollama server.

Everything the app talks to Ollama for goes through :class:`DaemonClient`.
The server is addressed over its HTTP API (``/api/version``, ``/api/tags``,
``/api/ps`` and ``/api/generate``), which is cleaner and more reliable than
parsing CLI tables. Even unloading a model — which has no dedicated endpoint —
is a generate call with a ``keep_alive`` of zero, so every operation reaches
the configured server directly, including a remote one.

The base URL is configurable so the app can benchmark a server on another host
or port: pass it in, or set the ``OLLAMA_HOST`` environment variable.
"""

from __future__ import annotations

import json
import shutil
import threading
import time

import requests

from .interrupt import POLL_SECONDS, RunInterrupted, StopSignal

# How long a single generate may wait between bytes, in seconds. Model loading
# happens in ``preload`` before a timed run, so this bounds a slow first token
# rather than a load.
GENERATE_READ_TIMEOUT = 600
CONNECT_TIMEOUT = 10


class DaemonError(RuntimeError):
    """Raised when Ollama answers with an error or is unreachable in a way
    that matters to the operation in flight."""


class DaemonClient:
    """A thin, stateless client for one Ollama server."""

    def __init__(self, base_url: str = "http://127.0.0.1:11434") -> None:
        """Point the client at a server.

        Args:
            base_url: Root of the Ollama HTTP API, without a trailing path.
        """
        self.base_url = base_url.rstrip("/")

    # ------------------------------------------------------------------ #
    # Server health and model inventory
    # ------------------------------------------------------------------ #

    def probe_status(self) -> dict:
        """Report whether Ollama is installed, running, and at which version.

        The server version comes from the API itself; the client version is
        read from the local CLI, which on a remote-``OLLAMA_HOST`` setup can
        run a different build than the server it benchmarks.

        Returns:
            dict: ``installed`` (server reachable or CLI present),
            ``running`` (the HTTP API answered), ``version`` (server version),
            ``client_version`` (``{"client", "server"}`` from the local CLI,
            or None), and ``host`` (the configured API root).
        """
        from .server_control import read_cli_version

        try:
            response = requests.get(
                f"{self.base_url}/api/version",
                timeout=(CONNECT_TIMEOUT, CONNECT_TIMEOUT),
            )

            if response.ok:
                return {
                    "installed": True,
                    "running": True,
                    "version": response.json().get("version"),
                    "client_version": read_cli_version(),
                    "host": self.base_url,
                }
        except requests.RequestException:
            pass

        installed = shutil.which("ollama") is not None

        return {
            "installed": installed,
            "running": False,
            "version": None,
            "client_version": read_cli_version() if installed else None,
            "host": self.base_url,
        }

    def installed_models(self) -> list[dict]:
        """Return every installed model with its size and architecture fields.

        The server reports more than a name for each model — the size it
        occupies on disk, when it was last modified, and a ``details`` mapping
        with the architecture (family, parameter count, quantisation level).
        Fields the server does not report for a model come back as None.

        Returns:
            list[dict]: One entry per model: ``name``, ``size`` (bytes),
            ``modified`` (epoch seconds) and ``details`` (mapping).

        Raises:
            DaemonError: If the server is unreachable.
        """
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=(CONNECT_TIMEOUT, CONNECT_TIMEOUT),
            )
            response.raise_for_status()
        except requests.RequestException as error:
            raise DaemonError(f"Failed to list models: {error}") from error

        models: list[dict] = []

        for item in response.json().get("models", []):
            if not isinstance(item, dict) or not item.get("name"):
                continue

            models.append({
                "name": item["name"],
                "size": item.get("size"),
                "modified": item.get("modified"),
                "details": item.get("details") or {},
            })

        return models

    def loaded_model_names(self) -> list[str]:
        """Return the names of the models currently resident in memory.

        Returns:
            list[str]: Loaded model names, in the server's order.

        Raises:
            DaemonError: If the server is unreachable.
        """
        try:
            response = requests.get(
                f"{self.base_url}/api/ps",
                timeout=(CONNECT_TIMEOUT, CONNECT_TIMEOUT),
            )
            response.raise_for_status()
        except requests.RequestException as error:
            raise DaemonError(f"Failed to list running models: {error}") from error

        loaded = response.json().get("models", [])

        return [item["name"] for item in loaded if isinstance(item, dict) and item.get("name")]

    def loaded_models(self) -> list[dict]:
        """Return the models resident in memory, with what they occupy.

        Returns:
            list[dict]: One entry per loaded model: ``name``, ``size_vram``
            (bytes resident) and ``expires_at`` (epoch seconds).

        Raises:
            DaemonError: If the server is unreachable.
        """
        try:
            response = requests.get(
                f"{self.base_url}/api/ps",
                timeout=(CONNECT_TIMEOUT, CONNECT_TIMEOUT),
            )
            response.raise_for_status()
        except requests.RequestException as error:
            raise DaemonError(f"Failed to list running models: {error}") from error

        loaded: list[dict] = []

        for item in response.json().get("models", []):
            if not isinstance(item, dict) or not item.get("name"):
                continue

            loaded.append({
                "name": item["name"],
                "size_vram": item.get("size_vram"),
                "expires_at": item.get("expires_at"),
            })

        return loaded

    def delete_model(self, model: str) -> None:
        """Delete a model from Ollama's storage.

        Done over the HTTP API (the server keeps the weights, not this
        machine's CLI), so it works for a remote ``OLLAMA_HOST`` as well. The
        name must be the model's full name and tag as the server reports it.

        Args:
            model: Model name and tag to delete.

        Raises:
            DaemonError: If the delete is refused or the server is unreachable.
        """
        try:
            response = requests.post(
                f"{self.base_url}/api/delete",
                json={"name": model},
                timeout=(CONNECT_TIMEOUT, 60),
            )
            response.raise_for_status()
        except requests.RequestException as error:
            raise DaemonError(f"Failed to remove model {model}: {error}") from error

    # ------------------------------------------------------------------ #
    # Model residency
    # ------------------------------------------------------------------ #

    def _is_resident(self, model: str) -> bool:
        """Return whether a model is already resident in memory.

        Ollama reports loaded models with an explicit tag, so a bare name such
        as "llama3" must also match the ":latest" form it was loaded under.

        Args:
            model: Model name or tag.

        Returns:
            bool: True when the model is already loaded.
        """
        for name in self.loaded_model_names():
            if name == model or (":" not in model and name == f"{model}:latest"):
                return True

        return False

    def preload(self, model: str, keep_alive: str = "30m") -> None:
        """Load a model into memory if it is not already resident.

        A generation with an empty prompt makes Ollama pull the weights into
        memory and return without producing any output. ``keep_alive`` is set
        generously so a model warmed once stays resident for the whole run.

        Args:
            model: Model name or tag to load.
            keep_alive: Duration string to keep the model loaded (e.g. '30m').

        Raises:
            DaemonError: If loading fails.
        """
        if self._is_resident(model):
            return

        payload = {"model": model, "prompt": "", "stream": False, "keep_alive": keep_alive}

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=(CONNECT_TIMEOUT, GENERATE_READ_TIMEOUT),
            )
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as error:
            raise DaemonError(f"Failed to load model: {model}") from error

        if isinstance(data, dict) and "error" in data:
            raise DaemonError(data["error"])

    def evict(self, model: str) -> bool:
        """Unload a model from memory.

        Done over the HTTP API with a ``keep_alive`` of zero, which tells the
        server to drop the model at once. This reaches the configured server
        directly, so it works for a remote ``OLLAMA_HOST`` where a local
        ``ollama stop`` would address the wrong machine. A failure is treated
        as a note rather than a stop: the model may not have loaded yet, or
        Ollama may already be gone, and either way the next load proceeds
        regardless.

        Args:
            model: Model name or tag to unload.

        Returns:
            bool: True when the server confirmed the unload.
        """
        payload = {"model": model, "prompt": "", "stream": False, "keep_alive": 0}

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=(CONNECT_TIMEOUT, CONNECT_TIMEOUT),
            )
            response.raise_for_status()
            data = response.json()
        except (requests.RequestException, ValueError):
            return False

        return not (isinstance(data, dict) and "error" in data)

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #

    def stream_generate(
        self,
        model: str,
        prompt: str,
        options: dict | None = None,
        interrupt: StopSignal | None = None,
    ) -> dict:
        """Run one temporary model configuration against a single prompt.

        The configuration options are applied only to this specific request.
        The request is streamed so an interruption can take effect
        mid-generation: with a single buffered response there is no point
        between sending the prompt and receiving the whole answer at which the
        operation could stop. The final streamed object carries the same timing
        and token fields a buffered response would, so the returned dictionary
        is unchanged.

        Args:
            model: Ollama model identifier tag.
            prompt: Text prompt string to evaluate.
            options: Optional generation parameters (e.g. temperature, num_ctx).
            interrupt: Optional signal that stops the generation part-way.

        Returns:
            dict: The final streamed event, with the generated text collected
            under 'response' and the time-to-first-token under 'ttft_seconds'
            (None when nothing was generated).

        Raises:
            DaemonError: If the connection fails or Ollama returns an error.
            RunInterrupted: If the signal is tripped during generation.
        """
        signal = interrupt if interrupt is not None else StopSignal()

        signal.raise_if_stopped()

        payload = {"model": model, "prompt": prompt, "stream": True}

        if options:
            payload["options"] = options

        # The time-to-first-token clock starts when the request is sent: the
        # delay a caller experiences covers the connection, the prompt's
        # processing and the first token's generation alike.
        request_started = time.perf_counter()

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                stream=True,
                timeout=(CONNECT_TIMEOUT, GENERATE_READ_TIMEOUT),
            )
            response.raise_for_status()
        except requests.RequestException as error:
            raise DaemonError(f"Failed to run model: {error}") from error

        # Closing the response is what actually interrupts a generation in
        # progress: the reader below is blocked in a socket read that no flag
        # check can reach, so the close is done from a watcher thread and
        # surfaces here as a read failure, which the signal check just after
        # turns into a clean stop.
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

                try:
                    event = json.loads(line.decode("utf-8"))
                except json.JSONDecodeError as error:
                    raise DaemonError("Ollama returned invalid JSON") from error

                if "error" in event:
                    raise DaemonError(event["error"])

                text = event.get("response")
                if text:
                    if ttft is None:
                        # First content of the answer: this is the latency a
                        # reader of a streamed response actually perceives.
                        ttft = time.perf_counter() - request_started

                    chunks.append(text)

                if event.get("done"):
                    final = event
        except RunInterrupted:
            raise
        except DaemonError:
            # Already a described failure — an error payload or malformed JSON.
            raise
        except Exception as error:
            # Interrupting a generation means closing the socket out from under
            # this reader, and what that surfaces as depends on how far the
            # response had got. So the signal is consulted before the error is
            # believed — otherwise a cancellation would be recorded as a failed
            # prompt.
            signal.raise_if_stopped()
            raise DaemonError(f"Failed to run model: {error}") from error
        finally:
            watcher.stop()
            try:
                response.close()
            except Exception:
                pass

        if not final:
            signal.raise_if_stopped()
            raise DaemonError("Ollama closed the stream before finishing")

        final["response"] = "".join(chunks)
        # A generation that produced no content has no first token to time.
        final["ttft_seconds"] = ttft

        return final


class _StopWatcher:
    """Closes an open HTTP response as soon as a stop signal is tripped."""

    def __init__(self, response, signal: StopSignal) -> None:
        """Prepare a watcher for one response.

        Args:
            response: Open HTTP response to close on interruption.
            signal: Signal to watch.
        """
        self._response = response
        self._signal = signal
        self._consumed = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Begin watching the response."""
        self._thread = threading.Thread(
            target=self._watch,
            name="ollama-generate-cancel",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop watching once the response has been consumed."""
        self._consumed.set()

    def _watch(self) -> None:
        """Close the response when interrupted, or exit when it is consumed."""
        # Waits on the signal rather than polling it, so an interruption
        # closes the socket at once; the interval only bounds how long the
        # watcher takes to notice that the response was consumed and it can
        # retire.
        while not self._consumed.is_set():
            if self._signal.wait(POLL_SECONDS):
                try:
                    self._response.close()
                except Exception:
                    pass
                return
