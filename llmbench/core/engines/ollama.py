"""The Ollama backend.

Ollama keeps one long-running server that loads and evicts models on demand,
so this engine maps the benchmark core's needs onto that model:

- ``ensure_ready`` preloads the weights with a generate call and leaves the
  model resident (a generous ``keep_alive`` keeps it there for the run);
- ``release`` evicts the model the same way;
- every option in the catalog is request-scoped: Ollama takes generation
  parameters per call, so nothing here needs a server restart.
"""

from __future__ import annotations

from ..daemon_client import DaemonClient, DaemonError
from ..interrupt import StopSignal
from ..server_control import (
    START_WAIT_SECONDS,
    STOP_WAIT_SECONDS,
    bring_server_up,
    shut_server_down,
)
from .base import BenchEngine, EngineError, EngineUnsupported


class OllamaEngine(BenchEngine):
    """Benchmarks models served by an Ollama server."""

    id = "ollama"
    label = "Ollama"

    def __init__(self, base_url: str) -> None:
        """Configure the engine against one Ollama server.

        Args:
            base_url: Root of the Ollama HTTP API, without a trailing path.
        """
        self.base_url = base_url.rstrip("/")
        self._client = DaemonClient(self.base_url)

    # ------------------------------------------------------------------ #
    # Server lifecycle and inventory
    # ------------------------------------------------------------------ #

    def probe(self) -> dict:
        """Report whether Ollama is installed, running, and at which version."""
        return self._client.probe_status()

    def bring_up(self, timeout: float = START_WAIT_SECONDS) -> None:
        """Start the local Ollama server and wait for its API to answer."""
        bring_server_up(self.base_url, timeout)

    def shut_down(self, timeout: float = STOP_WAIT_SECONDS) -> bool:
        """Terminate the local Ollama server and wait for it to go quiet."""
        return shut_server_down(self.base_url, timeout)

    def installed_models(self) -> list[dict]:
        """Return every installed model with its size and details."""
        return self._client.installed_models()

    def loaded_models(self) -> list[dict]:
        """Return the models resident in memory, with their VRAM share."""
        return self._client.loaded_models()

    def unload(self, model: str) -> bool:
        """Unload one model resident in memory."""
        return self._client.evict(model)

    def delete_model(self, model: str) -> None:
        """Delete one installed model from Ollama's storage."""
        self._client.delete_model(model)

    # ------------------------------------------------------------------ #
    # Benchmark interface
    # ------------------------------------------------------------------ #

    def ensure_ready(self, model: str, options: dict, interrupt: StopSignal | None = None) -> None:
        """Preload the model's weights.

        Every Ollama option is request-scoped, so only the weights need
        preparing; the options ride the generate calls.

        Raises:
            EngineError: If loading fails.
        """
        try:
            self._client.preload(model)
        except DaemonError as error:
            raise EngineError(str(error)) from error

    def release(self, model: str) -> None:
        """Evict the model so the next case starts with a clean memory."""
        self._client.evict(model)

    def generate(
        self,
        model: str,
        prompt: str,
        options: dict,
        interrupt: StopSignal | None = None,
    ) -> dict:
        """Stream one generation and return the final measured event."""
        try:
            return self._client.stream_generate(
                model=model,
                prompt=prompt,
                options=options,
                interrupt=interrupt,
            )
        except DaemonError as error:
            raise EngineError(str(error)) from error

    # ------------------------------------------------------------------ #
    # Option catalog
    # ------------------------------------------------------------------ #

    def option_catalog(self) -> list[dict]:
        """Return the Ollama generation options the editor renders."""
        return [
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
                "key": "num_ctx",
                "label": "num_ctx",
                "type": "int",
                "group": "Context and output",
                "min": 1,
                "step": 1,
                "placeholder": "4096",
                "hint": "Context window size in tokens.",
            },
            {
                "key": "num_predict",
                "label": "num_predict",
                "type": "int",
                "group": "Context and output",
                "step": 1,
                "placeholder": "256",
                "hint": "Maximum tokens to generate. -1 means unlimited.",
            },
            {
                "key": "num_keep",
                "label": "num_keep",
                "type": "int",
                "group": "Context and output",
                "min": 0,
                "step": 1,
                "placeholder": "24",
                "hint": "Tokens kept from the prompt when the context overflows.",
            },
            {
                "key": "num_batch",
                "label": "num_batch",
                "type": "int",
                "group": "Context and output",
                "min": 1,
                "step": 1,
                "placeholder": "512",
                "hint": "Prompt batch size. Affects prompt throughput.",
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
                "key": "repeat_last_n",
                "label": "repeat_last_n",
                "type": "int",
                "group": "Repetition",
                "step": 1,
                "placeholder": "64",
                "hint": "How far back the repetition penalty looks.",
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
                "key": "penalize_newline",
                "label": "penalize_newline",
                "type": "bool",
                "group": "Repetition",
                "hint": "Whether newlines are penalised like other tokens.",
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
            {
                "key": "num_gpu",
                "label": "num_gpu",
                "type": "int",
                "group": "Runtime",
                "min": 0,
                "step": 1,
                "placeholder": "99",
                "hint": "Layers offloaded to the GPU. 0 forces CPU only.",
            },
            {
                "key": "num_thread",
                "label": "num_thread",
                "type": "int",
                "group": "Runtime",
                "min": 1,
                "step": 1,
                "placeholder": "8",
                "hint": "CPU threads used for generation.",
            },
            {
                "key": "use_mmap",
                "label": "use_mmap",
                "type": "bool",
                "group": "Runtime",
                "hint": "Memory-map the weights instead of reading them in.",
            },
            {
                "key": "use_mlock",
                "label": "use_mlock",
                "type": "bool",
                "group": "Runtime",
                "hint": "Lock the weights in RAM so they are never swapped out.",
            },
        ]
