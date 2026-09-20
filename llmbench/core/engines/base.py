"""The interface every inference engine implements.

The benchmark core is engine-agnostic: it plans a matrix of model and
configuration pairs, asks the engine to make one pair ready at a time, and
streams generations from it. An engine's responsibilities, in the order the
core uses them:

- :meth:`BenchEngine.probe` — is the engine installed and running;
- :meth:`BenchEngine.bring_up` / :meth:`BenchEngine.shut_down` — the server
  lifecycle the dashboard's start and stop buttons drive;
- :meth:`BenchEngine.installed_models` / :meth:`BenchEngine.loaded_models` —
  the model inventory the picker lists;
- :meth:`BenchEngine.ensure_ready` — make one model, under one configuration,
  ready to answer (loading weights, spawning or reconfiguring a server);
- :meth:`BenchEngine.generate` — one streamed prompt, with the timing and
  token fields the statistics fold;
- :meth:`BenchEngine.release` — undo what ``ensure_ready`` set up.

Options arrive on two planes, and each engine declares which of its own
option keys is which through :meth:`BenchEngine.option_catalog`:

- *request-scoped* options ride every generate call (sampling settings);
- *session-scoped* options shape the loaded model or server and only take
  effect through ``ensure_ready`` (GPU layer counts, context sizes).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..interrupt import StopSignal


class EngineError(RuntimeError):
    """Raised when an engine fails an operation it was asked to perform."""


class EngineUnsupported(NotImplementedError):
    """Raised when an engine does not implement an optional capability."""


class BenchEngine(ABC):
    """One configured inference backend."""

    #: Short identifier used in the API, in payloads and in saved profiles.
    id: str = ""

    #: Human-readable name shown in the interface.
    label: str = ""

    #: Whether this engine's models can be deleted through the app.
    can_delete_models: bool = True

    #: Whether the interface may edit this engine's paths and address.
    configurable: bool = False

    #: Whether a benchmark needs the engine's server already running. An engine
    #: with one resident server (Ollama) is True; an engine that spawns and
    #: stops a server per model as part of a run (llama.cpp) is False, so its
    #: normal "nothing running" idle state never blocks a run.
    requires_server: bool = True

    @abstractmethod
    def probe(self) -> dict:
        """Report whether the engine is installed, running, and at which
        version.

        Returns:
            dict: At least ``installed``, ``running``, ``version`` and
            ``host``; extra keys are allowed.
        """

    @abstractmethod
    def bring_up(self, timeout: float) -> None:
        """Start the engine's server, if it owns one, and wait for readiness.

        Args:
            timeout: Maximum seconds to wait for readiness.

        Raises:
            EngineError: If the engine could not be brought up.
        """

    @abstractmethod
    def shut_down(self, timeout: float) -> bool:
        """Stop the engine's server and wait for it to go quiet.

        Args:
            timeout: Maximum seconds to wait for the shutdown.

        Returns:
            bool: True when something was actually stopped.
        """

    @abstractmethod
    def installed_models(self) -> list[dict]:
        """Return every model the engine can run.

        Returns:
            list[dict]: One entry per model with at least ``name`` and
            ``size``; ``modified`` and ``details`` where the engine knows
            them.
        """

    @abstractmethod
    def loaded_models(self) -> list[dict]:
        """Return the models currently resident in memory.

        Returns:
            list[dict]: One entry per loaded model with at least ``name``.
        """

    @abstractmethod
    def unload(self, model: str) -> bool:
        """Make a loaded model resident no longer.

        Args:
            model: Model name to unload.

        Returns:
            bool: True when the engine confirmed the unload.
        """

    @abstractmethod
    def delete_model(self, model: str) -> None:
        """Remove an installed model from the engine's storage.

        Args:
            model: Model name to delete.

        Raises:
            EngineUnsupported: If the engine cannot delete models.
            EngineError: If the delete fails.
        """

    @abstractmethod
    def ensure_ready(self, model: str, options: dict, interrupt: StopSignal | None = None) -> None:
        """Make one model, under one configuration, ready to answer.

        Called before a case's first prompt. Whatever loading happens here
        sits outside the benchmark timer. Calling it again for the same model
        and the same session-scoped options should be a no-op; changed
        session-scoped options must take effect before the next prompt.

        Args:
            model: Model name or tag to prepare.
            options: The case's full option set, request- and session-scoped
                alike.
            interrupt: Optional signal consulted during a slow preparation.

        Raises:
            EngineError: If the model could not be prepared.
        """

    @abstractmethod
    def release(self, model: str) -> None:
        """Undo a previous :meth:`ensure_ready` for this model.

        Best-effort by contract: a failed release must never fail the caller.
        It frees VRAM or stops a server so the next case starts clean.

        Args:
            model: Model name that was prepared.
        """

    @abstractmethod
    def generate(
        self,
        model: str,
        prompt: str,
        options: dict,
        interrupt: StopSignal | None = None,
    ) -> dict:
        """Run one prompt and return the measured generation.

        Args:
            model: Model name or tag to generate with.
            prompt: The prompt text.
            options: Request-scoped options for this generation.
            interrupt: Optional signal that stops the generation part-way.

        Returns:
            dict: The final generation event, carrying at least:

                - ``response``: the generated text;
                - ``prompt_eval_count`` / ``eval_count``: token counts;
                - ``prompt_eval_duration`` / ``eval_duration``: durations in
                  nanoseconds, the units Ollama reports;
                - ``ttft_seconds``: measured time to the first streamed
                  content;
                - ``done``: whether generation finished normally.

        Raises:
            EngineError: If the generation fails.
        """

    @abstractmethod
    def option_catalog(self) -> list[dict]:
        """Return the configuration options this engine understands.

        Each entry carries ``key``, ``label``, ``type`` (``int``, ``float``,
        ``bool`` or ``list``), ``group``, and optionally ``min``, ``max``,
        ``step``, ``placeholder``, ``hint`` and ``scope``. ``scope`` is
        ``"session"`` when the option only takes effect through
        :meth:`ensure_ready`, otherwise it is request-scoped.

        Returns:
            list[dict]: The option table the configuration editor renders.
        """

    def identity(self) -> dict:
        """Return the engine facts a saved run's profile should carry.

        Returns:
            dict: Engine ``id``, ``label`` and ``version`` when known.
        """
        status = self.probe()

        return {
            "engine": self.id,
            "label": self.label,
            "version": status.get("version"),
            "host": status.get("host"),
        }
