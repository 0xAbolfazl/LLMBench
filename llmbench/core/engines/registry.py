"""The named collection of engines the app can benchmark against.

The registry is built once in the app factory and reached everywhere through
it; the worker resolves a run's engine by the ``engine`` id its payload
carries, and the interface answers engine lists and per-engine probes.
"""

from __future__ import annotations

from .base import BenchEngine


class UnknownEngine(KeyError):
    """Raised when a payload names an engine the app does not have."""


class EngineRegistry:
    """The engines configured for this app, by id."""

    def __init__(self, engines: list[BenchEngine]) -> None:
        """Index the given engines by their id.

        Args:
            engines: One configured engine per backend, in display order.
        """
        self._engines: dict[str, BenchEngine] = {}

        for engine in engines:
            self._engines[engine.id] = engine

    def get(self, engine_id: str) -> BenchEngine:
        """Return one engine by id.

        Args:
            engine_id: Identifier as carried in API payloads.

        Returns:
            BenchEngine: The engine registered under that id.

        Raises:
            UnknownEngine: If no engine is registered under the id.
        """
        try:
            return self._engines[engine_id]
        except KeyError:
            raise UnknownEngine(engine_id) from None

    def default_id(self) -> str:
        """Return the id of the first registered engine.

        Returns:
            str: The default engine id.
        """
        return next(iter(self._engines))

    def catalog(self) -> list[dict]:
        """Describe every registered engine for the interface.

        Returns:
            list[dict]: One entry per engine: ``id``, ``label`` and whether
            it is currently ``running``.
        """
        entries: list[dict] = []

        for engine in self._engines.values():
            try:
                running = bool(engine.probe().get("running"))
            except Exception:
                running = False

            entries.append({
                "id": engine.id,
                "label": engine.label,
                "running": running,
                "can_delete_models": engine.can_delete_models,
                "configurable": engine.configurable,
                "requires_server": engine.requires_server,
            })

        return entries

    def __iter__(self):
        return iter(self._engines.values())
