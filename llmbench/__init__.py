"""LLMBench — a web app for benchmarking local LLMs across inference engines.

The app is a thin JSON layer over the benchmark core plus the dashboard it
serves. The pieces are layered, each knowing only the layer beneath it:

    core/          the domain: engines, options, planning, the matrix
                   executor, statistics, the verdict, telemetry
    services/      the stateful middle: the single-run worker and the
                   persisted run archive
    api/           the HTTP surface: one blueprint per feature area, plus
                   the JSON envelope every endpoint answers with

Templates and static files stay at the project root so the package holds
only Python.
"""

from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, jsonify

BASE_DIR = Path(__file__).resolve().parent.parent


def create_app() -> Flask:
    """Build the Flask app, wire the services, register every blueprint.

    Returns:
        Flask: Configured application instance.
    """
    app = Flask(
        __name__,
        template_folder=str(BASE_DIR / "templates"),
        static_folder=str(BASE_DIR / "static"),
    )

    from .api.engines import blueprint as engines_api
    from .api.events import blueprint as events_api
    from .api.history import blueprint as history_api
    from .api.pages import blueprint as pages_api
    from .api.runs import blueprint as runs_api
    from .api.settings import blueprint as settings_api
    from .core.engines.llama_cpp import LlamaCppEngine
    from .core.engines.ollama import OllamaEngine
    from .core.engines.registry import EngineRegistry
    from .core.settings import Settings
    from .core.storage import make_directory, runs_directory
    from .services.archive import RunArchive
    from .services.run_manager import RunManager

    # The engines the app can benchmark against, in display order. A value
    # saved through the settings interface wins over the environment, so a
    # path chosen once in the interface survives a restart.
    settings = Settings()
    engines = EngineRegistry([
        OllamaEngine(
            os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
        ),
        LlamaCppEngine(settings),
    ])
    app.config["OLLAMA_HOST"] = engines.get("ollama").base_url

    archive = RunArchive()
    app.extensions["llmbench"] = {
        "archive": archive,
        "engines": engines,
        "run_manager": RunManager(archive=archive, engines=engines),
    }

    # The archive's directory is made eagerly, so the first run does not race
    # its own save to the disk.
    make_directory(runs_directory())

    app.register_blueprint(pages_api)
    app.register_blueprint(engines_api)
    app.register_blueprint(runs_api)
    app.register_blueprint(history_api)
    app.register_blueprint(events_api)
    app.register_blueprint(settings_api)

    @app.errorhandler(Exception)
    def handle_unexpected(error: Exception):
        """Answer any unhandled error with the same envelope every route
        uses."""
        if isinstance(error, (SystemExit, KeyboardInterrupt)):
            raise error

        from werkzeug.exceptions import HTTPException

        if isinstance(error, HTTPException):
            return jsonify({"ok": False, "error": error.description, "data": None}), error.code

        return (
            jsonify({"ok": False, "error": f"{type(error).__name__}: {error}", "data": None}),
            500,
        )

    return app
