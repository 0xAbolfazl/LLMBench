"""LLMBench — application entrypoint.

Run with ``python app.py``. Application wiring lives in the ``llmbench``
package; this file only builds and starts it.
"""

from llmbench import create_app

app = create_app()


if __name__ == "__main__":
    # threaded=True so a poll can answer while a run is being set up; the run
    # itself works on its own background thread.
    app.run(debug=True, port=5000, threaded=True)
