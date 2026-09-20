"""A mock Ollama server for end-to-end testing of LLMBench.

Implements just the endpoints the app talks to — /api/version, /api/tags,
/api/ps and /api/generate (streamed) — with canned timing numbers, so a full
benchmark run can be exercised on a machine that has no model pulled.

Run with: python mock_ollama.py [port]
"""

import json
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

MODELS = ["mock-llama:latest", "mock-llama2:latest"]

# Deterministic per-model output timing: the two models measure at different
# rates (20 vs 16 tokens/s) so a cross-model run has a real gap for the
# significance logic to judge, while two configurations of one model measure
# identically — a dead heat.
TIMINGS = {
    "mock-llama:latest": {"eval_count": 80, "eval_duration": 4_000_000_000},
    "mock-llama2:latest": {"eval_count": 80, "eval_duration": 5_000_000_000},
}


class Handler(BaseHTTPRequestHandler):
    """Answers the four Ollama endpoints with deterministic data."""

    def _send(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/version":
            self._send({"version": "0.0.0-mock"})
        elif self.path == "/api/tags":
            self._send({"models": [{"name": name} for name in MODELS]})
        elif self.path == "/api/ps":
            # Nothing resident: the client will warm every model itself.
            self._send({"models": []})
        else:
            self._send({"error": "not found"}, 404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        request = json.loads(self.rfile.read(length) or b"{}")

        if self.path != "/api/generate":
            self._send({"error": "not found"}, 404)
            return

        prompt = request.get("prompt", "")

        if prompt == "":
            # A warm-up: load the (mock) weights and produce nothing.
            self._send(
                {
                    "model": request["model"],
                    "done": True,
                    "prompt_eval_count": 0,
                    "eval_count": 0,
                    "prompt_eval_duration": 0,
                    "eval_duration": 0,
                }
            )
            return

        # A streamed answer: a few content chunks, then the final event with
        # the timing fields Ollama reports. The delays make the wall-clock
        # duration non-zero so a rate can be computed. A prompt containing
        # "slow" stretches the delays so a cancellation can be tested mid-run.
        slow = "slow" in prompt
        pause = 0.5 if slow else 0.02
        chunks = [f"Token {i} " for i in range(1, 4)]
        timing = TIMINGS.get(request["model"], TIMINGS[MODELS[0]])
        final = {
            "model": request["model"],
            "response": "",
            "done": True,
            "prompt_eval_count": 64,
            "eval_count": timing["eval_count"],
            "prompt_eval_duration": 320_000_000,
            "eval_duration": timing["eval_duration"],
        }

        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()

        for chunk in chunks:
            self.wfile.write(
                (json.dumps({"response": chunk, "done": False}) + "\n").encode("utf-8")
            )
            self.wfile.flush()
            time.sleep(pause)

        self.wfile.write((json.dumps(final) + "\n").encode("utf-8"))
        self.wfile.flush()

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 11599
    print(f"mock ollama on http://127.0.0.1:{port}")
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()
