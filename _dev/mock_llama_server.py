"""A mock llama.cpp server for end-to-end testing of LLMBench.

Implements just the endpoints the app's llama.cpp engine talks to —
/health, /props and /completion (streamed) — with canned timing numbers, so
the engine's HTTP translation can be exercised on a machine that never
spawns the real llama-server.

Run with: python mock_llama_server.py [port]
"""

import json
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

# Deterministic timing, mirroring the Ollama mock's rates so a cross-engine
# comparison has comparable numbers.
PROMPT_TOKENS = 64
PREDICTED_TOKENS = 80
PROMPT_MS = 320.0
PREDICTED_MS = 4000.0


class Handler(BaseHTTPRequestHandler):
    """Answers the three llama-server endpoints with deterministic data."""

    def _send(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send({"status": "ok"})
        elif self.path == "/props":
            self._send({
                "model_path": "C:/models/mock-q4_0.gguf",
                "default_generation_settings": {
                    "model": "C:/models/mock-q4_0.gguf",
                    "model_vram": 4_000_000_000,
                },
            })
        else:
            self._send({"error": "not found"}, 404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        request = json.loads(self.rfile.read(length) or b"{}")

        if self.path != "/completion":
            self._send({"error": "not found"}, 404)
            return

        prompt = request.get("prompt", "")

        # A prompt containing "slow" stretches the delays so a cancellation
        # can be tested mid-generation.
        pause = 0.5 if "slow" in prompt else 0.02
        chunks = [f"Token {i} " for i in range(1, 4)]

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()

        # Server-sent events, the format the real llama-server answers with.
        for chunk in chunks:
            self.wfile.write(
                ("data: "
                 + json.dumps({"content": chunk, "stopped_eos": False})
                 + "\n\n").encode("utf-8")
            )
            self.wfile.flush()
            time.sleep(pause)

        self.wfile.write((
            "data: "
            + json.dumps({
                "content": "",
                "stopped_eos": True,
                "stopped_word": False,
                "timings": {
                    "prompt_n": PROMPT_TOKENS,
                    "predicted_n": PREDICTED_TOKENS,
                    "prompt_ms": PROMPT_MS,
                    "predicted_ms": PREDICTED_MS,
                },
            })
            + "\n\n"
        ).encode("utf-8"))
        self.wfile.flush()

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 11598
    print(f"mock llama-server on http://127.0.0.1:{port}")
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()
