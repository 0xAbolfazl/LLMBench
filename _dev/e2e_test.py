"""End-to-end check for LLMBench against the mock Ollama server.

Exercises the full pipeline: the Ollama station endpoints (status, model
catalog, resident models), a run start, polling it to completion, verifying
the progress accounting, the significance halves, the history save, the export
endpoints and the execution log. Exits non-zero on the first failed check.

Runs self-contained: it starts the mock Ollama (port 11599) and the app
(port 5099, pointed at the mock) in a throwaway data directory, unless both
are already answering on those ports. It never touches a real Ollama server
or the user's saved history.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import time

import requests

MOCK_PORT = 11599
MOCK_LLAMA_PORT = 11598
APP_PORT = 5099
MOCK_URL = f"http://127.0.0.1:{MOCK_PORT}"
MOCK_LLAMA_URL = f"http://127.0.0.1:{MOCK_LLAMA_PORT}"
BASE = f"http://127.0.0.1:{APP_PORT}"
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)


def check(label, condition, extra=""):
    if not condition:
        print(f"FAIL: {label} {extra}")
        sys.exit(1)
    print(f"ok:   {label} {extra}")


def wait_for_done(runs):
    for _ in range(120):
        job = requests.get(f"{BASE}/api/benchmark/status").json()["data"]
        runs[-1] = job
        if job and job["status"] != "running":
            return job
        time.sleep(0.5)
    raise AssertionError("run did not finish in time")


def _alive(url):
    try:
        requests.get(url, timeout=0.5)
        return True
    except requests.RequestException:
        return False


def _wait_for(url, what):
    for _ in range(100):
        if _alive(url):
            return
        time.sleep(0.2)
    raise SystemExit(f"{what} did not come up on {url}")


def start_fleet():
    """Start whatever the checks need that is not already answering.

    Returns:
        list[tuple[str, subprocess.Popen, str|None]]: (name, process,
        data_directory) for everything started, in start order.
    """
    started = []

    if not _alive(f"{MOCK_URL}/api/version"):
        mock = subprocess.Popen(
            [sys.executable, os.path.join(HERE, "mock_ollama.py"), str(MOCK_PORT)]
        )
        started.append(("mock Ollama", mock, None))
        _wait_for(f"{MOCK_URL}/api/version", "mock Ollama")

    if not _alive(f"{MOCK_LLAMA_URL}/health"):
        mock_llama = subprocess.Popen(
            [sys.executable, os.path.join(HERE, "mock_llama_server.py"), str(MOCK_LLAMA_PORT)]
        )
        started.append(("mock llama-server", mock_llama, None))
        _wait_for(f"{MOCK_LLAMA_URL}/health", "mock llama-server")

    if not _alive(f"{BASE}/api/ollama/status"):
        # A throwaway LOCALAPPDATA keeps the run's history and log out of the
        # user's real store, so the checks hold on every run, not only the
        # first. The llama.cpp engine is pointed at a throwaway home (with a
        # dummy executable, enough for "installed") and the mock server, so
        # its HTTP translation is tested without the real binary.
        data_dir = tempfile.mkdtemp(prefix="llmbench-e2e-")
        llama_home = os.path.join(data_dir, "llama-home")
        llama_models = os.path.join(data_dir, "llama-models")
        os.makedirs(llama_home)
        os.makedirs(llama_models)
        open(os.path.join(llama_home, "llama-server.exe"), "wb").close()
        open(os.path.join(llama_models, "mock-q4_0.gguf"), "wb").close()
        env = dict(
            os.environ,
            OLLAMA_HOST=MOCK_URL,
            LOCALAPPDATA=data_dir,
            LLAMA_CPP_HOME=llama_home,
            LLAMA_CPP_MODELS=llama_models,
            LLAMA_CPP_HOST=MOCK_LLAMA_URL,
        )

        app = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "from llmbench import create_app; "
                f"create_app().run(host='127.0.0.1', port={APP_PORT}, "
                "threaded=True, use_reloader=False)",
            ],
            env=env,
            cwd=PROJECT_ROOT,
        )
        started.append(("LLMBench app", app, data_dir))
        _wait_for(f"{BASE}/api/ollama/status", "LLMBench app")

    return started


def stop_fleet(started):
    for name, process, data_dir in reversed(started):
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        print(f"stopped: {name}")
        if data_dir:
            shutil.rmtree(data_dir, ignore_errors=True)


def main():
    # 0. The engines: the registry and the per-engine endpoints the panel reads.
    engines = requests.get(f"{BASE}/api/engines").json()
    check("engines endpoint answers", engines["ok"], f"got {engines.get('error')}")
    check(
        "both engines registered",
        [e["id"] for e in engines["data"]] == ["ollama", "llama-cpp"],
        f"got {[e['id'] for e in engines['data']]}",
    )

    status = requests.get(f"{BASE}/api/engines/ollama/status").json()
    check("ollama status answers", status["ok"], f"got {status.get('error')}")
    check("mock server reported running", status["data"]["running"] is True)
    check(
        "status carries the mock version",
        status["data"]["version"] == "0.0.0-mock",
        f"got {status['data']['version']}",
    )
    check("status names the host", status["data"]["host"] == MOCK_URL)

    models = requests.get(f"{BASE}/api/engines/ollama/models").json()
    check("models endpoint answers", models["ok"], f"got {models.get('error')}")
    check(
        "both mock models listed",
        [m["name"] for m in models["data"]] == ["mock-llama:latest", "mock-llama2:latest"],
        f"got {[m['name'] for m in models['data']]}",
    )
    check(
        "model entries carry the detail fields",
        all(set(m) == {"name", "size", "modified", "details"} for m in models["data"]),
    )

    running = requests.get(f"{BASE}/api/engines/ollama/models/running").json()
    check("running endpoint answers", running["ok"], f"got {running.get('error')}")
    check("nothing resident before a run", running["data"] == [], f"got {running['data']}")

    llama_status = requests.get(f"{BASE}/api/engines/llama-cpp/status").json()
    check("llama.cpp status answers", llama_status["ok"], f"got {llama_status.get('error')}")
    check("llama.cpp installed via throwaway home", llama_status["data"]["installed"] is True)
    check("llama.cpp sees the mock server", llama_status["data"]["running"] is True)

    llama_models = requests.get(f"{BASE}/api/engines/llama-cpp/models").json()
    check(
        "llama.cpp lists the gguf library",
        [m["name"] for m in llama_models["data"]] == ["mock-q4_0.gguf"],
        f"got {[m['name'] for m in llama_models['data']]}",
    )

    llama_running = requests.get(f"{BASE}/api/engines/llama-cpp/models/running").json()
    check(
        "llama.cpp reports the served model",
        llama_running["ok"] and llama_running["data"][0]["name"] == "mock-q4_0.gguf",
        f"got {llama_running['data']}",
    )

    schema = requests.get(f"{BASE}/api/benchmark/schema", params={"engine": "llama-cpp"}).json()
    check("llama.cpp schema answers", schema["ok"], f"got {schema.get('error')}")
    gpu_layers = next(o for o in schema["data"]["options"] if o["key"] == "n_gpu_layers")
    check("llama.cpp catalog carries the GPU layer knob",
          gpu_layers.get("scope") == "session", f"got {gpu_layers}")

    # The llama.cpp engine's HTTP translation, against the mock server. The
    # engine reads its address from settings, so hand it a throwaway store
    # pointed at the mock.
    sys.path.insert(0, PROJECT_ROOT)
    from llmbench.core.engines.llama_cpp import LlamaCppEngine
    from llmbench.core.settings import Settings
    engine_settings = Settings(path=os.path.join(tempfile.mkdtemp(), "settings.json"))
    engine_settings.replace_section("llama_cpp", {"host": MOCK_LLAMA_URL})
    llama_engine = LlamaCppEngine(engine_settings)
    generation = llama_engine.generate("mock-q4_0.gguf", "Say hello.", {})
    check(
        "llama.cpp generation translates timings",
        generation["eval_count"] == 80
        and generation["prompt_eval_count"] == 64
        and generation["eval_duration"] == 4_000_000_000
        and generation["ttft_seconds"] is not None
        and generation["response"].startswith("Token 1"),
        f"got { {k: generation[k] for k in ('eval_count', 'prompt_eval_count', 'eval_duration', 'ttft_seconds')} }",
    )

    # 1. Cross-model run: two models, one prompt, two repetitions.
    response = requests.post(
        f"{BASE}/api/benchmark/run",
        json={
            "models": ["mock-llama:latest", "mock-llama2:latest"],
            "prompts": "Say hello.",
            "repetitions": 2,
        },
    )
    body = response.json()
    check("run starts", response.ok and body["ok"], f"status={response.status_code}")
    check(
        "planned runs = 2 models x 1 prompt x 2 reps",
        body["data"]["planned_runs"] == 4,
        f"got {body['data']['planned_runs']}",
    )

    runs = [None]
    job = wait_for_done(runs)
    check("run finished", job["status"] == "done", f"status={job['status']} err={job.get('error')}")
    check("progress reached 100%", job["progress"]["percent"] == 100.0, f"got {job['progress']}")
    check("steps_done == steps_total == 4", job["progress"]["steps_total"] == 4 and job["progress"]["steps_done"] == 4)

    result = job["result"]
    check("two tests in the result", len(result["tests"]) == 2)
    check("models echo", result["models"] == ["mock-llama:latest", "mock-llama2:latest"])

    test = result["tests"][0]
    check("per-prompt repetitions averaged", test["results"][0]["repetitions"] == 2)
    check(
        "faster model measures 20 tokens/s",
        test["summary"]["average_output_tokens_per_second"] == 20.0,
        f"got {test['summary']['average_output_tokens_per_second']}",
    )
    check(
        "slower model measures 16 tokens/s",
        result["tests"][1]["summary"]["average_output_tokens_per_second"] == 16.0,
        f"got {result['tests'][1]['summary']['average_output_tokens_per_second']}",
    )
    check(
        "noise measured with 2 reps",
        test["summary"]["output_tokens_per_second_stddev"] is not None,
        f"got {test['summary']['output_tokens_per_second_stddev']}",
    )

    sig = result["significance"]
    check("across_models verdict present", isinstance(sig.get("across_models"), dict), f"got {sig.get('across_models')}")
    check("cross-model verdict names models", sig["across_models"]["leader"] in result["models"])
    # The mock repeats each model's timing exactly, so a 4 tokens/s gap with
    # zero noise must be judged a real difference, with the faster model on
    # top.
    check(
        "cross-model gap judged real",
        sig["across_models"]["significant"] is True
        and sig["across_models"]["leader"] == "mock-llama:latest"
        and "the difference is real" in sig["across_models"]["message"],
        f"got {sig['across_models']['message']}",
    )

    # 2. The finished run must be in the history.
    history = requests.get(f"{BASE}/api/history").json()["data"]
    check("history holds the run", len(history) == 1, f"got {len(history)}")
    record = requests.get(f"{BASE}/api/history/{history[0]['id']}").json()["data"]
    check("history record loads in full", record["id"] == history[0]["id"] and "result" in record)
    check(
        "history winner names the faster model",
        record["summary"]["winner"] == "mock-llama:latest",
        f"got {record['summary']['winner']!r}",
    )
    check("history record names its engine", record.get("engine") == "ollama", f"got {record.get('engine')}")
    check(
        "history record carries the machine profile",
        isinstance(record.get("profile"), dict) and record["profile"].get("captured_at"),
        f"got {record.get('profile')}",
    )

    # 3. The export endpoints.
    csv_response = requests.post(
        f"{BASE}/api/benchmark/results-csv",
        json={"id": job["id"], "result": result},
    )
    check("csv export", csv_response.ok and "output tok/s" in csv_response.text, f"status={csv_response.status_code}")
    json_response = requests.post(
        f"{BASE}/api/benchmark/results-json",
        json={"id": job["id"], "result": result},
    )
    check("json export", json_response.ok and len(json_response.json()["tests"]) == 2)

    # 4. The execution log must carry the run's steps.
    logs = requests.get(f"{BASE}/api/logs").json()["data"]
    check("log has entries", len(logs["entries"]) > 0, f"got {len(logs['entries'])}")
    actions = {entry["action"] for entry in logs["entries"]}
    check("log recorded the comparison", "compare" in actions and "case" in actions, f"actions={actions}")
    errors = [entry for entry in logs["entries"] if entry["level"] == "ERROR"]
    check("no errors in the log", not errors, f"errors={errors}")

    # 5. Single-model configuration run: the by_model half of the verdict.
    requests.post(f"{BASE}/api/benchmark/clear").json()
    response = requests.post(
        f"{BASE}/api/benchmark/run",
        json={
            "models": ["mock-llama:latest"],
            "prompts": "Say hello.",
            "repetitions": 1,
            "configurations": [
                {"name": "hot", "options": {"temperature": "1.0"}},
                {"name": "cool", "options": {"temperature": "0.2"}},
            ],
        },
    )
    body = response.json()
    check("config run starts", response.ok and body["ok"])
    check(
        "config run plan = 2 configs x 1 prompt",
        body["data"]["planned_runs"] == 2,
        f"got {body['data']['planned_runs']}",
    )

    runs = [None]
    job = wait_for_done(runs)
    check("config run finished", job["status"] == "done", f"status={job['status']} err={job.get('error')}")
    result = job["result"]
    check("one-model job names the model", job["model"] == "mock-llama:latest")
    check("two configuration tests", [t["name"] for t in result["tests"]] == ["hot", "cool"])
    sig = result["significance"]["by_model"]["mock-llama:latest"]
    check("by_model verdict present", sig is not None)
    # Both configurations run the same model, so they measure identically and
    # the verdict must be a dead heat, not a winner.
    check(
        "identical configs reported as a tie",
        sig["significant"] is False and "dead heat" in sig["message"],
        f"got {sig['message']}",
    )
    # The dead heat must show as no winner in the history, unlike the
    # cross-model run above.
    config_record = (
        requests.get(f"{BASE}/api/history").json()["data"][0]
    )
    check(
        "config-run history winner is a dead heat",
        config_record["winner"] is None,
        f"got {config_record['winner']!r}",
    )

    # 6. Single-rep cross-model run: a visible gap with no repetition data
    #    must report itself as unmeasured rather than guess.
    requests.post(f"{BASE}/api/benchmark/clear").json()
    requests.post(
        f"{BASE}/api/benchmark/run",
        json={
            "models": ["mock-llama:latest", "mock-llama2:latest"],
            "prompts": "Say hello.",
            "repetitions": 1,
        },
    )
    runs = [None]
    job = wait_for_done(runs)
    check("single-rep run finished", job["status"] == "done", f"status={job['status']} err={job.get('error')}")
    sig = job["result"]["significance"]["across_models"]
    check(
        "single-rep verdict reports noise unmeasured",
        sig["significant"] is None and "noise cannot be told apart" in sig["message"],
        f"got {sig['message']}",
    )

    # 7. Cancellation of a new run: it must settle into cancelled. The "slow"
    #    word stretches the mock's generation so the run is still in flight
    #    when the cancel lands.
    requests.post(f"{BASE}/api/benchmark/run", json={
        "models": ["mock-llama:latest"],
        "prompts": "Say hello. slow",
        "repetitions": 1,
    })
    time.sleep(0.4)
    cancelled = requests.post(f"{BASE}/api/benchmark/cancel").json()
    check("cancel accepted", cancelled["ok"])
    runs = [None]
    job = wait_for_done(runs)
    check("run settled cancelled", job["status"] == "cancelled", f"got {job['status']}")

    print("\nall e2e checks passed")


if __name__ == "__main__":
    started = start_fleet()
    try:
        main()
    finally:
        stop_fleet(started)
