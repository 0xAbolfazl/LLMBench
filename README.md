[![en](https://img.shields.io/badge/lang-en-red.svg)](README.md) [![fa](https://img.shields.io/badge/lang-fa-blue.svg)](README.fa.md)

## What LLMBench is

LLMBench is a benchmarking system for local language models. It runs your models
over a set of prompts and configurations so that comparison is possible, records
the timings, and — most importantly — judges them: is the difference you see in
the output a real difference, or just the jitter of any single measurement?

Runs are supported on two engines (runtimes):

- **Ollama** — models are pulled by a tag and served by one long-running server.
- **llama.cpp** — models are `.gguf` files on your disk. A separate server
  (`llama-server`) is brought up for each model and shut down when the work is
  done, so nothing stays up forever. All that is asked of you is that the files
  exist on the machine and the paths are set.

![App overview](images/overview.png)
*App overview: the top bar, the stations, and the Setup section side by side.*

Every run is stored **with its profile**: which engine, at which version, on
which machine (including the GPU identifier). That way, any number you see in
the history months later can be traced back to the exact engine, version and
settings that produced it.

## Four comparison modes

What keeps LLMBench simple is that you do not pick a "mode"; it is worked out
automatically from what you put in. There are four main modes:

1. **Single-model test** — one model, no dedicated configuration. You just want
   to know how fast this model is, right here, right now.
2. **Comparing configurations** — one model, several configurations. For example
   the same model with 999 GPU layers versus 32, or with and without
   flash-attention. This is really a test of the configurations themselves, and
   it tells you which settings work better for that model.
3. **Comparing models** — several models, one setting. When you want to know
   which model is faster for this use.
4. **Tournament mode** — several models and several configurations. Each model is
   paired with the configuration it will compete under; the most complete form of
   comparison.

### How you do it

All of these modes start from the same single form; only the models and
configurations you choose differ:

- Tick the models from the list (one = mode 1 or 2, several = mode 3 or 4).
- Write the prompts (each prompt in its own block, separated from the rest by a
  blank line).
- Create a configuration or import a configuration file, if needed.
- Set the number of repetitions (default 1) and press Run.

**Before starting, the engine must be available:**

- For **Ollama**: Ollama must be installed on the machine and its server up
  first (or started from within the app itself).
- For **llama.cpp**: no service needs to be installed; the `llama-server` files
  and the `.gguf` models just need to be on the machine, and you set the
  installation folder and the model folder(s) in the Engine station. The app
  brings up the server for each model on its own.

![Engine Tab](images/engines.png)
*The Engine tab: the engine chooser, the path-settings block, the status cards,
and the list of models found.*

![Engine Tab](images/add-cnf.png)
*Add Configuration Page*

## What is measured

Each measurement records a single prompt from start to finish, and several
numbers come out of it:

- **Prompt (prefill) speed** — how many input tokens are processed per second.
- **Generation speed** — how many output tokens are produced per second. This is
  the headline "model speed" number.
- **Time to First Token (TTFT)** — how long it takes before the first output
  appears.
- **Total duration of each measurement** — the whole time of that single run, in
  seconds.
- In addition, every measurement takes a snapshot of the device: **VRAM used**,
  **GPU temperature** and **GPU clock** — so you can see what the model is doing
  on the hardware while it works.

### Repetition and noise detection

A single measurement is not reliable; each prompt can be run several times (the
repetition count). The result is the mean of those runs, and their **standard
deviation** is also recorded as a measure of spread — the lower the standard
deviation, the more confident you can be in the number.

More than that, LLMBench gives a **significance judgment**: it compares the
difference between the two top entries against the measured noise (the standard
deviation) and says:

- "The difference is real — the leader is X faster than the other, and it clears
  the noise", or
- "The difference is within the noise; you can't say which is actually better",
  or
- honestly, when each input has only been measured once: "to judge, increase the
  repetition count."

https://github.com/user-attachments/assets/ad4e82bb-2029-4a42-b33b-74792f2715a7

*The output of a run: the comparison table with the speed columns, the
significance judgment, and the device readings.*

## Install and run

Requirements: Python 3.10 or higher, and one (or both) of the engines above.
The NVIDIA driver with `nvidia-smi` on PATH is optional — if it is missing, the
GPU columns stay empty and the rest of the measurements continue without problem.

```bash
pip install -r requirements.txt
python app.py
```

Then open <http://127.0.0.1:5000>. To set the paths you can either enter them in
the **Engine station** inside the app (they are stored in `settings.json` and
take priority over the environment variables), or use environment variables:

| Variable | Engine | Meaning |
| --- | --- | --- |
| `OLLAMA_HOST` | Ollama | API address (default `http://127.0.0.1:11434`) |
| `LLAMA_CPP_HOME` | llama.cpp | The folder containing `llama-server.exe` |
| `LLAMA_CPP_MODELS` | llama.cpp | Model folder(s) — scanned recursively for `.gguf` files |
| `LLAMA_CPP_HOST` | llama.cpp | Address of the server the app brings up (default `http://127.0.0.1:8082`) |

## History

Every run is automatically recorded in the history (newest on top, capped at 100
entries) and stays with its profile — so you can trace any number back to the
machine, engine and settings that produced it.

---

---

The idea for LLMBench emerged during the design and development of [ModelSetupHub](https://github.com/ModelSetupHub/ModelSetupHub). It was the result of collaboration and discussions between [Parsa Safaie](https://github.com/parsasafaie) and me, with parts of the original idea and several features of LLMBench also shaped by our shared suggestions and designs.

LLMBench is an independent project with a more specific focus on benchmarking, measurement reproducibility, and the analysis of statistical significance.

---
