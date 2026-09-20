"""The engine abstraction: pluggable inference backends.

An engine is anything that can answer prompts for a benchmark: a local
Ollama server, a llama.cpp ``llama-server`` process, and later others. The
rest of the core (suite, matrix, worker) talks to the interface defined in
:mod:`.base` and never to a backend directly, so adding an engine is adding
one module here — nothing else in the app changes.

    base.py       the BenchEngine interface every backend implements
    ollama.py     the Ollama backend
    llama_cpp.py  the llama.cpp backend
    registry.py   the named collection of configured engines
"""
