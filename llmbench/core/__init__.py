"""Domain layer of LLMBench — everything that is not HTTP or Flask.

The modules are split by responsibility rather than by the original code's
file layout:

    limits.py          the hard caps a run request must respect
    options.py         the option catalog and input normalisation
    plan.py            run planning: a request becomes the matrix to execute
    daemon_client.py   the Ollama HTTP client (status, models, generation)
    server_control.py  starting and stopping the local Ollama process
    interrupt.py       cooperative stop signals for long-running operations
    telemetry.py       nvidia-smi samples (VRAM, temperature, clock)
    suite.py           one benchmark case: a model under one configuration
    matrix.py          the matrix executor over a whole plan
    stats.py           the statistics: folding, averaging, spread
    verdict.py         the significance judgement on a finished matrix
    archive.py         the persisted run store (services layer)
    run_manager.py     the single background run worker (services layer)
    eventlog.py        the execution event log
    storage.py         every filesystem location the app uses
"""
