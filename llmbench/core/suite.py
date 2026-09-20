"""One benchmark case: a single model under one configuration, measured.

A case is one cell of the run matrix. :func:`measure_case` runs it: each
prompt executes ``repetitions`` times, the repetitions fold into per-prompt
means with their spread, and the prompts fold again into the case summary.
Alongside the timing metrics, machine readings are reported as taken: the
time to the answer's first streamed token, the VRAM the driver reported right
after the generation finished, and the hottest GPU's temperature and clock in
that same moment.

The model is preloaded once, before the first prompt — loading is not part of
the measured time. One prompt list per case is what makes the numbers
comparable: the caller guarantees every case answers the same prompts.
"""

from __future__ import annotations

from collections.abc import Callable
import time

from .engines.base import BenchEngine
from .eventlog import record_event
from .interrupt import RunInterrupted, StopSignal, record_interruption
from .stats import aggregate_case, fold_repetitions, tokens_per_second
from .telemetry import sample_thermals, sample_vram

COMPONENT = "bench.case"


def _emit_progress(
    on_progress: Callable[[dict], None],
    phase: str,
    prompt_index: int,
    prompt_count: int,
    repetition: int,
    repetition_count: int,
    completed: int,
) -> None:
    """Hand one progress step to a case's callback, absorbing its failures.

    A progress callback exists so a caller can show the user where a long run
    is; a bug in it must not be able to end a measurement that has already
    been running for minutes. Anything it raises is logged and the run
    continues.

    Args:
        on_progress: The caller's callback.
        phase: Either 'prompt_start' (a prompt is about to run) or
            'repetition_done' (one repetition just finished).
        prompt_index: 1-based prompt position within the case.
        prompt_count: Prompts the case will run.
        repetition: 1-based repetition that just finished, or 0 on
            'prompt_start'.
        repetition_count: Repetitions each prompt runs.
        completed: Repetitions finished so far in this case.
    """
    try:
        on_progress({
            "phase": phase,
            "prompt_index": prompt_index,
            "prompt_count": prompt_count,
            "repetition": repetition,
            "repetition_count": repetition_count,
            "completed": completed,
        })
    except Exception as error:
        record_event(
            level="WARNING",
            component=COMPONENT,
            action="progress",
            message="Progress callback failed",
            details={
                "phase": phase,
                "prompt_index": prompt_index,
                "repetition": repetition,
                "error": str(error),
            },
        )


def _machine_readings(row: dict) -> None:
    """Attach the GPU readings taken right after one generation.

    Args:
        row: The repetition row to add the readings to, in place.
    """
    vram = sample_vram()

    if vram:
        row["vram_used_mb"] = max(vram)

    thermals = sample_thermals()

    if thermals:
        # The hottest GPU is the one whose throttle, if any, shaped this
        # run's tail; its figures are the ones worth carrying.
        hottest = max(
            thermals,
            key=lambda reading: (
                reading.get("temperature_c") is not None,
                reading.get("temperature_c") or 0,
            ),
        )

        if hottest.get("temperature_c") is not None:
            row["gpu_temperature_c"] = hottest["temperature_c"]

        if hottest.get("sm_clock_mhz") is not None:
            row["gpu_clock_mhz"] = hottest["sm_clock_mhz"]


def _run_one_repetition(
    engine: BenchEngine,
    model: str,
    prompt: str,
    options: dict,
    interrupt: StopSignal,
    keep_output: bool,
) -> dict:
    """Execute one prompt once and collect its raw measurements.

    Args:
        engine: The inference engine the run talks through.
        model: Target model name or tag.
        prompt: The prompt text to send.
        options: Generation options for this case.
        interrupt: Signal consulted before and during the generation.
        keep_output: Whether to carry the generated text on the row.

    Returns:
        dict: One raw repetition row with timings, token counts and machine
        readings.

    Raises:
        RunInterrupted: If the signal is tripped.
        DaemonError: If the generation fails.
    """
    interrupt.raise_if_stopped()

    started_at = time.perf_counter()

    response = engine.generate(
        model=model,
        prompt=prompt,
        options=options,
        interrupt=interrupt,
    )

    duration = time.perf_counter() - started_at

    prompt_tokens = response.get("prompt_eval_count", 0)
    output_tokens = response.get("eval_count", 0)
    prompt_duration_ns = response.get("prompt_eval_duration", 0)
    output_duration_ns = response.get("eval_duration", 0)

    row = {
        "success": True,
        "duration_seconds": duration,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "prompt_tokens_per_second": tokens_per_second(
            prompt_tokens, prompt_duration_ns
        ),
        "output_tokens_per_second": tokens_per_second(
            output_tokens, output_duration_ns
        ),
        "ttft_seconds": response.get("ttft_seconds"),
        "done": response.get("done", True),
    }

    # One snapshot right after the generation: the numbers describe the GPU
    # as the driver saw it at that moment.
    _machine_readings(row)

    if keep_output:
        row["response"] = response.get("response", "")

    return row


def _release_cancelled_case(
    engine: BenchEngine,
    model: str,
    name: str,
    completed: int,
    total: int,
    reason: str,
) -> None:
    """Undo an interrupted case's side effects and record the interruption.

    A benchmark writes nothing to disk, so its only side effect is the model
    it loaded into memory to run the prompts. Unloading it frees the VRAM the
    run was holding, which leaves the machine as it was before the run
    started; the log entry is the only thing that remains.

    Args:
        engine: The engine to release the model through.
        model: Model the run had loaded.
        name: Case label.
        completed: Prompts that had finished before the interruption.
        total: Prompts the case was going to execute.
        reason: Why the run was interrupted.
    """
    unloaded = False
    unload_error: str | None = None

    try:
        engine.release(model)
        unloaded = True
    except Exception as error:
        # The model may not have loaded yet, or the engine may already be
        # gone; neither is a reason to fail the cancellation.
        unload_error = str(error)

    record_interruption(
        component=COMPONENT,
        action="case",
        message="Case interrupted",
        details={
            "name": name,
            "model": model,
            "prompts_completed": completed,
            "prompts_total": total,
            "partial_results_discarded": completed,
            "model_unloaded": unloaded,
            "unload_error": unload_error,
            "reason": reason,
        },
    )


def measure_case(
    engine: BenchEngine,
    model: str,
    prompts: list[str],
    options: dict | None = None,
    label: str = "case",
    keep_output: bool = False,
    interrupt: StopSignal | None = None,
    repetitions: int = 1,
    on_progress: Callable[[dict], None] | None = None,
) -> dict:
    """Measure one temporary model configuration against multiple prompts.

    Each prompt runs ``repetitions`` times and the per-prompt result averages
    those runs, with the spread of the timing and rate metrics reported along
    the means. None is judged here — the numbers are offered to the caller.

    The model is checked and preloaded once, before the first prompt. Model
    loading is not included in case timing or performance results.

    Args:
        engine: The inference engine the run talks through.
        model: Target model name or tag.
        prompts: List of prompt strings to execute.
        options: Optional model parameter dictionary.
        label: Identifier name for the case.
        keep_output: Whether to include the generated text in results.
        interrupt: Optional signal that stops the run between or during
            prompts.
        repetitions: How many times every prompt is executed, from 1.
        on_progress: Optional callable called once after every individual
            repetition finishes with the step the run is at.

    Returns:
        dict: Case execution results and summary statistics, including the
            repetition count every prompt was averaged over.

    Raises:
        ValueError: If model name is empty, prompts is empty, or repetitions
            is not 1 or greater.
        TypeError: If keep_output is not boolean or any prompt is not a
            string.
        RunInterrupted: If the signal is tripped. Partial results are
            discarded and the model this run loaded is unloaded first.
    """
    if not model.strip():
        raise ValueError("Model name is required")

    if not prompts:
        raise ValueError("At least one prompt is required")

    if not isinstance(keep_output, bool):
        raise TypeError("keep_output must be a boolean")

    if not isinstance(repetitions, int) or isinstance(repetitions, bool):
        raise TypeError("repetitions must be an integer")

    if repetitions < 1:
        raise ValueError(f"repetitions must be 1 or greater, got {repetitions}")

    # One signal for the whole case, so the prompt loop and the generation all
    # consult the same flag whether or not a caller supplied one. An
    # unsupplied signal is simply never tripped.
    signal = interrupt if interrupt is not None else StopSignal()

    configuration = dict(options or {})

    record_event(
        level="INFO",
        component=COMPONENT,
        action="case",
        message="Case started",
        details={"name": label},
    )

    rows: list[dict] = []

    try:
        # The model is loaded once for the whole case: it stays resident
        # across the case's prompts and repetitions (the matrix unloads it
        # between cases), so the pre-load is a single step here rather than an
        # extra request before every repetition. It sits outside the benchmark
        # timer and never enters the results. A model that cannot be loaded
        # fails the case outright, which says the same thing a column of
        # failed prompts would, only sooner.
        signal.raise_if_stopped()
        engine.ensure_ready(model, configuration, interrupt=signal)

        for index, prompt in enumerate(prompts, start=1):
            if not isinstance(prompt, str):
                raise TypeError(f"Prompt {index} must be a string")

            signal.raise_if_stopped()

            # Successful repetitions only: a failed run contributes its error
            # but no timing, so averaging never mixes numbers with failures.
            successful: list[dict] = []
            errors: list[str] = []

            if on_progress is not None:
                _emit_progress(
                    on_progress,
                    phase="prompt_start",
                    prompt_index=index,
                    prompt_count=len(prompts),
                    repetition=0,
                    repetition_count=repetitions,
                    completed=0,
                )

            for repetition in range(1, repetitions + 1):
                signal.raise_if_stopped()

                try:
                    row = _run_one_repetition(
                        engine=engine,
                        model=model,
                        prompt=prompt,
                        options=configuration,
                        interrupt=signal,
                        keep_output=keep_output,
                    )
                except RunInterrupted:
                    # Cancellation is not a prompt failure: it must not be
                    # recorded as a result, and it stops the run rather than
                    # continuing.
                    raise
                except Exception as error:
                    errors.append(str(error))

                    record_event(
                        level="ERROR",
                        component=COMPONENT,
                        action="case",
                        message="Prompt execution failed",
                        details={
                            "name": label,
                            "prompt_index": index,
                            "repetition": repetition,
                            "error": str(error),
                        },
                    )
                else:
                    row.update({
                        "index": index,
                        "prompt": prompt,
                    })

                    successful.append(row)

                    record_event(
                        level="INFO",
                        component=COMPONENT,
                        action="case",
                        message="Prompt executed",
                        details={
                            "name": label,
                            "prompt_index": index,
                            "repetition": repetition,
                            "duration_seconds": row["duration_seconds"],
                            "prompt_tokens": row["prompt_tokens"],
                            "output_tokens": row["output_tokens"],
                            "prompt_tokens_per_second": (
                                row["prompt_tokens_per_second"]
                            ),
                            "output_tokens_per_second": (
                                row["output_tokens_per_second"]
                            ),
                            "ttft_seconds": row["ttft_seconds"],
                            "vram_used_mb": row.get("vram_used_mb"),
                            "gpu_temperature_c": row.get("gpu_temperature_c"),
                            "gpu_clock_mhz": row.get("gpu_clock_mhz"),
                        },
                    )

                if on_progress is not None:
                    completed = len(successful) + len(errors)
                    _emit_progress(
                        on_progress,
                        phase="repetition_done",
                        prompt_index=index,
                        prompt_count=len(prompts),
                        repetition=repetition,
                        repetition_count=repetitions,
                        completed=completed,
                    )

            rows.append(_fold_prompt(
                index=index,
                prompt=prompt,
                successful=successful,
                errors=errors,
                repetitions=repetitions,
                keep_output=keep_output,
            ))

    except RunInterrupted as error:
        _release_cancelled_case(
            engine=engine,
            model=model,
            name=label,
            completed=len(rows),
            total=len(prompts),
            reason=str(error),
        )
        rows.clear()
        raise

    successful_rows = [row for row in rows if row["success"]]

    result = {
        "name": label,
        "model": model,
        "configuration": configuration,
        "repetitions": repetitions,
        "results": rows,
        "summary": aggregate_case(successful_rows),
    }

    record_event(
        level="INFO",
        component=COMPONENT,
        action="case",
        message="Case completed",
        details={"name": label},
    )

    return result


def _fold_prompt(
    index: int,
    prompt: str,
    successful: list[dict],
    errors: list[str],
    repetitions: int,
    keep_output: bool,
) -> dict:
    """Collapse one prompt's repetitions into its report row.

    Args:
        index: 1-based prompt position within the case.
        prompt: The prompt text.
        successful: Successful repetition rows, in execution order.
        errors: One message per failed repetition.
        repetitions: Repetitions the prompt was meant to run.
        keep_output: Whether to carry the generated text into the row.

    Returns:
        dict: The prompt's row — averaged metrics when anything succeeded,
        the last error when everything did.
    """
    if successful:
        row = fold_repetitions(successful)
        row.update({
            "index": index,
            "success": True,
            "prompt": prompt,
        })

        if keep_output:
            # The generated text of the last successful run stands for the
            # prompt: every repetition sees the same seed unless the
            # configuration asks otherwise.
            row["response"] = successful[-1].get("response", "")

        if errors:
            # Some repetitions failed while others succeeded: the averages
            # stand on the successful runs alone, and the count says how much
            # of the work did not report.
            row["failed_repetitions"] = len(errors)

        return row

    # Every repetition failed: the prompt reports the last error, which is
    # the one a caller retrying would face again.
    row = {
        "index": index,
        "success": False,
        "prompt": prompt,
        "duration_seconds": 0.0,
        "error": errors[-1] if errors else "Unknown error",
    }

    if repetitions > 1:
        row["error_count"] = len(errors)

    return row
