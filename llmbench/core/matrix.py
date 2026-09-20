"""Matrix execution: walk a run plan case by case, then judge the results.

:func:`execute_matrix` is the whole comparison: it validates and normalises a
plan, measures every model-configuration pair over one shared prompt list,
and hands the finished cases to :mod:`llmbench.core.verdict` for the
significance judgement.

Models are measured one after another, and the model measured before is
unloaded before the next one loads: two models resident at once would compete
for the same VRAM and make every timing in both meaningless. The
configurations of one model run back to back while it stays loaded. A model
that was already loaded before the comparison began is not the comparison's
to stop — only the models this call loaded are unloaded, so the machine is
left as it was found.
"""

from __future__ import annotations

from collections.abc import Callable

from .eventlog import record_event
from .engines.base import BenchEngine
from .interrupt import RunInterrupted, StopSignal, record_interruption
from .suite import measure_case
from .verdict import judge_matrix

COMPONENT = "bench.matrix"


def _normalise_lanes(lanes: list[dict]) -> list[dict]:
    """Validate a run plan and fill in every configuration's defaults.

    Args:
        lanes: One dictionary per model, in run order.

    Returns:
        list[dict]: Normalised lanes, each configuration carrying a name and
        an options mapping.

    Raises:
        ValueError: If a lane names no model or carries no configuration.
        TypeError: If a lane or a configuration has the wrong shape.
    """
    normalised: list[dict] = []

    for position, lane in enumerate(lanes, start=1):
        if not isinstance(lane, dict):
            raise TypeError(f"Lane {position} must be a dictionary")

        model = lane.get("model")

        if not isinstance(model, str) or not model.strip():
            raise ValueError(f"Lane {position} must name a model")

        raw_configurations = lane.get("configurations")

        if raw_configurations is None:
            # A model with no configurations of its own runs once, under its
            # defaults, under the name a reader would recognize it by — its
            # own.
            raw_configurations = [{"name": model, "options": {}}]

        if not isinstance(raw_configurations, list) or not raw_configurations:
            raise ValueError(
                f"Lane for '{model}' must carry at least one configuration"
            )

        configurations: list[dict] = []

        for config_position, configuration in enumerate(
            raw_configurations, start=1
        ):
            if not isinstance(configuration, dict):
                raise TypeError(
                    f"Configuration {config_position} of '{model}' must be a "
                    f"dictionary"
                )

            name = configuration.get("name", f"configuration_{config_position}")
            options = configuration.get("options", {})

            if not isinstance(options, dict):
                raise TypeError(
                    f"Configuration '{name}' options must be a dictionary"
                )

            # Prompts are shared by design: every configuration answers the
            # same list, so a gap between two configurations' numbers
            # describes the configurations and never the questions. A
            # configuration carrying its own 'prompts' is refused rather than
            # quietly run over a different prompt set than its siblings.
            if "prompts" in configuration:
                raise TypeError(
                    f"Configuration '{name}' carries 'prompts', but prompts are "
                    f"shared across every configuration: pass them as "
                    f"shared_prompts instead"
                )

            configurations.append({"name": name, "options": dict(options)})

        normalised.append({
            "model": model,
            "configurations": configurations,
        })

    return normalised


def _release_quietly(engine: BenchEngine, model: str) -> None:
    """Release one model, treating a failure as a note rather than a stop.

    Args:
        engine: The engine the model was prepared on.
        model: Model name to release.
    """
    try:
        engine.release(model)
    except Exception as error:
        record_event(
            level="WARNING",
            component=COMPONENT,
            action="release",
            message="Model release failed between comparison steps",
            details={"model": model, "error": str(error)},
        )


def execute_matrix(
    engine: BenchEngine,
    lanes: list[dict],
    shared_prompts: list[str],
    keep_output: bool = False,
    interrupt: StopSignal | None = None,
    repetitions: int = 1,
    on_progress: Callable[[dict], None] | None = None,
) -> dict:
    """Benchmark a matrix of models and configurations over shared prompts.

    Every configuration answers the same ``shared_prompts``, so every
    model-configuration pair is measured the same way: prompts averaged over
    ``repetitions`` runs, with the run-to-run spread reported beside the
    means.

    Args:
        engine: The inference engine the run talks through.
        lanes: One dictionary per model, in run order. 'model' is the Ollama
            model name or tag, and 'configurations' — optional — is a list of
            dictionaries. Each configuration carries an optional 'name'
            (defaulting to its position) and optional 'options' (defaulting
            to empty).
        shared_prompts: The prompts every configuration runs. Required.
        keep_output: Whether to keep generated text in results.
        interrupt: Optional signal that stops the comparison part-way.
        repetitions: How many times every prompt runs per configuration,
            from 1.
        on_progress: Optional callable receiving one progress dict per step of
            the comparison: which model and configuration, which prompt and
            repetition, and how many steps have completed and will run in
            total.

    Returns:
        dict: 'experiments' echoes the normalised matrix that ran, 'models'
            lists the model names in run order, and 'tests' holds one case
            result per model-configuration pair. 'significance' judges the
            matrix the two ways it can fairly be judged: 'by_model' assesses
            each model's configurations against one another, and
            'across_models' assesses the models themselves, each represented
            by its fastest configuration.

    Raises:
        ValueError: If lanes is empty, a lane names no model, shared_prompts
            is empty, or repetitions is not 1 or greater.
        TypeError: If lanes, a configuration, shared_prompts or repetitions
            have the wrong types.
        RunInterrupted: If the signal is tripped. Results collected so far
            are discarded, and no model is left in memory.
    """
    if not isinstance(lanes, list) or not lanes:
        raise ValueError("At least one experiment is required")

    if not isinstance(shared_prompts, (list, tuple)):
        raise TypeError("shared_prompts must be a list of strings")

    if not all(isinstance(prompt, str) for prompt in shared_prompts):
        raise TypeError("shared_prompts must be a list of strings")

    if not shared_prompts:
        raise ValueError("At least one shared prompt is required")

    if not isinstance(keep_output, bool):
        raise TypeError("keep_output must be a boolean")

    if not isinstance(repetitions, int) or isinstance(repetitions, bool):
        raise TypeError("repetitions must be an integer")

    if repetitions < 1:
        raise ValueError(f"repetitions must be 1 or greater, got {repetitions}")

    # One signal for every case below, so one interruption stops the whole
    # comparison rather than only the lane that was running.
    signal = interrupt if interrupt is not None else StopSignal()

    normalised_lanes = _normalise_lanes(lanes)
    models = [lane["model"] for lane in normalised_lanes]

    record_event(
        level="INFO",
        component=COMPONENT,
        action="compare",
        message="Comparison started",
        details={
            "experiments": normalised_lanes,
            "repetitions": repetitions,
        },
    )

    # One step is one prompt of one configuration, and every configuration
    # answers the whole shared list, so the total is the matrix's leaves
    # times the shared prompt count, times the repetitions each prompt runs.
    total_steps = (
        len(shared_prompts)
        * repetitions
        * sum(len(lane["configurations"]) for lane in normalised_lanes)
    )

    cases: list[dict] = []
    resident: str | None = None
    steps_before = 0

    try:
        for model_position, lane in enumerate(normalised_lanes, start=1):
            signal.raise_if_stopped()

            if resident is not None and resident != lane["model"]:
                _release_quietly(engine, resident)

            for config_position, configuration in enumerate(
                lane["configurations"], start=1
            ):
                signal.raise_if_stopped()

                # The comparison's steps are its leaves' steps, prefixed with
                # which model and configuration each belongs to, so one
                # counter covers the whole comparison for a caller drawing a
                # single progress bar. The loop variables are bound as
                # defaults, so each closure keeps the step it was made for.
                def _comparison_progress(
                    step: dict,
                    _model=lane["model"],
                    _model_position=model_position,
                    _model_count=len(normalised_lanes),
                    _configuration=configuration["name"],
                    _configuration_position=config_position,
                    _configuration_count=len(lane["configurations"]),
                    _steps_before=steps_before,
                ) -> None:
                    on_progress({
                        **step,
                        "model": _model,
                        "model_index": _model_position,
                        "model_count": _model_count,
                        "configuration": _configuration,
                        "configuration_index": _configuration_position,
                        "configuration_count": _configuration_count,
                        "completed": _steps_before + step.get("completed", 0),
                        "total": total_steps,
                    })

                try:
                    case = measure_case(
                        engine=engine,
                        model=lane["model"],
                        prompts=list(shared_prompts),
                        options=configuration["options"],
                        label=configuration["name"],
                        keep_output=keep_output,
                        interrupt=signal,
                        repetitions=repetitions,
                        on_progress=(
                            _comparison_progress
                            if on_progress is not None
                            else None
                        ),
                    )
                except RunInterrupted as error:
                    # measure_case has already unloaded the model it had
                    # loaded and logged its own interruption; discard the
                    # finished pairs so no partial comparison survives, and
                    # record what the comparison as a whole lost.
                    record_interruption(
                        component=COMPONENT,
                        action="compare",
                        message="Comparison interrupted",
                        details={
                            "models": models,
                            "experiments_completed": model_position - 1,
                            "experiments_total": len(normalised_lanes),
                            "partial_results_discarded": len(cases),
                            "reason": str(error),
                        },
                    )
                    cases.clear()
                    raise

                cases.append(case)
                steps_before += len(shared_prompts) * repetitions

            resident = lane["model"]

    finally:
        # measure_case unloads its own model when an interruption ends it; on
        # every other exit the last model measured is unloaded here, so the
        # comparison leaves no model in memory whatever its outcome.
        if resident is not None:
            _release_quietly(engine, resident)

    result = {
        "experiments": normalised_lanes,
        "models": models,
        "tests": cases,
        "significance": judge_matrix(cases, models),
    }

    record_event(
        level="INFO",
        component=COMPONENT,
        action="compare",
        message="Comparison completed",
        details={"models": models},
    )

    return result
