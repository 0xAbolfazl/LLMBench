"""Run planning: shape a benchmark request into the matrix that will execute.

A benchmark run is a matrix — models down one axis, configurations down the
other, one shared prompt list every leaf answers. What kind of run a request
describes follows from its inputs rather than a mode the caller picks first:

- a model listed in the per-model options races its own configuration alone;
- a single model with several shared configurations races all of them;
- several models under one shared configuration all run under it;
- otherwise every model runs once under its own defaults.

:func:`build_plan` turns any of those shapes into one uniform plan the
executor can walk without knowing which shape produced it.
"""

from __future__ import annotations


def build_plan(
    models: list[str],
    shared_configurations: list[dict] | None = None,
    per_model_options: dict[str, dict] | None = None,
) -> list[dict]:
    """Shape a run request into the list of lanes the executor walks.

    Args:
        models: Model names, in run order.
        shared_configurations: Optional shared configurations, each
            ``{name, options}``.
        per_model_options: Optional per-model overrides, keyed by model name.

    Returns:
        list[dict]: One lane per model, each ``{"model", "configurations"}``
        with a configurations list of ``{name, options}``.
    """
    shared = shared_configurations or []
    overrides = per_model_options or {}
    lanes: list[dict] = []

    for model in models:
        if model in overrides:
            options = dict(overrides[model])
            lanes.append({
                "model": model,
                "configurations": [{"name": model, "options": options}],
            })
        elif len(models) == 1 and shared:
            # One model with its own set of configurations: race them all.
            configurations = [
                {"name": config["name"], "options": dict(config["options"])}
                for config in shared
            ]
            lanes.append({"model": model, "configurations": configurations})
        elif len(shared) == 1:
            # Several models under one shared configuration.
            options = dict(shared[0]["options"])
            lanes.append({
                "model": model,
                "configurations": [{"name": model, "options": options}],
            })
        else:
            # Default: the model runs once under its own defaults, under the
            # name a reader would recognize it by — its own.
            lanes.append({
                "model": model,
                "configurations": [{"name": model, "options": {}}],
            })

    return lanes
