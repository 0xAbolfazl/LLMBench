"""The significance verdict: is a measured gap real, or noise?

A comparison is only worth ranking when its leaders are actually apart: two
averages separated by less than their combined standard deviations describe
the same speed, and naming a winner between them would be inventing a fact.

A run matrix holds two comparison questions at once — which of a model's
configurations is faster, and which model is — and one flat ranking cannot
answer both. So :func:`judge_matrix` assesses the matrix the two ways it can
fairly be judged: within each model, across that model's configurations; and
across models, each represented by its fastest configuration.

The metric every verdict turns on is output generation rate, the figure the
comparison is about.
"""

from __future__ import annotations

# How many combined standard deviations two entries' averages must be apart
# before their difference counts as real. Two overlapping spreads are noise;
# one clear gap between them is a finding. Deliberately conservative —
# claiming a difference that does not exist sends a user tuning a parameter
# that never mattered.
NOISE_RATIO_CUTOFF = 2.0


def judge_gap(cases: list[dict]) -> dict | None:
    """Compare the leading entries' averages against their own noise.

    With a single repetition there is no spread to argue from — the verdict
    then reports itself as unmeasured rather than pretending to a conclusion.
    An entry whose prompts all failed has no average and takes itself out of
    the running.

    Args:
        cases: One case result per configuration, in run order. Each carries
            ``name`` and a ``summary`` with the rate figures.

    Returns:
        dict | None: The verdict, or None when fewer than two entries
        produced a measurable average.
    """
    ranked: list[tuple] = []

    for case in cases:
        rate = case["summary"].get("average_output_tokens_per_second")

        if rate is not None:
            ranked.append((
                case["name"],
                rate,
                case["summary"].get("output_tokens_per_second_stddev"),
            ))

    if len(ranked) < 2:
        return None

    ranked.sort(key=lambda item: item[1], reverse=True)

    leader_name, leader_rate, leader_spread = ranked[0]
    runner_name, runner_rate, runner_spread = ranked[1]

    spreads = [
        spread for spread in (leader_spread, runner_spread) if spread is not None
    ]

    gap = leader_rate - runner_rate

    combined_spread = (
        (sum(spread ** 2 for spread in spreads) / len(spreads)) ** 0.5
        if spreads
        else None
    )

    if gap == 0:
        # A dead heat: the two measured exactly the same speed, so naming a
        # winner would be inventing one. This holds with or without
        # repetition data — a measured tie is a tie either way.
        return {
            "metric": "output_tokens_per_second",
            "leader": leader_name,
            "runner_up": runner_name,
            "difference": 0.0,
            "noise_level": combined_spread,
            "difference_to_noise_ratio": None,
            "significant": False,
            "message": (
                f"{leader_name} and {runner_name} measured exactly the same "
                f"speed ({leader_rate:.3g} tokens/s) — a dead heat; pick on "
                "other grounds."
            ),
        }

    if not spreads:
        # No repetition data anywhere: a difference is visible but nothing
        # measures whether it is real.
        return {
            "metric": "output_tokens_per_second",
            "leader": leader_name,
            "runner_up": runner_name,
            "difference": gap,
            "significant": None,
            "message": (
                "Measured once per prompt, so noise cannot be told apart from a "
                "real difference. Run again with repetitions set higher to learn "
                "whether this gap is real."
            ),
        }

    # A zero spread on both sides makes the ratio infinite, which is correct:
    # two perfectly repeatable measurements a hair apart really do differ.
    ratio = gap / combined_spread if combined_spread > 0 else None

    if ratio is None:
        significant = True
        description = (
            f"{leader_name} is faster by {gap:.3g} tokens/s, and both "
            f"measurements repeated exactly — the difference is real."
        )
    elif ratio >= NOISE_RATIO_CUTOFF:
        significant = True
        description = (
            f"{leader_name} is faster by {gap:.3g} tokens/s "
            f"({ratio:.1f}x the noise level) — the difference is real."
        )
    else:
        significant = False
        description = (
            f"{leader_name} and {runner_name} are within noise of each other "
            f"(gap {gap:.3g} tokens/s is {ratio:.1f}x the noise level). Either "
            f"configuration is fine; pick on other grounds."
        )

    return {
        "metric": "output_tokens_per_second",
        "leader": leader_name,
        "runner_up": runner_name,
        "difference": gap,
        "noise_level": combined_spread,
        "difference_to_noise_ratio": ratio,
        "significant": significant,
        "message": description,
    }


def judge_matrix(cases: list[dict], models: list[str]) -> dict:
    """Judge a run matrix per model and across models.

    A model with fewer than two measurable averages cannot support a
    within-model verdict and carries None there; fewer than two measurable
    models and the cross-model verdict is None too.

    Args:
        cases: One case result per model-configuration pair, in run order.
        models: The model names the run covered, in run order.

    Returns:
        dict: 'by_model' maps each model's name to its within-model verdict
        (or None when it cannot support one), and 'across_models' holds the
        cross-model verdict or None.
    """
    by_model: dict[str, dict | None] = {}

    for model in models:
        if model in by_model:
            # A model named twice runs twice; one verdict covers its cases.
            continue

        by_model[model] = judge_gap(
            [case for case in cases if case.get("model") == model]
        )

    representatives: list[dict] = []

    for model in by_model:
        candidates = [
            case
            for case in cases
            if case.get("model") == model
            and case.get("summary", {}).get("average_output_tokens_per_second")
            is not None
        ]

        if candidates:
            fastest = max(
                candidates,
                key=lambda case: case["summary"]["average_output_tokens_per_second"],
            )

            # The cross-model verdict speaks of models, so each representative
            # stands in under its model's name rather than its case's.
            representatives.append(dict(fastest, name=model))

    return {
        "by_model": by_model,
        "across_models": (
            judge_gap(representatives) if len(representatives) >= 2 else None
        ),
    }
