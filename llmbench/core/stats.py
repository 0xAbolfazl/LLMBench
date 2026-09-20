"""Statistics that turn raw repetition rows into the figures reports show.

Three levels of aggregation live here, one per level of the run:

- :func:`fold_repetitions` collapses one prompt's repetitions into a mean
  with its spread (standard deviation, minimum, maximum);
- :func:`aggregate_case` sums one case's prompts into the averages a table
  row shows, plus the pooled output-rate noise the verdict argues from;
- :func:`tokens_per_second` is the shared rate arithmetic.

Nothing here judges — the numbers are offered to whoever asks, and the
judgement lives in :mod:`llmbench.core.verdict`.
"""

from __future__ import annotations

# Per-repetition metrics averaged across a prompt's repetitions, and the
# subset whose spread is reported alongside the means. Every entry is
# optional on a run: ttft_seconds is absent when a generation produced no
# content, and the GPU figures when the machine has no NVIDIA GPU.
AVERAGED_METRICS = (
    "duration_seconds",
    "prompt_tokens",
    "output_tokens",
    "prompt_tokens_per_second",
    "output_tokens_per_second",
    "ttft_seconds",
    "vram_used_mb",
    "gpu_temperature_c",
    "gpu_clock_mhz",
)

DISPERSED_METRICS = (
    "duration_seconds",
    "prompt_tokens_per_second",
    "output_tokens_per_second",
    "ttft_seconds",
    "gpu_temperature_c",
)


def tokens_per_second(
    token_count: int | float,
    duration_ns: int | float,
) -> float | None:
    """Calculate token generation rate in tokens per second.

    Args:
        token_count: Number of processed or evaluated tokens.
        duration_ns: Processing duration in nanoseconds.

    Returns:
        float | None: Tokens per second rate, or None if invalid.
    """
    if not token_count or not duration_ns:
        return None

    duration_seconds = duration_ns / 1_000_000_000

    if duration_seconds <= 0:
        return None

    return token_count / duration_seconds


def mean(values: list[float]) -> float | None:
    """Return the arithmetic mean, or None for an empty sample.

    Args:
        values: Sample of one measured metric across repetitions.

    Returns:
        float | None: The mean, or None when nothing was measured.
    """
    if not values:
        return None

    return sum(values) / len(values)


def deviation(values: list[float]) -> float | None:
    """Return the population standard deviation of a sample.

    A single measurement has no spread to speak of, so it reports zero rather
    than None: callers can compare variability across prompts without
    special-casing the one-repetition runs that are the default.

    Args:
        values: Sample of one measured metric across repetitions.

    Returns:
        float | None: The standard deviation, or None for an empty sample.
    """
    if not values:
        return None

    if len(values) == 1:
        return 0.0

    average = sum(values) / len(values)

    variance = sum((value - average) ** 2 for value in values) / len(values)

    return variance ** 0.5


def fold_repetitions(repetitions: list[dict]) -> dict | None:
    """Collapse one prompt's repetitions into averaged metrics with spread.

    Only successful repetitions carry timings, so an all-failed prompt has
    nothing to average and reports None — the caller then shows the error
    instead of invented numbers.

    Args:
        repetitions: Successful repetition rows for one prompt, in execution
            order.

    Returns:
        dict | None: Mean for every numeric metric plus standard deviation,
        minimum and maximum for the timing and rate metrics, and the
        repetition count; or None when the list is empty.
    """
    if not repetitions:
        return None

    folded: dict = {"repetitions": len(repetitions)}

    for metric in AVERAGED_METRICS:
        values = [
            repetition[metric]
            for repetition in repetitions
            if repetition.get(metric) is not None
        ]

        folded[metric] = mean(values)

        if metric in DISPERSED_METRICS:
            folded[f"{metric}_stddev"] = deviation(values)
            folded[f"{metric}_min"] = min(values) if values else None
            folded[f"{metric}_max"] = max(values) if values else None

    folded["done"] = all(
        repetition.get("done", True) for repetition in repetitions
    )

    return folded


def aggregate_case(rows: list[dict]) -> dict:
    """Build aggregate metrics from a case's successful prompt rows.

    The output-rate noise level combines every prompt's own run-to-run spread
    into one pooled figure. It exists for the comparison verdict: how far two
    configurations' averages sit apart only means something against the noise
    both of them carry. A single-repetition case has no spread to pool and
    reports None, so a caller never mistakes "measured once" for "perfectly
    repeatable".

    Args:
        rows: List of successful prompt result dictionaries.

    Returns:
        dict: Summary statistics including average durations, rates, and
        token counts, plus the output-rate noise level when repetitions were
        run.
    """
    if not rows:
        return {
            "average_duration_seconds": None,
            "average_prompt_tokens_per_second": None,
            "average_output_tokens_per_second": None,
            "total_output_tokens": 0,
            "output_tokens_per_second_stddev": None,
        }

    average_duration = sum(row["duration_seconds"] for row in rows) / len(rows)

    prompt_rates = [
        row["prompt_tokens_per_second"]
        for row in rows
        if row.get("prompt_tokens_per_second") is not None
    ]

    output_rates = [
        row["output_tokens_per_second"]
        for row in rows
        if row.get("output_tokens_per_second") is not None
    ]

    total_output_tokens = sum(row["output_tokens"] for row in rows)

    # Only prompts that actually repeated carry a spread; averaging their
    # variances is the pooled estimate of the run-to-run noise the whole case
    # was subject to. Prompts measured once contribute nothing and say
    # nothing.
    spreads = [
        row["output_tokens_per_second_stddev"]
        for row in rows
        if row.get("repetitions", 1) > 1
        and row.get("output_tokens_per_second_stddev") is not None
    ]

    output_noise = (
        (sum(spread ** 2 for spread in spreads) / len(spreads)) ** 0.5
        if spreads
        else None
    )

    return {
        "average_duration_seconds": average_duration,
        "average_prompt_tokens_per_second": (
            sum(prompt_rates) / len(prompt_rates) if prompt_rates else None
        ),
        "average_output_tokens_per_second": (
            sum(output_rates) / len(output_rates) if output_rates else None
        ),
        "total_output_tokens": total_output_tokens,
        "output_tokens_per_second_stddev": output_noise,
    }
