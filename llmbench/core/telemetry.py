"""GPU telemetry, sampled from ``nvidia-smi``.

A benchmark wants to know not only how fast the model generated but what the
machine did while it worked: how much VRAM the driver reported right after
each generation, and the temperature and clock of the hottest GPU. These
figures are reported as measured — judging them (throttling, headroom) is the
reader's part.

Every probe is best-effort: a machine without an NVIDIA GPU, or without
``nvidia-smi`` on the PATH, simply reports nothing, and the run carries on
without the optional columns.
"""

from __future__ import annotations

import subprocess


def _run_probe(command: list[str], timeout: int = 5) -> str | None:
    """Run a system probe command and return stripped stdout if successful.

    Args:
        command: Command list to execute via subprocess.
        timeout: Maximum execution time in seconds.

    Returns:
        str | None: Stripped stdout output if exit code is 0, otherwise None.
    """
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            # A host may be a long-lived process whose stdin it wants to keep,
            # so a child that inherits it could consume the protocol's bytes.
            stdin=subprocess.DEVNULL,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        # A missing tool, a timeout or a permission error all mean "unknown"
        # rather than a failed measurement.
        pass

    return None


def sample_vram() -> list[int]:
    """Sample the currently used VRAM of every NVIDIA GPU, in megabytes.

    One ``nvidia-smi --query-gpu=memory.used`` call, one snapshot. The number
    is what the driver reports for the whole GPU — every process on the machine
    is in it — so a caller that wants a model's own footprint samples while
    that model alone is the thing that changed.

    Returns:
        list[int]: Megabytes in use, one entry per GPU in nvidia-smi's order.
        Empty when nvidia-smi is missing, fails, or reports nothing.
    """
    output = _run_probe([
        "nvidia-smi",
        "--query-gpu=memory.used",
        "--format=csv,noheader,nounits",
    ])

    if not output:
        return []

    readings: list[int] = []

    for line in output.splitlines():
        try:
            readings.append(int(float(line.strip())))
        except ValueError:
            continue

    return readings


def sample_gpu_identity() -> list[dict]:
    """Sample each NVIDIA GPU's identity: name, driver, total memory.

    One ``nvidia-smi`` call, one snapshot. This is the static side of the
    machine a saved run's profile wants — what hardware actually produced
    the numbers — while :func:`sample_vram` and :func:`sample_thermals` are
    the moving side taken per generation.

    Returns:
        list[dict]: One entry per GPU carrying ``name``, ``driver_version``
        and ``memory_total_mb``, any of them None when not reported. Empty
        when nvidia-smi is missing, fails, or reports nothing.
    """
    output = _run_probe([
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ])

    if not output:
        return []

    readings: list[dict] = []

    for line in output.splitlines():
        parts = [item.strip() for item in line.split(",")]

        if not parts or not parts[0]:
            continue

        def _value(index: int) -> int | None:
            try:
                return int(float(parts[index]))
            except (IndexError, ValueError):
                return None

        readings.append({
            "name": parts[0],
            "driver_version": parts[1] if len(parts) > 1 else None,
            "memory_total_mb": _value(2),
        })

    return readings


def machine_profile() -> dict:
    """Collect the machine facts a saved run's profile carries.

    Returns:
        dict: ``gpus`` when an NVIDIA GPU answered, absent otherwise.
    """
    gpus = sample_gpu_identity()

    if gpus:
        return {"gpus": gpus}

    return {}


def sample_thermals() -> list[dict]:
    """Sample each NVIDIA GPU's temperature and current SM clock.

    One ``nvidia-smi`` call, one snapshot. The clock is the one the driver
    reports at that moment — after a GPU has throttled, the falling figure is
    the trace it leaves, which is what makes the reading worth pairing with
    the temperature that caused it.

    Returns:
        list[dict]: One entry per GPU carrying 'temperature_c' and
        'sm_clock_mhz', either None when that figure was not reported.
        Empty when nvidia-smi is missing, fails, or reports nothing.
    """
    output = _run_probe([
        "nvidia-smi",
        "--query-gpu=temperature.gpu,clocks.current.sm",
        "--format=csv,noheader,nounits",
    ])

    if not output:
        return []

    def _number(value: str) -> int | None:
        try:
            return int(float(value))
        except ValueError:
            return None

    readings: list[dict] = []

    for line in output.splitlines():
        parts = [item.strip() for item in line.split(",")]

        if len(parts) < 2:
            continue

        temperature = _number(parts[0])
        clock = _number(parts[1])

        if temperature is not None or clock is not None:
            readings.append({
                "temperature_c": temperature,
                "sm_clock_mhz": clock,
            })

    return readings
