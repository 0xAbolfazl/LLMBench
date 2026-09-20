"""Benchmark run options: the option table and input normalisation.

This module is the single source of truth for the benchmark form: the browser
builds its manual configuration editor from :data:`OPTION_CATALOG` and uploaded
files are validated against the same table, so a hand-typed configuration and
an uploaded one can never disagree about what a field means.

Unknown option keys are deliberately kept rather than rejected. Ollama accepts
more options than are worth listing here, and a configuration file written for
a newer server should still be runnable.
"""

from __future__ import annotations

import json

from .limits import MAX_CONFIGURATIONS, MAX_PROMPTS, MAX_REPETITIONS


def _catalog_index(catalog: list[dict] | None) -> dict[str, dict]:
    """Index an engine's option catalog by key.

    Args:
        catalog: One engine's option table, or None when the caller has no
            catalog to coerce against (unknown keys then pass through
            untouched).

    Returns:
        dict: Option key to option spec.
    """
    if not catalog:
        return {}

    return {option["key"]: option for option in catalog}

# Keys that identify a wrapper object rather than a bare options mapping.
_DOCUMENT_KEYS = {
    "model",
    "models",
    "prompts",
    "configurations",
    "configs",
    "include_output",
    "repetitions",
}
_CONFIG_KEYS = {"name", "options", "parameters", "config"}

_TRUE_WORDS = {"true", "yes", "on", "1"}
_FALSE_WORDS = {"false", "no", "off", "0"}

class OptionError(ValueError):
    """Raised when configuration input cannot be understood."""


def _coerce_number(key: str, value, want_int: bool):
    """Coerce a JSON or form value to a number for one known option.

    Args:
        key: Option name, used in error messages.
        value: Incoming value.
        want_int: Whether the option is an integer option.

    Returns:
        int | float: The coerced number.

    Raises:
        OptionError: If the value is not numeric.
    """
    if isinstance(value, bool):
        raise OptionError(f"Option '{key}' expects a number, not a true/false value")

    try:
        number = float(value)
    except (TypeError, ValueError):
        raise OptionError(f"Option '{key}' expects a number, got {value!r}") from None

    if number != number or number in (float("inf"), float("-inf")):
        raise OptionError(f"Option '{key}' must be a finite number")

    if not want_int:
        return number

    if not number.is_integer():
        raise OptionError(f"Option '{key}' expects a whole number, got {value!r}")

    return int(number)


def _coerce_bool(key: str, value) -> bool:
    """Coerce a value to a boolean.

    Args:
        key: Option name, used in error messages.
        value: Incoming value.

    Returns:
        bool: The coerced boolean.

    Raises:
        OptionError: If the value is not a recognisable boolean.
    """
    if isinstance(value, bool):
        return value

    text = str(value).strip().lower()

    if text in _TRUE_WORDS:
        return True

    if text in _FALSE_WORDS:
        return False

    raise OptionError(f"Option '{key}' expects true or false, got {value!r}")


def _coerce_list(key: str, value) -> list[str]:
    """Coerce a value to a list of non-empty strings.

    Ollama's ``stop`` option takes several sequences, which a form can only
    offer as one comma-separated field, so a plain string is split here.

    Args:
        key: Option name, used in error messages.
        value: Incoming value.

    Returns:
        list[str]: The coerced list.

    Raises:
        OptionError: If the value is neither a string nor a list of strings.
    """
    if isinstance(value, str):
        items = [part.strip() for part in value.split(",")]
    elif isinstance(value, list):
        items = [str(part).strip() for part in value]
    else:
        raise OptionError(f"Option '{key}' expects a string or a list of strings")

    return [item for item in items if item]


def normalize_options(options, catalog: list[dict] | None = None) -> dict:
    """Validate and coerce one configuration's option mapping.

    Keys the catalog does not know are passed through untouched: engines
    accept more options than this app lists, and rejecting them would make a
    valid configuration file unusable here.

    Args:
        options: Mapping of option names to values.
        catalog: The engine's option table, driving the coercion.

    Returns:
        dict: Coerced options, with blanks and nulls dropped.

    Raises:
        OptionError: If the mapping or any value is invalid.
    """
    if not isinstance(options, dict):
        raise OptionError(
            "Configuration options must be an object of name/value pairs"
        )

    index = _catalog_index(catalog)
    coerced: dict = {}

    for raw_key, value in options.items():
        key = str(raw_key).strip()

        if not key:
            continue

        # A blank field means "leave this at the model's default", which is not
        # the same as sending the option with an empty value.
        if value is None or (isinstance(value, str) and not value.strip()):
            continue

        spec = index.get(key)

        if spec is None:
            coerced[key] = value
            continue

        kind = spec["type"]

        if kind == "int":
            coerced[key] = _coerce_number(key, value, want_int=True)
        elif kind == "float":
            coerced[key] = _coerce_number(key, value, want_int=False)
        elif kind == "bool":
            coerced[key] = _coerce_bool(key, value)
        elif kind == "list":
            items = _coerce_list(key, value)
            if items:
                coerced[key] = items
        else:
            coerced[key] = value

    return coerced


def _unwrap_configuration(entry, index: int) -> tuple[str | None, dict]:
    """Read a name and an options mapping out of one configuration entry.

    Args:
        entry: Configuration object from a request body or uploaded file.
        index: 1-based position, used in error messages.

    Returns:
        tuple: The declared name or None, and the raw options mapping.

    Raises:
        OptionError: If the entry is not an object.
    """
    if not isinstance(entry, dict):
        raise OptionError(f"Configuration {index} must be an object")

    name = entry.get("name")

    # "options" is what the executor expects; the other two are what people
    # write when they are thinking of Modelfile PARAMETERs instead.
    for key in ("options", "parameters", "config"):
        if isinstance(entry.get(key), dict):
            return name, entry[key]

    if name is None:
        # No wrapper key and no name: the object is the options mapping itself.
        return None, entry

    return name, {}


def normalize_configurations(entries, catalog: list[dict] | None = None) -> list[dict]:
    """Validate a list of configurations and give every one a unique name.

    Names reach the results table and the log, so duplicates would make two
    columns indistinguishable. Later collisions are suffixed rather than
    rejected, since a file assembled from two others can easily repeat a name.

    Args:
        entries: List of configuration objects.
        catalog: The engine's option table, driving the coercion.

    Returns:
        list[dict]: Configurations as ``{"name": str, "options": dict}``.

    Raises:
        OptionError: If the list is empty, too long, or holds an invalid entry.
    """
    if not isinstance(entries, list) or not entries:
        raise OptionError("At least one configuration is required")

    if len(entries) > MAX_CONFIGURATIONS:
        raise OptionError(
            f"At most {MAX_CONFIGURATIONS} configurations can be compared in one run"
        )

    configurations: list[dict] = []
    used_names: set[str] = set()

    for index, entry in enumerate(entries, start=1):
        declared_name, raw_options = _unwrap_configuration(entry, index)
        options = normalize_options(raw_options, catalog)

        if not options:
            raise OptionError(
                f"Configuration {index} sets no options. Give it at least one "
                "parameter, or remove it."
            )

        name = str(declared_name or "").strip() or f"config_{index}"
        candidate = name
        suffix = 2

        while candidate in used_names:
            candidate = f"{name} ({suffix})"
            suffix += 1

        used_names.add(candidate)

        configurations.append({"name": candidate, "options": options})

    return configurations


def normalize_repetitions(value) -> int:
    """Validate and coerce the repetitions count for a comparison.

    Args:
        value: The repetitions value as sent, or None for the default of one.

    Returns:
        int: The repetition count, always at least one.

    Raises:
        OptionError: If the value is not a whole number in range.
    """
    if value is None or value == "":
        return 1

    try:
        count = int(value)
    except (TypeError, ValueError):
        raise OptionError("Repetitions must be a whole number")

    if count < 1:
        raise OptionError("Repetitions must be at least 1")

    if count > MAX_REPETITIONS:
        raise OptionError(
            f"At most {MAX_REPETITIONS} repetitions can be requested in one run"
        )

    return count


def normalize_prompts(value) -> list[str]:
    """Validate the prompt list, accepting one blob of text or a JSON array.

    The form offers a textarea where prompts are separated by blank lines,
    which keeps multi-line prompts intact while still allowing several of them.

    Args:
        value: List of prompt strings, or one string holding them all.

    Returns:
        list[str]: Non-empty prompts.

    Raises:
        OptionError: If no prompt survives, or there are too many.
    """
    if isinstance(value, str):
        blocks = value.split("\n\n")
    elif isinstance(value, list):
        blocks = [str(item) for item in value]
    else:
        raise OptionError("Prompts must be a list of strings")

    prompts = [block.strip() for block in blocks if block.strip()]

    if not prompts:
        raise OptionError("At least one prompt is required")

    if len(prompts) > MAX_PROMPTS:
        raise OptionError(f"At most {MAX_PROMPTS} prompts can be used in one run")

    return prompts


def decode_run_file(text: str, catalog: list[dict] | None = None) -> dict:
    """Read an uploaded configuration file into the fields the page can fill in.

    Every key is optional: a file holding only configurations is useful on its
    own, and the page keeps whatever the user already chose for the rest.

    Args:
        text: File contents as JSON.
        catalog: The engine's option table, driving the coercion.

    Returns:
        dict: ``model``, ``prompts``, ``configurations`` and ``include_output``,
            with None or an empty list where the file said nothing.

    Raises:
        OptionError: If the text is not JSON, or its contents are invalid.
    """
    stripped = (text or "").strip()

    if not stripped:
        raise OptionError("The file is empty")

    try:
        document = json.loads(stripped)
    except json.JSONDecodeError as error:
        raise OptionError(
            f"The file is not valid JSON — line {error.lineno}, column "
            f"{error.colno}: {error.msg}"
        ) from error

    if isinstance(document, list):
        return {
            "model": None,
            "prompts": [],
            "configurations": normalize_configurations(document, catalog),
            "include_output": None,
            "repetitions": None,
        }

    if not isinstance(document, dict):
        raise OptionError(
            "A configuration file must hold an object or a list of configurations"
        )

    # A bare options mapping has none of the structural keys, so treat the
    # whole object as a single unnamed configuration.
    if not (_DOCUMENT_KEYS | _CONFIG_KEYS) & document.keys():
        return {
            "model": None,
            "prompts": [],
            "configurations": normalize_configurations([document], catalog),
            "include_output": None,
            "repetitions": None,
        }

    entries = document.get("configurations")

    if entries is None:
        entries = document.get("configs")

    if entries is None and _CONFIG_KEYS & document.keys():
        entries = [document]

    configurations = (
        normalize_configurations(entries, catalog) if entries is not None else []
    )

    model = document.get("model")
    raw_prompts = document.get("prompts")
    include_output = document.get("include_output")
    repetitions = document.get("repetitions")

    return {
        "model": (
            str(model).strip()
            if isinstance(model, str) and model.strip()
            else None
        ),
        "prompts": normalize_prompts(raw_prompts) if raw_prompts else [],
        "configurations": configurations,
        "include_output": (
            _coerce_bool("include_output", include_output)
            if include_output is not None
            else None
        ),
        "repetitions": (
            normalize_repetitions(repetitions)
            if repetitions is not None
            else None
        ),
    }
