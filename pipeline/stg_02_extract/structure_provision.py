from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml


PROVISIONS_DIR = Path(__file__).with_name("provisions")

_PROVISION_NAME = re.compile(r"[a-z][a-z0-9_]*")
_CONFIG_KEYS = {
    "provision_type",
    "prompt",
    "N_EXTRACTION_PASSES",
    "LANGEXTRACT_MAX_WORKERS",
    "LANGEXTRACT_BATCH_LENGTH",
    "MAX_CHAR_BUFFER",
}


@dataclass(frozen=True)
class ProvisionSpec:
    provision_type: str
    prompt_description: str
    extraction_passes: int
    langextract_max_workers: int
    langextract_batch_length: int
    max_char_buffer: int


def _config_object(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("provision config must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise ValueError("provision config keys must be strings")

    actual_keys = set(value)
    missing = sorted(_CONFIG_KEYS - actual_keys)
    unknown = sorted(actual_keys - _CONFIG_KEYS)
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing keys: {', '.join(missing)}")
        if unknown:
            details.append(f"unknown keys: {', '.join(unknown)}")
        raise ValueError(
            f"provision config has an invalid shape ({'; '.join(details)})"
        )
    return value


def _nonempty_string(value: object, key: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _positive_integer(value: object, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{key} must be a positive integer")
    return value


def _prompt_description(prompt: str, provision_type: str) -> str:
    return (
        f"{prompt.strip()}\n\n"
        f'Use "{provision_type}" as extraction_class for every extraction. '
        "For each extraction, return attributes with exactly one key named "
        "context. Set context to a concise, faithful description of any "
        "information elsewhere in the current text chunk that is relevant to "
        "understanding the extracted text. Use only information from the current "
        "chunk, and set context to null when the chunk contains no relevant "
        "context."
    )


def load_provision(name: str) -> ProvisionSpec:
    """Load and validate a prompt-only provision extraction config."""

    if not isinstance(name, str) or _PROVISION_NAME.fullmatch(name) is None:
        raise ValueError(
            "provision name must start with a lowercase letter and contain only "
            "lowercase letters, digits, and underscores"
        )

    path = PROVISIONS_DIR / f"{name}.yaml"
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML in provision config {path.name}") from exc
    config = _config_object(raw)

    provision_type = _nonempty_string(config["provision_type"], "provision_type")
    if _PROVISION_NAME.fullmatch(provision_type) is None:
        raise ValueError("provision_type must be a safe snake_case name")
    if provision_type != name:
        raise ValueError(f'provision_type must match filename "{name}.yaml"')

    prompt = _nonempty_string(config["prompt"], "prompt")
    return ProvisionSpec(
        provision_type=provision_type,
        prompt_description=_prompt_description(prompt, provision_type),
        extraction_passes=_positive_integer(
            config["N_EXTRACTION_PASSES"],
            "N_EXTRACTION_PASSES",
        ),
        langextract_max_workers=_positive_integer(
            config["LANGEXTRACT_MAX_WORKERS"],
            "LANGEXTRACT_MAX_WORKERS",
        ),
        langextract_batch_length=_positive_integer(
            config["LANGEXTRACT_BATCH_LENGTH"],
            "LANGEXTRACT_BATCH_LENGTH",
        ),
        max_char_buffer=_positive_integer(
            config["MAX_CHAR_BUFFER"],
            "MAX_CHAR_BUFFER",
        ),
    )
