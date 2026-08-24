from __future__ import annotations

from collections.abc import Mapping, Sequence
import re
from dataclasses import dataclass
from pathlib import Path

import yaml


PROVISIONS_DIR = Path(__file__).resolve().parents[1] / "provisions"

_PROVISION_NAME = re.compile(r"[a-z][a-z0-9_]*")
_CONFIG_KEYS = {
    "clause_type",
    "clause_description",
    "N_EXTRACTION_PASSES",
    "LANGEXTRACT_MAX_WORKERS",
    "LANGEXTRACT_BATCH_LENGTH",
    "MAX_CHAR_BUFFER",
}
# Keys a provision config may declare but that extraction does not need; stage 3
# classification reads the subtype taxonomy from the same file.
_OPTIONAL_CONFIG_KEYS = {
    "subtype_taxonomy",
}
_MINIMUM_SUBTYPES = 2
_NODE_KEYS = {"description", "subtypes"}
# Stage 3 injects this label at every level as the "none of the above" escape
# hatch, so a config may not also declare it.
RESERVED_SUBTYPE_LABEL = "other"

# Which party a clause substantively benefits. Stage 2 asks the model for this
# alongside each extraction; stage 3 and the analysis utils only read it back,
# so this is the single definition of the vocabulary.
BENEFICIARY_OPTIONS: Mapping[str, str] = {
    "worker": (
        "The clause's substantive benefit accrues to workers or the union: it creates a "
        "right, protection, entitlement, or constraint on management that workers can invoke."
    ),
    "employer": (
        "The clause's substantive benefit accrues to the employer or management: it affirms "
        "or expands management's discretion, or limits what workers may demand."
    ),
    "unclear": (
        "The clause confers no substantive benefit on either party, or the benefit cannot be "
        "assigned: it is purely procedural, genuinely mutual, or too vague to judge."
    ),
}
DEFAULT_BENEFICIARY = "unclear"


@dataclass(frozen=True)
class SubtypeNode:
    """One label in a provision's subtype taxonomy.

    ``children`` is empty for a leaf, and holds the next level down otherwise, in
    the order the config declares them.
    """

    label: str
    description: str
    children: Mapping[str, "SubtypeNode"]


@dataclass(frozen=True)
class ProvisionSpec:
    clause_type: str
    prompt_description: str
    extraction_passes: int
    langextract_max_workers: int
    langextract_batch_length: int
    max_char_buffer: int
    # Top level of the classification taxonomy, in config order; None when the
    # provision declares no subtypes and therefore cannot be classified.
    subtype_taxonomy: Mapping[str, SubtypeNode] | None = None


def _config_object(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("provision config must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise ValueError("provision config keys must be strings")

    actual_keys = set(value)
    missing = sorted(_CONFIG_KEYS - actual_keys)
    unknown = sorted(actual_keys - _CONFIG_KEYS - _OPTIONAL_CONFIG_KEYS)
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


def _subtype_level(
    value: object,
    trail: str,
    seen: set[str],
) -> dict[str, SubtypeNode]:
    """Validate one level of the taxonomy and recurse, preserving YAML order.

    ``seen`` accumulates every label in the whole tree: downstream analysis keys
    columns by bare label, so a label repeated under a different parent would
    silently merge two distinct classes.
    """

    if not isinstance(value, dict):
        raise ValueError(f"{trail} must be a mapping of label to subtype node")
    if len(value) < _MINIMUM_SUBTYPES:
        raise ValueError(f"{trail} must define at least {_MINIMUM_SUBTYPES} labels")

    level: dict[str, SubtypeNode] = {}
    for label, node in value.items():
        if not isinstance(label, str) or _PROVISION_NAME.fullmatch(label) is None:
            raise ValueError(
                f"{trail} labels must start with a lowercase letter and contain "
                "only lowercase letters, digits, and underscores"
            )
        if label == RESERVED_SUBTYPE_LABEL:
            raise ValueError(
                f'"{RESERVED_SUBTYPE_LABEL}" is reserved: classification offers it '
                "at every level automatically"
            )
        if label in seen:
            raise ValueError(f"duplicate subtype label across the taxonomy: {label}")
        seen.add(label)

        if not isinstance(node, dict):
            raise ValueError(f"{trail}.{label} must be a mapping with a description")
        unknown = sorted(set(node) - _NODE_KEYS)
        if unknown:
            raise ValueError(
                f"{trail}.{label} has unknown keys: {', '.join(unknown)} "
                f"(expected {', '.join(sorted(_NODE_KEYS))})"
            )

        description = _nonempty_string(
            node.get("description"), f"{trail}.{label}.description"
        ).strip()
        raw_children = node.get("subtypes")
        children = (
            {}
            if raw_children is None
            else _subtype_level(raw_children, f"{trail}.{label}.subtypes", seen)
        )
        level[label] = SubtypeNode(
            label=label, description=description, children=children
        )
    return level


def _subtype_taxonomy(value: object) -> dict[str, SubtypeNode]:
    return _subtype_level(value, "subtype_taxonomy", set())


def taxonomy_depth(taxonomy: Mapping[str, SubtypeNode]) -> int:
    """Number of levels the taxonomy declares; 1 when every label is a leaf."""

    if not taxonomy:
        return 0
    return 1 + max(taxonomy_depth(node.children) for node in taxonomy.values())


def children_of(
    taxonomy: Mapping[str, SubtypeNode],
    path: Sequence[str],
) -> Mapping[str, SubtypeNode]:
    """Children of the node reached by ``path``.

    Returns an empty mapping when the path runs off the tree -- which is how the
    injected "other" label terminates a classification cascade -- or when it
    lands on a leaf.
    """

    level = taxonomy
    for label in path:
        node = level.get(label)
        if node is None:
            return {}
        level = node.children
    return level


def labels_at_level(
    taxonomy: Mapping[str, SubtypeNode],
    level: int,
) -> dict[str, str]:
    """Every label at ``level`` across the whole tree, in declaration order."""

    if level < 1:
        raise ValueError("level must be at least 1")
    if level == 1:
        return {label: node.description for label, node in taxonomy.items()}
    labels: dict[str, str] = {}
    for node in taxonomy.values():
        labels.update(labels_at_level(node.children, level - 1))
    return labels



def render_options(options: Mapping[str, str]) -> str:
    """Label-to-description choices as a flat bullet list for a prompt."""

    return "\n".join(
        f"- {label}: {' '.join(description.split())}"
        for label, description in options.items()
    )


def _prompt_description(clause_type: str, clause_description: str) -> str:
    return (
        f'You are a legal expert reviewing chunks of contract text and extracting '
        f'{clause_type} class clauses.\n\n'
        f'Extract verbatim quotes from the text that meet the following criteria:\n'
        f'  1. Clearly defines a legal right, permission, obligation, or prohibition.\n'
        f'  2. Has a clear beneficiary who benefits from this clause.\n\n'
        f'In your response include the following attributes:\n'
        f'  1. context: a concise, faithful description of any additional information '
        f'in the text chunk relevant to understanding the extracted text\n'
        f'  2. beneficiary: which party receives a substantive benefit from this clause:\n'
        f'{render_options(BENEFICIARY_OPTIONS)}\n\n'
        f'{clause_type} Description: {clause_description}'
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

    clause_type = _nonempty_string(config["clause_type"], "clause_type")
    if _PROVISION_NAME.fullmatch(clause_type) is None:
        raise ValueError("clause_type must be a safe snake_case name")
    if clause_type != name:
        raise ValueError(f'clause_type must match filename "{name}.yaml"')

    clause_description = _nonempty_string(
        config["clause_description"], "clause_description"
    )
    raw_taxonomy = config.get("subtype_taxonomy")
    return ProvisionSpec(
        clause_type=clause_type,
        prompt_description=_prompt_description(clause_type, clause_description),
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
        subtype_taxonomy=(
            None if raw_taxonomy is None else _subtype_taxonomy(raw_taxonomy)
        ),
    )
