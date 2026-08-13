"""Detect and safely dispose of OCR decoding degeneration loops.

Collective bargaining agreements contain many legitimate repeated characters and
table rows.  Detection is consequently conservative: only a periodic run at the
absolute end of the response is considered, and every length/share gate described
in the stage-01 benchmark design must pass.
"""

from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Callable, Literal, Mapping
import warnings


MIN_RUN_LENGTH = 200
MIN_NON_ALNUM_RUN_LENGTH = 1_001
MIN_STOP_RUN_SHARE_NUMERATOR = 1
MIN_STOP_RUN_SHARE_DENOMINATOR = 4


def _copy_mapping(value: Mapping[str, object]) -> dict[str, object]:
    copied: dict[str, object] = {}
    for key, item in value.items():
        copied[key] = dict(item) if isinstance(item, Mapping) else item
    return copied


@dataclass(frozen=True)
class SamplingAttempt:
    """Sampling changes for one request in a model-specific retry ladder.

    ``repetition_penalty`` is placed in vLLM's ``extra_body`` while standard
    OpenAI parameters are placed at the request top level.  ``overrides`` supports
    model-specific additions such as ``top_p`` or ``top_k``.
    """

    temperature: float | None = 0.0
    repetition_penalty: float | None = None
    seed: int | None = None
    overrides: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.temperature is not None and self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if self.repetition_penalty is not None and self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive")
        if not isinstance(self.overrides, Mapping):
            raise TypeError("overrides must be a mapping")
        object.__setattr__(
            self,
            "overrides",
            MappingProxyType(_copy_mapping(self.overrides)),
        )

    def request_overrides(self) -> dict[str, object]:
        """Return fresh request kwargs for this attempt."""

        result = _copy_mapping(self.overrides)
        if self.temperature is not None:
            result["temperature"] = self.temperature
        if self.seed is not None:
            result["seed"] = self.seed
        if self.repetition_penalty is not None:
            existing = result.get("extra_body", {})
            if not isinstance(existing, Mapping):
                raise TypeError("overrides['extra_body'] must be a mapping")
            extra_body = dict(existing)
            extra_body["repetition_penalty"] = self.repetition_penalty
            result["extra_body"] = extra_body
        return result

    def apply_to(self, base_kwargs: Mapping[str, object]) -> dict[str, object]:
        """Merge this attempt into a request without mutating the base mapping."""

        result = _copy_mapping(base_kwargs)
        attempt = self.request_overrides()
        attempt_extra = attempt.pop("extra_body", None)
        result.update(attempt)
        if attempt_extra is not None:
            if not isinstance(attempt_extra, Mapping):  # defensive; validated above
                raise TypeError("extra_body must be a mapping")
            base_extra = result.get("extra_body", {})
            if not isinstance(base_extra, Mapping):
                raise TypeError("base_kwargs['extra_body'] must be a mapping")
            merged_extra = dict(base_extra)
            merged_extra.update(attempt_extra)
            result["extra_body"] = merged_extra
        return result


@dataclass(frozen=True)
class RepetitionPolicy:
    """A complete initial-attempt/retry ladder and final disposition."""

    attempts: tuple[SamplingAttempt, ...]
    fail_on_repetition: bool = False
    reject_length_finish: bool = False
    output_rejection_reason: Callable[[str, object | None], str | None] | None = None
    # Some model authors provide their own deterministic final cleanup.  These
    # policies still use the shared detector to drive retries, but leave the
    # exhausted response intact for the runner-specific postprocessor.
    trim_on_exhaustion: bool = True

    def __post_init__(self) -> None:
        attempts = tuple(self.attempts)
        if not attempts:
            raise ValueError("a repetition policy needs at least one sampling attempt")
        if not all(isinstance(attempt, SamplingAttempt) for attempt in attempts):
            raise TypeError("all policy attempts must be SamplingAttempt instances")
        if self.output_rejection_reason is not None and not callable(
            self.output_rejection_reason
        ):
            raise TypeError("output_rejection_reason must be callable")
        if not isinstance(self.trim_on_exhaustion, bool):
            raise TypeError("trim_on_exhaustion must be a bool")
        object.__setattr__(self, "attempts", attempts)

    @property
    def retry_count(self) -> int:
        return len(self.attempts) - 1


@dataclass(frozen=True)
class DegenerationMatch:
    """A repeated suffix selected for trimming after retry exhaustion."""

    start: int
    end: int
    unit: str
    repetitions: int
    kind: Literal["cycle", "line"] = "cycle"

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError("invalid degeneration match bounds")
        if not self.unit:
            raise ValueError("a degeneration unit cannot be empty")
        if self.repetitions < 2:
            raise ValueError("a degeneration match needs at least two repetitions")

    @property
    def run_length(self) -> int:
        return self.end - self.start


class RepetitionError(RuntimeError):
    """Raised when fail-on-repetition is selected and all attempts degenerate."""


class GenerationLengthError(RuntimeError):
    """Raised when a runner rejects a completion that exhausted its token budget."""


class GenerationQualityError(RuntimeError):
    """Raised when every attempt produces model-specific invalid output."""


def _z_values(value: str) -> list[int]:
    """Compute the Z-array in linear time (used on the reversed response)."""

    length = len(value)
    z = [0] * length
    left = right = 0
    for index in range(1, length):
        if index <= right:
            z[index] = min(right - index + 1, z[index - left])
        while index + z[index] < length and value[z[index]] == value[index + z[index]]:
            z[index] += 1
        if index + z[index] - 1 > right:
            left = index
            right = index + z[index] - 1
    if z:
        z[0] = length
    return z


def _cycle_match(text: str) -> DegenerationMatch | None:
    length = len(text)
    if length < MIN_RUN_LENGTH:
        return None

    reversed_text = text[::-1]
    z = _z_values(reversed_text)
    best: DegenerationMatch | None = None
    # For a suffix period p, z[p] tells how much of the preceding text repeats
    # backwards from the end.  Full p-sized blocks are exact repeated cycles.
    for period in range(1, length // 2 + 1):
        if z[period] < period:
            continue
        repetitions = 1 + z[period] // period
        run_length = repetitions * period
        if run_length < MIN_RUN_LENGTH:
            continue
        candidate = DegenerationMatch(
            start=length - run_length,
            end=length,
            unit=text[length - period :],
            repetitions=repetitions,
            kind="cycle",
        )
        if best is None or (candidate.start, len(candidate.unit)) < (
            best.start,
            len(best.unit),
        ):
            best = candidate
    return best


def _line_body(line: str) -> str:
    if line.endswith("\r\n"):
        return line[:-2]
    if line.endswith(("\n", "\r")):
        return line[:-1]
    return line


def _line_match(text: str) -> DegenerationMatch | None:
    lines = text.splitlines(keepends=True)
    if len(lines) < 2:
        return None
    body = _line_body(lines[-1])
    repetitions = 1
    for line in reversed(lines[:-1]):
        if _line_body(line) != body:
            break
        repetitions += 1
    if repetitions < 2:
        return None
    run_length = sum(len(line) for line in lines[-repetitions:])
    if run_length < MIN_RUN_LENGTH:
        return None
    return DegenerationMatch(
        start=len(text) - run_length,
        end=len(text),
        unit=lines[-1],
        repetitions=repetitions,
        kind="line",
    )


def _finish_reason_is_length(finish_reason: object) -> bool:
    value = getattr(finish_reason, "value", finish_reason)
    return isinstance(value, str) and value.casefold() == "length"


def _passes_gates(
    text: str,
    match: DegenerationMatch,
    finish_reason: object,
) -> bool:
    if match.run_length < MIN_RUN_LENGTH:
        return False
    run = text[match.start : match.end]
    if not any(character.isalnum() for character in run):
        if match.run_length < MIN_NON_ALNUM_RUN_LENGTH:
            return False
    if _finish_reason_is_length(finish_reason):
        return True
    return (
        match.run_length * MIN_STOP_RUN_SHARE_DENOMINATOR
        >= len(text) * MIN_STOP_RUN_SHARE_NUMERATOR
    )


def detect_degeneration(
    text: str | None,
    finish_reason: object = None,
) -> DegenerationMatch | None:
    """Return a strictly end-anchored repeated run, or ``None``.

    A candidate must be at least 200 characters.  Punctuation/whitespace-only
    candidates must be longer than 1000 characters.  Finally, either generation
    ended for length or the run occupies at least one quarter of the response.
    """

    if not text or len(text) < MIN_RUN_LENGTH:
        return None

    candidates = [
        match
        for match in (_cycle_match(text), _line_match(text))
        if match is not None and _passes_gates(text, match, finish_reason)
    ]
    if not candidates:
        return None
    # Prefer the longest suffix.  For equal runs, the smaller cycle is the more
    # useful exemplar and the direct periodicity is less likely to over-trim.
    return min(candidates, key=lambda match: (match.start, len(match.unit)))


def trim_degeneration(text: str, match: DegenerationMatch) -> str:
    """Trim a matched loop while retaining one exemplar of its repeated unit."""

    if match.end != len(text) or text[match.start : match.end] == "":
        raise ValueError("degeneration match must end at the end of the supplied text")
    if match.start + len(match.unit) > match.end:
        raise ValueError("degeneration unit does not fit inside its match")
    return text[: match.start] + match.unit


def apply_repetition_disposition(
    text: str,
    match: DegenerationMatch,
    policy: RepetitionPolicy,
    *,
    warning_prefix: str = "OCR output",
) -> str:
    """Fail or trim-and-warn once after the policy ladder is exhausted."""

    message = (
        f"{warning_prefix} ended in a {match.run_length}-character repeated "
        f"{match.kind}; exhausted {policy.retry_count} repetition retries"
    )
    if policy.fail_on_repetition:
        raise RepetitionError(message)
    warnings.warn(message + "; trimming repeated suffix", RuntimeWarning, stacklevel=2)
    return trim_degeneration(text, match)


def default_policy(args: Namespace | object | None = None) -> RepetitionPolicy:
    """Build the general runner's changing-sampling retry ladder.

    Specialised models can declare their documented values directly with
    :class:`SamplingAttempt`.  This default remains useful for Ovis and tests.
    """

    retries = int(getattr(args, "repetition_retries", 2)) if args is not None else 2
    if retries < 0:
        raise ValueError("repetition_retries must be non-negative")
    fail = bool(getattr(args, "fail_on_repetition", False)) if args is not None else False

    attempts: list[SamplingAttempt] = []
    for index in range(retries + 1):
        if index == 0:
            attempts.append(
                SamplingAttempt(temperature=0.0, repetition_penalty=None, seed=None)
            )
            continue
        attempts.append(
            SamplingAttempt(
                temperature=min(round(0.1 + (index - 1) * 0.2, 10), 1.0),
                repetition_penalty=1.0 + index * 0.1,
                seed=index,
            )
        )
    return RepetitionPolicy(tuple(attempts), fail_on_repetition=fail)


def apply_sampling_attempt(
    base_kwargs: Mapping[str, object],
    attempt: SamplingAttempt,
) -> dict[str, object]:
    """Functional form of :meth:`SamplingAttempt.apply_to`."""

    return attempt.apply_to(base_kwargs)


__all__ = [
    "DegenerationMatch",
    "GenerationLengthError",
    "GenerationQualityError",
    "MIN_NON_ALNUM_RUN_LENGTH",
    "MIN_RUN_LENGTH",
    "RepetitionError",
    "RepetitionPolicy",
    "SamplingAttempt",
    "apply_repetition_disposition",
    "apply_sampling_attempt",
    "default_policy",
    "detect_degeneration",
    "trim_degeneration",
]
