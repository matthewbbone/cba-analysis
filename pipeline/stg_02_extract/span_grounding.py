"""Locate returned passages in contract text without consulting gold labels.

Only whitespace and common output wrappers are normalized. Altered sentences
are retained as ungrounded predictions; no fuzzy or gold-guided alignment occurs.
"""
from bisect import bisect_right
from dataclasses import dataclass, field
import re


@dataclass
class GroundedAnswer:
    spans: list[tuple[int, int]] = field(default_factory=list)
    ungrounded: list[str] = field(default_factory=list)
    ambiguous_passages: int = 0


def _variants(text: str) -> list[str]:
    variants = [text.strip()]
    without_marker = re.sub(r"^(?:>\s*|[-*•]\s+|\d+[.)]\s+)", "", variants[0])
    if without_marker != variants[0]:
        variants.append(without_marker)
    for value in list(variants):
        for left, right in [('"', '"'), ("'", "'"), ('“', '”'), ('**', '**'), ('`', '`'), ('```', '```')]:
            if value.startswith(left) and value.endswith(right) and len(value) > len(left) + len(right):
                variants.append(value[len(left):-len(right)].strip())
    return list(dict.fromkeys(v for v in variants if v))


def ground_answer(answer: str, context: str) -> GroundedAnswer:
    """Return original-context offsets and unmatched passage text.

    Prefer a whole paragraph; if it is not contiguous in the source, try lines
    and then sentences. Merge adjacent fragments within a paragraph only when
    the source gap is whitespace and no unmatched output intervenes. Repeated
    text maps to the next unused occurrence in source order, falling back to the
    first unused occurrence, then a duplicate prediction if all are used.
    """
    result = GroundedAnswer()
    words = list(re.finditer(r"\S+", context))
    normalized = " ".join(word.group() for word in words)
    starts, cursor = [], 0
    for word in words:
        starts.append(cursor)
        cursor += len(word.group()) + 1

    def locate(passage):
        for variant in _variants(passage):
            needle = " ".join(variant.split())
            candidates, start = [], 0
            while needle and (found := normalized.find(needle, start)) != -1:
                end = found + len(needle)
                if ((needle[0].isalnum() and found and normalized[found - 1].isalnum()) or
                    (needle[-1].isalnum() and end < len(normalized) and normalized[end].isalnum())):
                    start = found + 1
                    continue
                first = bisect_right(starts, found) - 1
                last = bisect_right(starts, end - 1) - 1
                candidates.append((words[first].start() + found - starts[first],
                                   words[last].start() + end - starts[last]))
                start = found + 1
            if candidates:
                unused = [span for span in candidates if not any(
                    span[0] < old[1] and old[0] < span[1] for old in result.spans)]
                choices = unused or candidates
                previous_end = result.spans[-1][1] if result.spans else 0
                chosen = next((span for span in choices if span[0] >= previous_end), choices[0])
                result.ambiguous_passages += int(len(candidates) > 1)
                return chosen
        return None

    def abstention(text):
        return any(v.casefold().rstrip('.') == 'no related clause' for v in _variants(text))

    # Remove fence delimiter lines, retaining all prose inside/outside the fence.
    answer = re.sub(r"(?m)^[ \t]*```(?:[A-Za-z0-9_-]+)?[ \t]*$", "", answer)
    for block in re.split(r"\n\s*\n", answer.strip()):
        if not block.strip() or abstention(block):
            continue
        span = locate(block)
        if span is not None:
            result.spans.append(span)
            continue
        can_merge = False
        for line in block.splitlines():
            if not line.strip():
                continue
            if re.match(r"^\s*(?:[-*•]\s+|\d+[.)]\s+)", line):
                can_merge = False
            span = locate(line)
            fragments = [line] if span is not None else re.split(r'(?<=[.!?])\s+(?=[A-Z("“])', line)
            for index, fragment in enumerate(fragments):
                if abstention(fragment):
                    can_merge = False
                    continue
                matched = span if span is not None and index == 0 else locate(fragment)
                if matched is None:
                    result.ungrounded.append(fragment.strip())
                    can_merge = False
                elif can_merge and result.spans[-1][1] <= matched[0] and not context[result.spans[-1][1]:matched[0]].strip():
                    result.spans[-1] = (result.spans[-1][0], matched[1])
                else:
                    result.spans.append(matched)
                if matched is not None:
                    can_merge = True
    return result
