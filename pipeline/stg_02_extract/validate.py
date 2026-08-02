"""Post-extraction validation for wage-table spans.

Two layers of defense against non-wage-table extractions slipping through
langextract:

* ``rejection_reason`` — a deterministic, LLM-free filter that drops objective
  junk (spans with no pay amount at all, header-only fragments, and
  percentage-only pay-policy prose). Cheap and unit-testable.
* ``verify_is_base_wage_table`` — an optional single-call LLM guardrail that
  enforces the semantic "base wages only" scope. A coaching-stipend table is
  structurally identical to a base wage table (dollar amounts + job labels), so
  only a semantic check can reliably separate them.

The deterministic filter is intentionally conservative: many genuine wage
tables (teacher salary schedules, the Pennsylvania "SCHEDULE S" pay grids) use
bare integers with no ``$`` sign, so we detect rate-like numeric content in a
table rather than requiring a currency symbol.
"""

from __future__ import annotations

import re


# A currency figure such as "$27.44" or "$ 15".
_CURRENCY_RE = re.compile(r"\$\s?\d")
# A comma-grouped amount such as "1,905" or "26,873.00" (calendar years are
# never comma-grouped, so this cannot be tripped by a year).
_COMMA_GROUPED_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?")
# A two-decimal money value such as "13.15", "7.90", or "1052.00".
_DECIMAL_MONEY_RE = re.compile(r"(?<!\d)\d+\.\d{2}(?!\d)")
# A standalone integer of 4+ digits, not part of a job code like "J01100".
_INT_4PLUS_RE = re.compile(r"(?<![\w.])\d{4,}(?![\d.])")
# Header tokens that mark a wage-table header row.
_HEADER_TOKENS_RE = re.compile(
    r"\b(STEP|POSITION|RANGE|CLASSIFICATION|RANK|SCHEDULE|GRADE|LANE)\b",
    re.IGNORECASE,
)
# A percentage figure such as "80%" or "8 %".
_PERCENT_RE = re.compile(r"\d\s?%")


def _is_calendar_year(token: str) -> bool:
    return len(token) == 4 and 1900 <= int(token) <= 2099


def has_pay_amount(text: str) -> bool:
    """Return True if the text contains a rate-like monetary pay amount.

    Accepts currency figures, comma-grouped amounts, two-decimal money values,
    and bare integers laid out in a table (salary grids without ``$``). Bare
    4-digit calendar years are ignored so that a prose block that merely
    mentions a year is not mistaken for a rate table.
    """
    if _CURRENCY_RE.search(text):
        return True
    if _COMMA_GROUPED_RE.search(text):
        return True
    if _DECIMAL_MONEY_RE.search(text):
        return True
    for line in text.splitlines():
        pay_like = [
            token
            for token in _INT_4PLUS_RE.findall(line)
            if not _is_calendar_year(token)
        ]
        if len(pay_like) >= 2:
            return True
        if "|" in line and pay_like:
            return True
    return False


def rejection_reason(extraction_text: str) -> str | None:
    """Return a reason string if the span is not a base wage table, else None.

    Deterministic checks only. Returns ``None`` to keep the span; a non-empty
    reason to drop it. Reasons: ``no_pay_amount``, ``header_only_fragment``,
    ``percentage_only_pay_policy``.
    """
    if has_pay_amount(extraction_text):
        return None
    if _PERCENT_RE.search(extraction_text):
        return "percentage_only_pay_policy"
    if _HEADER_TOKENS_RE.search(extraction_text):
        return "header_only_fragment"
    return "no_pay_amount"


_VERIFY_SYSTEM_PROMPT = (
    "You classify text spans extracted from collective bargaining agreements. "
    "Answer only YES or NO."
)

_VERIFY_USER_TEMPLATE = (
    "Is the following span a table of BASE WAGES or SALARIES for job "
    "classifications, salary steps, salary lanes, or pay grades, where every "
    "data row states an absolute pay amount?\n\n"
    "Answer NO if it is any of the following instead:\n"
    "- a stipend or supplemental-pay schedule (coaching, athletic or activity "
    "pay, per-session or per-game pay, advisor or chaperone pay)\n"
    "- a longevity or career-increment clause\n"
    "- a percentage-of-pay or pay-differential rule (e.g. \"85% of regular "
    "rate\" or \"8% above\" another rank)\n"
    "- a header row or list of job titles with no pay amounts\n"
    "- a leave, benefit, allowance, or reimbursement schedule\n\n"
    "Answer only YES or NO.\n\n"
    "Span:\n{span}"
)

_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)


def _parse_yes_no(answer: str) -> bool:
    cleaned = _THINK_BLOCK_RE.sub("", answer or "").strip().lower()
    # Look at the first yes/no token so trailing rationale does not confuse it.
    match = re.search(r"\b(yes|no)\b", cleaned)
    return match is not None and match.group(1) == "yes"


def verify_is_base_wage_table(text: str, client, model_name: str) -> bool:
    """Ask the LLM whether the span is a base wage table.

    ``client`` is an OpenAI-compatible client (as used against the local vLLM
    endpoint). Returns True to keep, False to drop. On any client error the
    span is kept (fail-open) so verification never silently deletes data.
    """
    try:
        response = client.chat.completions.create(
            model=model_name,
            temperature=0,
            messages=[
                {"role": "system", "content": _VERIFY_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": _VERIFY_USER_TEMPLATE.format(span=text),
                },
            ],
        )
    except Exception:
        return True
    answer = response.choices[0].message.content
    return _parse_yes_no(answer)


def make_verify_client(port: int):
    """Build an OpenAI-compatible client for the local vLLM endpoint."""
    from openai import OpenAI

    return OpenAI(api_key="EMPTY", base_url=f"http://localhost:{port}/v1")
