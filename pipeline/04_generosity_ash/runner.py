from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

SENTENCE_TYPES = ("obligations", "rights", "permissions", "prohibitions", "other")
AGENT_TYPES = ("worker", "firm", "union", "manager")
AGENT_BUCKETS = AGENT_TYPES + ("other",)
AUTH_CATEGORIES = ("obligation", "constraint", "permission", "entitlement")
SEGMENT_GENEROSITY_COLUMNS = [
    "document_id",
    "segment_number",
    "clause_type",
    "worker_rights",
    "worker_permissions",
    "worker_prohibitions",
    "worker_obligations",
    "worker_benefit",
    "firm_rights",
    "firm_permissions",
    "firm_prohibitions",
    "firm_obligations",
    "firm_benefit",
    "worker_over_firm_ratio",
    "ratio_status",
]

SUBJECT_DEPS = {
    "nsubj",
    "nsubjpass",
    "csubj",
    "csubjpass",
    "agent",
}
OBJECT_DEPS = {
    "dobj",
    "obj",
    "iobj",
    "pobj",
    "attr",
    "oprd",
    "acomp",
    "xcomp",
    "ccomp",
}

RESTRICTIVE_MODALS = {"shall", "must", "will", "should", "ought"}
PERMISSIVE_MODALS = {"may", "can"}
NEGATION_LEMMAS = {"not", "never", "no"}

# Auth formulas aligned with labor-contracts/src/main04_compute_auth.py.
AUTH_STRICT_MODALS = {"shall", "must", "will"}
AUTH_OBLIGATION_VERBS_PASSIVE = {"require", "expect", "compel", "oblige", "obligate"}
AUTH_OBLIGATION_VERBS_ACTIVE = {"agree", "promise"}
AUTH_CONSTRAINT_VERBS_PASSIVE = {"prohibit", "forbid", "ban", "bar", "restrict", "proscribe"}
AUTH_PERMISSION_VERBS_PASSIVE = {"allow", "permit", "authorize"}
AUTH_ENTITLEMENT_VERBS_ACTIVE = {"receive", "gain", "earn"}
AUTH_ENTITLEMENT_VERBS_PASSIVE = {
    "entitle",
    "give",
    "offer",
    "reimburse",
    "pay",
    "grant",
    "provide",
    "compensate",
    "guarantee",
    "hire",
    "train",
    "supply",
    "protect",
    "cover",
    "inform",
    "notify",
    "grant_off",
    "select",
    "allow_off",
    "award",
    "give_off",
    "pay_out",
    "allow_up",
}
AUTH_PROMISE_VERBS_ACTIVE = {
    "commit",
    "recognize",
    "consent",
    "assent",
    "affirm",
    "assure",
    "guarantee",
    "insure",
    "ensure",
    "stipulate",
    "undertake",
    "pledge",
}

OBLIGATION_SPECIAL_LEMMAS = {"require", "expect", "compel", "oblige", "obligate"}
PROHIBITION_SPECIAL_LEMMAS = {"prohibit", "forbid", "ban", "bar", "restrict", "proscribe"}
PERMISSION_SPECIAL_LEMMAS = {"allow", "permit", "authorize"}
RIGHTS_SPECIAL_LEMMAS = {
    "receive",
    "gain",
    "earn",
    "entitle",
    "grant",
    "offer",
    "provide",
    "compensate",
    "guarantee",
    "reimburse",
    "pay",
    "protect",
    "cover",
    "inform",
    "notify",
    "select",
    "award",
    "hire",
    "train",
    "supply",
}

AGENT_KEYWORDS = {
    "worker": {
        "employee",
        "employees",
        "worker",
        "workers",
        "staff",
        "teacher",
        "nurse",
        "mechanic",
        "operator",
        "steward",
        "personnel",
        "individual employee",
        "bargaining unit member",
        "member",
        "members",
    },
    "firm": {
        "employer",
        "company",
        "board",
        "hospital",
        "corporation",
        "owner",
        "superintendent",
        "firm",
        "commission",
        "authority",
        "agency",
    },
    "union": {
        "union",
        "association",
        "representative",
        "labor organization",
        "bargaining representative",
    },
    "manager": {
        "manager",
        "management",
        "administration",
        "administrator",
        "supervisor",
        "director",
        "principal",
        "foreman",
        "chief",
        "department head",
    },
}

STATEMENT_ROW_COLUMNS = [
    "contract_id",
    "document_id",
    "article_num",
    "segment_number",
    "sentence_num",
    "sentence_index",
    "statement_num",
    "subject",
    "subnorm",
    "subject_phrases",
    "md",
    "modal",
    "modal_verbs",
    "strict_modal",
    "permissive_modal",
    "verb",
    "root_verb_variants",
    "passive",
    "neg",
    "full_sentence",
    "sentence_type",
    "obligation_verb",
    "constraint_verb",
    "permission_verb",
    "entitlement_verb",
    "promise_verb",
    "special_verb",
    "active_verb",
    "obligation",
    "constraint",
    "permission",
    "entitlement",
]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_path(raw: str, base: Path) -> Path:
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p
    return (base / p).resolve()


def _default_cache_dir() -> str:
    return os.environ.get("CACHE_DIR", "").strip()


def _default_input_dir() -> Path:
    cache_dir = _default_cache_dir()
    if cache_dir:
        return _resolve_path(cache_dir, _project_root()) / "02_segmentation_output" / "dol_archive"
    return (_project_root() / "outputs" / "02_segmentation_output" / "dol_archive").resolve()


def _default_output_dir() -> Path:
    cache_dir = _default_cache_dir()
    if cache_dir:
        return _resolve_path(cache_dir, _project_root()) / "04_generosity_ash_output" / "dol_archive"
    return (_project_root() / "outputs" / "04_generosity_ash_output" / "dol_archive").resolve()


def _default_classification_dir() -> Path:
    cache_dir = _default_cache_dir()
    if cache_dir:
        return _resolve_path(cache_dir, _project_root()) / "03_classification_output" / "dol_archive"
    return (_project_root() / "outputs" / "03_classification_output" / "dol_archive").resolve()


def _resolve_io_path(path: Path, cache_dir: Path | None) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    if cache_dir is not None:
        return (cache_dir / candidate).resolve()
    return (_project_root() / candidate).resolve()


def _normalize_document_id(raw: str | None) -> str | None:
    if raw is None:
        return None
    value = str(raw).strip()
    if not value:
        return None
    if re.fullmatch(r"\d+", value):
        return f"document_{int(value)}"
    return value


def _parse_document_num(path: Path) -> int:
    m = re.match(r"document_(\d+)$", path.name)
    return int(m.group(1)) if m else 10**12


def _parse_segment_num(path: Path) -> int:
    m = re.match(r"segment_(\d+)\.txt$", path.name)
    return int(m.group(1)) if m else 10**12


def _compile_agent_matchers() -> dict[str, list[re.Pattern[str]]]:
    compiled: dict[str, list[re.Pattern[str]]] = {}
    for agent_type, phrases in AGENT_KEYWORDS.items():
        compiled[agent_type] = [
            re.compile(rf"\b{re.escape(phrase)}\b", re.IGNORECASE)
            for phrase in sorted(phrases, key=len, reverse=True)
        ]
    return compiled


AGENT_MATCHERS = _compile_agent_matchers()


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _load_nlp(model_name: str):
    try:
        import spacy
    except Exception as err:
        raise RuntimeError(
            "spaCy is required. Install it first (for example: pip install spacy)."
        ) from err

    try:
        nlp = spacy.load(model_name)
    except Exception as err:
        raise RuntimeError(
            f"Failed to load spaCy model '{model_name}'. "
            f"Install it (for example: python -m spacy download {model_name})."
        ) from err

    if "parser" not in nlp.pipe_names and "senter" not in nlp.pipe_names and "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    return nlp


def _line_sentence_spans(text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    for m in re.finditer(r"[^\n]+", text):
        raw = m.group(0)
        if not raw.strip():
            continue
        lead_ws = len(raw) - len(raw.lstrip())
        trail_ws = len(raw) - len(raw.rstrip())
        start = m.start() + lead_ws
        end = m.end() - trail_ws
        if end > start:
            spans.append((start, end, text[start:end]))
    return spans


def _punct_sentence_spans(text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    for m in re.finditer(r"[^.!?\n]+[.!?](?:\s+|$)|[^.!?\n]+$", text, flags=re.MULTILINE):
        raw = m.group(0)
        if not raw.strip():
            continue
        lead_ws = len(raw) - len(raw.lstrip())
        trail_ws = len(raw) - len(raw.rstrip())
        start = m.start() + lead_ws
        end = m.end() - trail_ws
        if end > start:
            spans.append((start, end, text[start:end]))
    return spans


def _expand_phrase(token) -> str:
    parts = [t.text for t in token.subtree if not t.is_punct]
    return _normalize_space(" ".join(parts))


def _find_root(sent):
    root = next((tok for tok in sent if tok.dep_ == "ROOT"), None)
    if root is not None:
        return root
    return sent[0] if len(sent) else None


def _dedupe_tokens(tokens: list) -> list:
    seen = set()
    out = []
    for tok in tokens:
        if tok.i in seen:
            continue
        seen.add(tok.i)
        out.append(tok)
    return out


def _expand_with_conjuncts(tokens: list) -> list:
    expanded = list(tokens)
    for tok in tokens:
        expanded.extend(list(tok.conjuncts))
    return _dedupe_tokens(expanded)


def _extract_subject_tokens(sent, root) -> list:
    subjects = [tok for tok in sent if tok.dep_ in SUBJECT_DEPS]
    if not subjects and root is not None:
        subjects = [tok for tok in root.children if tok.dep_ in SUBJECT_DEPS]

    if not subjects:
        by_objects = []
        for tok in sent:
            if tok.dep_ == "pobj" and tok.head.dep_ == "prep" and tok.head.lemma_.lower() == "by":
                by_objects.append(tok)
        subjects = by_objects
    return _expand_with_conjuncts(subjects)


def _extract_object_tokens(sent, root) -> list:
    objects = [tok for tok in sent if tok.dep_ in OBJECT_DEPS]
    if root is not None:
        for child in root.children:
            if child.dep_ in OBJECT_DEPS:
                objects.append(child)
            if child.dep_ == "prep":
                for grandchild in child.children:
                    if grandchild.dep_ == "pobj":
                        objects.append(grandchild)
    return _expand_with_conjuncts(_dedupe_tokens(objects))


def _modal_features(sent, root) -> dict[str, Any]:
    modal_tokens = [
        tok
        for tok in sent
        if tok.dep_.startswith("aux") and tok.lemma_.lower() in (RESTRICTIVE_MODALS | PERMISSIVE_MODALS)
    ]
    if root is not None and root.lemma_.lower() in (RESTRICTIVE_MODALS | PERMISSIVE_MODALS):
        modal_tokens.append(root)
    modal_tokens = _dedupe_tokens(modal_tokens)

    modal_lemmas = sorted({tok.lemma_.lower() for tok in modal_tokens})
    has_restrictive = any(m in RESTRICTIVE_MODALS for m in modal_lemmas)
    has_permissive = any(m in PERMISSIVE_MODALS for m in modal_lemmas)

    if has_restrictive and has_permissive:
        force = "mixed"
    elif has_restrictive:
        force = "restrictive"
    elif has_permissive:
        force = "permissive"
    else:
        force = "none"

    has_have_to = bool(
        re.search(r"\b(?:has|have|had)\s+to\b", _normalize_space(sent.text), flags=re.IGNORECASE)
    )
    if has_have_to and force == "none":
        force = "restrictive"
        modal_lemmas.append("have_to")

    is_negative = any(tok.dep_ == "neg" or tok.lemma_.lower() in NEGATION_LEMMAS for tok in sent)
    if root is not None:
        is_negative = is_negative or any(child.dep_ == "neg" for child in root.children)

    passive = any(tok.dep_ in {"nsubjpass", "auxpass"} for tok in sent)
    if root is not None:
        passive = passive or any(child.dep_ == "auxpass" for child in root.children)
    voice = "passive" if passive else "active"

    return {
        "modal_verbs": sorted(set(modal_lemmas)),
        "modal_force": force,
        "is_negative": is_negative,
        "voice": voice,
    }


def _special_verb_type(sent, root) -> str:
    text = _normalize_space(sent.text)
    root_lemma = root.lemma_.lower() if root is not None else ""
    lemmas = {tok.lemma_.lower() for tok in sent}

    if root_lemma in PROHIBITION_SPECIAL_LEMMAS or lemmas.intersection(PROHIBITION_SPECIAL_LEMMAS):
        return "prohibition"
    if root_lemma in PERMISSION_SPECIAL_LEMMAS or lemmas.intersection(PERMISSION_SPECIAL_LEMMAS):
        return "permission"
    if (
        root_lemma in RIGHTS_SPECIAL_LEMMAS
        or lemmas.intersection(RIGHTS_SPECIAL_LEMMAS)
        or re.search(r"\b(?:entitled|eligible)\s+to\b", text, flags=re.IGNORECASE)
        or re.search(r"\bhave\s+the\s+right\s+to\b", text, flags=re.IGNORECASE)
    ):
        return "rights"
    if (
        root_lemma in OBLIGATION_SPECIAL_LEMMAS
        or lemmas.intersection(OBLIGATION_SPECIAL_LEMMAS)
        or re.search(r"\b(?:has|have|had)\s+to\b", text, flags=re.IGNORECASE)
        or re.search(r"\bought\s+to\b", text, flags=re.IGNORECASE)
        or re.search(r"\b(?:be|is|are|was|were)\s+(?:required|obliged|obligated)\s+to\b", text, flags=re.IGNORECASE)
    ):
        return "obligation"
    return "none"


def _reduce_clause(sent) -> dict[str, Any]:
    root = _find_root(sent)
    subject_tokens = _extract_subject_tokens(sent, root)
    object_tokens = _extract_object_tokens(sent, root)
    subject_phrases = [_expand_phrase(tok) for tok in subject_tokens]
    object_phrases = [_expand_phrase(tok) for tok in object_tokens]
    subject_phrases = [p for p in subject_phrases if p]
    object_phrases = [p for p in object_phrases if p]

    modal = _modal_features(sent, root)
    special_verb_type = _special_verb_type(sent, root)
    root_variants: list[str] = []
    if root is not None:
        root_lemma = root.lemma_.lower()
        root_variants.append(root_lemma)
        root_variants.extend(f"{root_lemma}_{child.lemma_.lower()}" for child in root.children if child.dep_ == "prt")
    root_variants = list(dict.fromkeys(v for v in root_variants if v))

    return {
        "root_verb": root.lemma_.lower() if root is not None else "",
        "root_text": root.text if root is not None else "",
        "root_verb_variants": root_variants,
        "subject_phrases": list(dict.fromkeys(subject_phrases)),
        "object_phrases": list(dict.fromkeys(object_phrases)),
        "modal_verbs": modal["modal_verbs"],
        "modal_force": modal["modal_force"],
        "is_negative": modal["is_negative"],
        "voice": modal["voice"],
        "special_verb_type": special_verb_type,
    }


def _classify_agent_types(subject_phrases: Iterable[str], sentence_text: str) -> list[str]:
    matches: list[str] = []
    normalized_subjects = [s for s in subject_phrases if s.strip()]
    search_texts = normalized_subjects if normalized_subjects else [sentence_text]
    for agent_type in AGENT_TYPES:
        patterns = AGENT_MATCHERS[agent_type]
        if any(p.search(text) for p in patterns for text in search_texts):
            matches.append(agent_type)

    if "union" in matches and "worker" in matches:
        if any(re.search(r"\bworkers?\s+union\b", text, re.IGNORECASE) for text in search_texts):
            matches = [m for m in matches if m != "worker"]
    return matches


def _compute_auth_features(clause: dict[str, Any]) -> dict[str, Any]:
    root_variants = {
        str(v).strip().lower()
        for v in clause.get("root_verb_variants", [])
        if str(v).strip()
    }
    root_verb = str(clause.get("root_verb", "")).strip().lower()
    if root_verb:
        root_variants.add(root_verb)

    modal_verbs = {
        str(m).strip().lower()
        for m in clause.get("modal_verbs", [])
        if str(m).strip()
    }
    md = any(m != "have_to" for m in modal_verbs)
    strict_modal = md and bool(modal_verbs.intersection(AUTH_STRICT_MODALS))
    permissive_modal = md and not strict_modal

    passive = str(clause.get("voice", "active")) == "passive"
    neg = bool(clause.get("is_negative", False))
    not_passive = not passive

    obligation_verb = (
        (passive and bool(root_variants.intersection(AUTH_OBLIGATION_VERBS_PASSIVE)))
        or (not_passive and bool(root_variants.intersection(AUTH_OBLIGATION_VERBS_ACTIVE)))
    )
    constraint_verb = passive and bool(root_variants.intersection(AUTH_CONSTRAINT_VERBS_PASSIVE))
    permission_verb = passive and bool(root_variants.intersection(AUTH_PERMISSION_VERBS_PASSIVE))
    entitlement_verb = (
        (not_passive and bool(root_variants.intersection(AUTH_ENTITLEMENT_VERBS_ACTIVE)))
        or (passive and bool(root_variants.intersection(AUTH_ENTITLEMENT_VERBS_PASSIVE)))
    )
    promise_verb = not_passive and bool(root_variants.intersection(AUTH_PROMISE_VERBS_ACTIVE))

    special_verb = obligation_verb or constraint_verb or permission_verb or entitlement_verb or promise_verb
    active_verb = not_passive and not special_verb

    modal_primary = ""
    for modal in ("shall", "must", "will", "may", "can", "have_to"):
        if modal in modal_verbs:
            modal_primary = modal
            break

    return {
        "root_verb": root_verb,
        "root_verb_variants": sorted(root_variants),
        "modal_verbs": sorted(modal_verbs),
        "modal": modal_primary,
        "md": bool(md),
        "strict_modal": bool(strict_modal),
        "permissive_modal": bool(permissive_modal),
        "passive": bool(passive),
        "neg": bool(neg),
        "obligation_verb": bool(obligation_verb),
        "constraint_verb": bool(constraint_verb),
        "permission_verb": bool(permission_verb),
        "entitlement_verb": bool(entitlement_verb),
        "promise_verb": bool(promise_verb),
        "special_verb": bool(special_verb),
        "active_verb": bool(active_verb),
    }


def _compute_auth_flags(auth_features: dict[str, Any]) -> dict[str, bool]:
    md = bool(auth_features.get("md", False))
    strict_modal = bool(auth_features.get("strict_modal", False))
    permissive_modal = bool(auth_features.get("permissive_modal", False))
    neg = bool(auth_features.get("neg", False))
    not_neg = not neg
    passive = bool(auth_features.get("passive", False))
    active_verb = bool(auth_features.get("active_verb", False))
    obligation_verb = bool(auth_features.get("obligation_verb", False))
    constraint_verb = bool(auth_features.get("constraint_verb", False))
    permission_verb = bool(auth_features.get("permission_verb", False))
    entitlement_verb = bool(auth_features.get("entitlement_verb", False))

    obligation = (
        (not_neg and strict_modal and active_verb)
        or (not_neg and strict_modal and obligation_verb)
        or (not_neg and (not md) and obligation_verb)
    )
    constraint = (
        (neg and md and active_verb)
        or (not_neg and strict_modal and constraint_verb)
        or (neg and passive and (entitlement_verb or permission_verb))
    )
    permission = (
        (not_neg and ((permissive_modal and active_verb) or permission_verb))
        or (neg and constraint_verb)
    )
    entitlement = (
        (not_neg and entitlement_verb)
        or (neg and obligation_verb)
    )

    return {
        "obligation": bool(obligation),
        "constraint": bool(constraint),
        "permission": bool(permission),
        "entitlement": bool(entitlement),
    }


def _auth_evidence(auth_features: dict[str, Any], auth_flags: dict[str, bool]) -> list[str]:
    evidence: list[str] = [
        f"auth:md={auth_features.get('md', False)}",
        f"auth:strict_modal={auth_features.get('strict_modal', False)}",
        f"auth:permissive_modal={auth_features.get('permissive_modal', False)}",
        f"auth:neg={auth_features.get('neg', False)}",
        f"auth:passive={auth_features.get('passive', False)}",
        f"auth:root={auth_features.get('root_verb', '') or 'none'}",
    ]
    root_variants = auth_features.get("root_verb_variants", [])
    if root_variants:
        evidence.append(f"auth:root_variants={','.join(root_variants)}")
    if auth_features.get("obligation_verb", False):
        evidence.append("auth:obligation_verb")
    if auth_features.get("constraint_verb", False):
        evidence.append("auth:constraint_verb")
    if auth_features.get("permission_verb", False):
        evidence.append("auth:permission_verb")
    if auth_features.get("entitlement_verb", False):
        evidence.append("auth:entitlement_verb")
    if auth_features.get("promise_verb", False):
        evidence.append("auth:promise_verb")
    active_labels = [name for name in AUTH_CATEGORIES if auth_flags.get(name, False)]
    evidence.append("auth:labels=" + ",".join(active_labels) if active_labels else "auth:labels=none")
    return evidence


def _classify_sentence_type(auth_flags: dict[str, bool]) -> str:
    if auth_flags.get("constraint", False):
        return "prohibitions"
    if auth_flags.get("entitlement", False):
        return "rights"
    if auth_flags.get("permission", False):
        return "permissions"
    if auth_flags.get("obligation", False):
        return "obligations"
    return "other"


def _classify_sentence(sent) -> dict[str, Any]:
    clause = _reduce_clause(sent)
    subject_phrases = list(clause.get("subject_phrases", []))
    auth_features = _compute_auth_features(clause)
    auth_flags = _compute_auth_flags(auth_features)
    sentence_type = _classify_sentence_type(auth_flags)
    evidence = _auth_evidence(auth_features, auth_flags)
    subject_agent_types = _classify_agent_types(subject_phrases, sent.text)

    return {
        "sentence_type": sentence_type if sentence_type in SENTENCE_TYPES else "other",
        "subject_agent_types": [a for a in subject_agent_types if a in AGENT_TYPES],
        "subject_phrases": subject_phrases,
        "object_phrases": clause.get("object_phrases", []),
        "root_verb": clause.get("root_verb", ""),
        "modal_verbs": clause.get("modal_verbs", []),
        "modal_force": clause.get("modal_force", "none"),
        "voice": clause.get("voice", "active"),
        "special_verb_type": clause.get("special_verb_type", "none"),
        "classification_evidence": evidence,
        "auth_features": auth_features,
        "auth_category_flags": auth_flags,
        "clause_reduction": clause,
    }


def _extract_tokens(sent) -> list[dict[str, Any]]:
    tokens: list[dict[str, Any]] = []
    for token in sent:
        tokens.append(
            {
                "i": int(token.i),
                "text": token.text,
                "lemma": token.lemma_,
                "pos": token.pos_,
                "tag": token.tag_,
                "dep": token.dep_,
                "head_i": int(token.head.i),
                "head_text": token.head.text,
            }
        )
    return tokens


def _parse_segment(nlp, segment_text: str, include_tokens: bool) -> list[dict[str, Any]]:
    doc = nlp(segment_text)
    spans = [
        (int(sent.start_char), int(sent.end_char), sent.text)
        for sent in doc.sents
        if sent.text.strip()
    ]

    sentence_source = "spacy"
    if len(spans) <= 1:
        line_spans = _line_sentence_spans(segment_text)
        if len(line_spans) > 1:
            spans = line_spans
            sentence_source = "line_fallback"

    if len(spans) <= 1:
        punct_spans = _punct_sentence_spans(segment_text)
        if len(punct_spans) > 1:
            spans = punct_spans
            sentence_source = "punct_fallback"

    sentence_payloads: list[dict[str, Any]] = []
    for sent_idx, (start_char, end_char, sent_text) in enumerate(spans, start=1):
        sent_doc = nlp(sent_text)
        sent_obj = next((s for s in sent_doc.sents if s.text.strip()), sent_doc[:])
        sentence_payload: dict[str, Any] = {
            "sentence_index": sent_idx,
            "start_char": start_char,
            "end_char": end_char,
            "text": sent_obj.text,
            "sentence_source": sentence_source,
            "classification": _classify_sentence(sent_obj),
        }
        if include_tokens:
            sentence_payload["tokens"] = _extract_tokens(sent_obj)
        sentence_payloads.append(sentence_payload)
    return sentence_payloads


def _normalize_agents(raw_agents: Iterable[str]) -> list[str]:
    normalized: list[str] = []
    for raw in raw_agents:
        item = str(raw).strip().lower()
        if item in AGENT_TYPES and item not in normalized:
            normalized.append(item)
    if not normalized:
        normalized.append("other")
    return normalized


def _init_nested_agent_counter() -> dict[str, Counter]:
    return {agent: Counter() for agent in AGENT_BUCKETS}


def _update_counters_from_classification(
    classification: dict[str, Any],
    sentence_type_counts: Counter,
    auth_category_counts: Counter,
    agent_counts: Counter,
    agent_sentence_type_counts: dict[str, Counter],
    agent_auth_category_counts: dict[str, Counter],
) -> None:
    sentence_type = str(classification.get("sentence_type", "other"))
    if sentence_type not in SENTENCE_TYPES:
        sentence_type = "other"
    sentence_type_counts[sentence_type] += 1

    auth_flags = classification.get("auth_category_flags", {})
    if isinstance(auth_flags, dict):
        for category in AUTH_CATEGORIES:
            if bool(auth_flags.get(category, False)):
                auth_category_counts[category] += 1

    agents = _normalize_agents(classification.get("subject_agent_types", []))
    for agent in agents:
        agent_counts[agent] += 1
        agent_sentence_type_counts[agent][sentence_type] += 1
        for category in AUTH_CATEGORIES:
            if bool(auth_flags.get(category, False)):
                agent_auth_category_counts[agent][category] += 1


def _finalize_counter(counter: Counter, keys: Iterable[str]) -> dict[str, int]:
    return {key: int(counter.get(key, 0)) for key in keys}


def _finalize_agent_counter(counter: Counter) -> dict[str, int]:
    return {agent: int(counter.get(agent, 0)) for agent in AGENT_BUCKETS}


def _finalize_nested_agent_counter(counter: dict[str, Counter], keys: Iterable[str]) -> dict[str, dict[str, int]]:
    return {
        agent: {key: int(counter[agent].get(key, 0)) for key in keys}
        for agent in AGENT_BUCKETS
    }


def _segment_totals(sentences: list[dict[str, Any]]) -> dict[str, Any]:
    sentence_type_counts: Counter = Counter()
    auth_category_counts: Counter = Counter()
    agent_counts: Counter = Counter()
    agent_sentence_type_counts = _init_nested_agent_counter()
    agent_auth_category_counts = _init_nested_agent_counter()

    for sentence in sentences:
        classification = sentence.get("classification", {})
        if not isinstance(classification, dict):
            continue
        _update_counters_from_classification(
            classification,
            sentence_type_counts,
            auth_category_counts,
            agent_counts,
            agent_sentence_type_counts,
            agent_auth_category_counts,
        )

    return {
        "sentence_type_counts": _finalize_counter(sentence_type_counts, SENTENCE_TYPES),
        "auth_category_counts": _finalize_counter(auth_category_counts, AUTH_CATEGORIES),
        "agent_counts": _finalize_agent_counter(agent_counts),
        "agent_sentence_type_counts": _finalize_nested_agent_counter(agent_sentence_type_counts, SENTENCE_TYPES),
        "agent_auth_category_counts": _finalize_nested_agent_counter(agent_auth_category_counts, AUTH_CATEGORIES),
    }


def _ratio_details(worker_benefit: int, firm_benefit: int) -> tuple[float | None, str, tuple[int, float]]:
    if firm_benefit == 0:
        if worker_benefit > 0:
            return None, "positive_infinity", (3, 0.0)
        if worker_benefit < 0:
            return None, "negative_infinity", (0, 0.0)
        return None, "undefined", (1, 0.0)
    ratio = worker_benefit / firm_benefit
    return ratio, "finite", (2, ratio)


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _agent_sentence_type_count(
    agent_sentence_type_counts: dict[str, Any],
    agent: str,
    sentence_type: str,
) -> int:
    if sentence_type not in SENTENCE_TYPES:
        return 0
    raw_agent_counts = agent_sentence_type_counts.get(agent, {})
    if not isinstance(raw_agent_counts, dict):
        return 0
    return _safe_int(raw_agent_counts.get(sentence_type, 0) or 0)


def _compute_segment_generosity(segment_totals: dict[str, Any]) -> dict[str, Any]:
    raw_agent_sentence_type_counts = segment_totals.get("agent_sentence_type_counts", {})
    if not isinstance(raw_agent_sentence_type_counts, dict):
        raw_agent_sentence_type_counts = {}

    worker_rights = _agent_sentence_type_count(raw_agent_sentence_type_counts, "worker", "rights")
    worker_permissions = _agent_sentence_type_count(raw_agent_sentence_type_counts, "worker", "permissions")
    worker_prohibitions = _agent_sentence_type_count(raw_agent_sentence_type_counts, "worker", "prohibitions")
    worker_obligations = _agent_sentence_type_count(raw_agent_sentence_type_counts, "worker", "obligations")

    firm_rights = _agent_sentence_type_count(raw_agent_sentence_type_counts, "firm", "rights")
    firm_permissions = _agent_sentence_type_count(raw_agent_sentence_type_counts, "firm", "permissions")
    firm_prohibitions = _agent_sentence_type_count(raw_agent_sentence_type_counts, "firm", "prohibitions")
    firm_obligations = _agent_sentence_type_count(raw_agent_sentence_type_counts, "firm", "obligations")

    worker_benefit = worker_rights + worker_permissions - worker_prohibitions - worker_obligations
    firm_benefit = firm_rights + firm_permissions - firm_prohibitions - firm_obligations
    ratio, ratio_status, _ = _ratio_details(worker_benefit, firm_benefit)

    return {
        "worker_rights": worker_rights,
        "worker_permissions": worker_permissions,
        "worker_prohibitions": worker_prohibitions,
        "worker_obligations": worker_obligations,
        "worker_benefit": worker_benefit,
        "firm_rights": firm_rights,
        "firm_permissions": firm_permissions,
        "firm_prohibitions": firm_prohibitions,
        "firm_obligations": firm_obligations,
        "firm_benefit": firm_benefit,
        "worker_over_firm_ratio": ratio,
        "ratio_status": ratio_status,
    }


def _build_clause_type_document_ratio_rankings(segment_rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in segment_rows:
        document_id = str(row.get("document_id", "")).strip()
        if not document_id:
            continue
        clause_type = str(row.get("clause_type", "")).strip() or "OTHER"

        doc_group = grouped.setdefault(clause_type, {})
        acc = doc_group.setdefault(
            document_id,
            {
                "document_id": document_id,
                "clause_type": clause_type,
                "segment_count": 0,
                "worker_rights_total": 0,
                "worker_permissions_total": 0,
                "worker_prohibitions_total": 0,
                "worker_obligations_total": 0,
                "worker_benefit_total": 0,
                "firm_rights_total": 0,
                "firm_permissions_total": 0,
                "firm_prohibitions_total": 0,
                "firm_obligations_total": 0,
                "firm_benefit_total": 0,
                "finite_segment_ratio_sum": 0.0,
                "finite_segment_ratio_count": 0,
                "segment_ratio_status_counts": Counter(),
            },
        )

        acc["segment_count"] += 1
        for field in (
            "worker_rights",
            "worker_permissions",
            "worker_prohibitions",
            "worker_obligations",
            "worker_benefit",
            "firm_rights",
            "firm_permissions",
            "firm_prohibitions",
            "firm_obligations",
            "firm_benefit",
        ):
            acc[f"{field}_total"] += _safe_int(row.get(field, 0) or 0)

        ratio_status = str(row.get("ratio_status", "undefined"))
        acc["segment_ratio_status_counts"][ratio_status] += 1
        raw_ratio = row.get("worker_over_firm_ratio")
        if isinstance(raw_ratio, (int, float)):
            acc["finite_segment_ratio_sum"] += float(raw_ratio)
            acc["finite_segment_ratio_count"] += 1

    rankings: dict[str, list[dict[str, Any]]] = {}
    for clause_type in sorted(grouped):
        rows: list[dict[str, Any]] = []
        for document_id, acc in grouped[clause_type].items():
            ratio, ratio_status, ratio_sort_key = _ratio_details(
                _safe_int(acc["worker_benefit_total"]),
                _safe_int(acc["firm_benefit_total"]),
            )
            finite_count = _safe_int(acc["finite_segment_ratio_count"])
            avg_segment_ratio = (
                float(acc["finite_segment_ratio_sum"]) / finite_count
                if finite_count > 0
                else None
            )
            rows.append(
                {
                    "document_id": document_id,
                    "clause_type": clause_type,
                    "segment_count": _safe_int(acc["segment_count"]),
                    "worker_rights_total": _safe_int(acc["worker_rights_total"]),
                    "worker_permissions_total": _safe_int(acc["worker_permissions_total"]),
                    "worker_prohibitions_total": _safe_int(acc["worker_prohibitions_total"]),
                    "worker_obligations_total": _safe_int(acc["worker_obligations_total"]),
                    "worker_benefit_total": _safe_int(acc["worker_benefit_total"]),
                    "firm_rights_total": _safe_int(acc["firm_rights_total"]),
                    "firm_permissions_total": _safe_int(acc["firm_permissions_total"]),
                    "firm_prohibitions_total": _safe_int(acc["firm_prohibitions_total"]),
                    "firm_obligations_total": _safe_int(acc["firm_obligations_total"]),
                    "firm_benefit_total": _safe_int(acc["firm_benefit_total"]),
                    "worker_over_firm_ratio": ratio,
                    "ratio_status": ratio_status,
                    "avg_finite_segment_ratio": avg_segment_ratio,
                    "finite_segment_ratio_count": finite_count,
                    "segment_ratio_status_counts": {
                        key: int(value)
                        for key, value in Counter(acc["segment_ratio_status_counts"]).items()
                    },
                    "_ratio_sort_key": ratio_sort_key,
                }
            )

        rows.sort(
            key=lambda row: (
                row["_ratio_sort_key"][0],
                row["_ratio_sort_key"][1],
                row["worker_benefit_total"],
                -row["firm_benefit_total"],
                row["segment_count"],
                row["document_id"],
            ),
            reverse=True,
        )
        for rank, row in enumerate(rows, start=1):
            row["rank"] = rank
            row.pop("_ratio_sort_key", None)
        rankings[clause_type] = rows
    return rankings


def _build_document_composite_clause_scores(
    clause_type_document_rankings: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    per_document_ratios: dict[str, list[float]] = {}
    for ranking_rows in clause_type_document_rankings.values():
        if not isinstance(ranking_rows, list):
            continue
        for row in ranking_rows:
            if not isinstance(row, dict):
                continue
            document_id = str(row.get("document_id", "")).strip()
            if not document_id:
                continue
            raw_ratio = row.get("worker_over_firm_ratio")
            if not isinstance(raw_ratio, (int, float)):
                continue
            ratio = float(raw_ratio)
            if math.isnan(ratio):
                continue
            per_document_ratios.setdefault(document_id, []).append(ratio)

    out: dict[str, dict[str, Any]] = {}
    for document_id, ratios in per_document_ratios.items():
        if not ratios:
            continue
        out[document_id] = {
            "composite_clause_score": float(sum(ratios) / len(ratios)),
            "composite_clause_type_count": int(len(ratios)),
        }
    return out


def _statement_rows_for_sentence(
    *,
    document_id: str,
    segment_number: int,
    sentence_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    classification = sentence_payload.get("classification", {})
    if not isinstance(classification, dict):
        return []

    auth_features = classification.get("auth_features", {})
    auth_flags = classification.get("auth_category_flags", {})
    subject_phrases = classification.get("subject_phrases", [])
    if not isinstance(subject_phrases, list):
        subject_phrases = []
    subject_phrases = [str(p) for p in subject_phrases if str(p).strip()]

    agents = _normalize_agents(classification.get("subject_agent_types", []))
    sentence_index = int(sentence_payload.get("sentence_index", 0) or 0)
    sentence_text = str(sentence_payload.get("text", ""))

    rows: list[dict[str, Any]] = []
    for statement_num, agent in enumerate(agents, start=1):
        row = {
            "contract_id": document_id,
            "document_id": document_id,
            "article_num": segment_number,
            "segment_number": segment_number,
            "sentence_num": sentence_index,
            "sentence_index": sentence_index,
            "statement_num": statement_num,
            "subject": subject_phrases[0] if subject_phrases else "",
            "subnorm": agent,
            "subject_phrases": "|".join(subject_phrases),
            "md": bool(auth_features.get("md", False)),
            "modal": str(auth_features.get("modal", "")),
            "modal_verbs": "|".join(str(v) for v in auth_features.get("modal_verbs", [])),
            "strict_modal": bool(auth_features.get("strict_modal", False)),
            "permissive_modal": bool(auth_features.get("permissive_modal", False)),
            "verb": str(auth_features.get("root_verb", "")),
            "root_verb_variants": "|".join(str(v) for v in auth_features.get("root_verb_variants", [])),
            "passive": bool(auth_features.get("passive", False)),
            "neg": bool(auth_features.get("neg", False)),
            "full_sentence": sentence_text,
            "sentence_type": str(classification.get("sentence_type", "other")),
            "obligation_verb": bool(auth_features.get("obligation_verb", False)),
            "constraint_verb": bool(auth_features.get("constraint_verb", False)),
            "permission_verb": bool(auth_features.get("permission_verb", False)),
            "entitlement_verb": bool(auth_features.get("entitlement_verb", False)),
            "promise_verb": bool(auth_features.get("promise_verb", False)),
            "special_verb": bool(auth_features.get("special_verb", False)),
            "active_verb": bool(auth_features.get("active_verb", False)),
            "obligation": bool(auth_flags.get("obligation", False)),
            "constraint": bool(auth_flags.get("constraint", False)),
            "permission": bool(auth_flags.get("permission", False)),
            "entitlement": bool(auth_flags.get("entitlement", False)),
        }
        rows.append(row)
    return rows


class GenerosityAshRunner:
    def __init__(
        self,
        *,
        cache_dir: str | Path | None,
        input_dir: Path,
        classification_dir: Path,
        output_dir: Path,
        model: str,
        include_tokens: bool,
    ):
        cache_base = None
        if cache_dir is not None and str(cache_dir).strip():
            cache_base = _resolve_path(str(cache_dir), _project_root())

        self.cache_dir = cache_base
        self.input_dir = _resolve_io_path(input_dir, cache_base)
        self.classification_dir = _resolve_io_path(classification_dir, cache_base)
        self.output_dir = _resolve_io_path(output_dir, cache_base)
        self.model = model
        self.include_tokens = include_tokens
        self.nlp = _load_nlp(self.model)

    def _list_document_dirs(self) -> list[Path]:
        if not self.input_dir.exists():
            return []
        candidates = [
            p for p in self.input_dir.glob("document_*")
            if p.is_dir() and re.match(r"^document_\d+$", p.name)
        ]
        return sorted(candidates, key=_parse_document_num)

    @staticmethod
    def _list_segment_files(doc_dir: Path) -> list[Path]:
        return sorted(
            [
                p for p in doc_dir.glob("segments/segment_*.txt")
                if p.is_file() and _parse_segment_num(p) != 10**12
            ],
            key=_parse_segment_num,
        )

    def _load_segment_clause_type(self, document_id: str, segment_number: int) -> str:
        payload_path = self.classification_dir / document_id / f"segment_{segment_number}.json"
        if not payload_path.exists():
            return "OTHER"
        try:
            payload = json.loads(payload_path.read_text(encoding="utf-8"))
        except Exception:
            return "OTHER"
        if not isinstance(payload, dict):
            return "OTHER"

        labels = payload.get("labels", [])
        if isinstance(labels, list):
            for label in labels:
                normalized = str(label).strip()
                if normalized:
                    return normalized

        label = str(payload.get("label", "")).strip()
        return label if label else "OTHER"

    def run(
        self,
        *,
        sample_size: int | None,
        seed: int,
        document_id: str | None,
        max_segments: int | None,
    ) -> dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        statement_rows_path = self.output_dir / "statement_rows.csv"
        segment_generosity_path = self.output_dir / "segment_generosity_scores.csv"

        doc_dirs = self._list_document_dirs()
        if document_id:
            doc_dirs = [d for d in doc_dirs if d.name == document_id]

        if sample_size is not None and sample_size < len(doc_dirs):
            random.seed(seed)
            doc_dirs = random.sample(doc_dirs, sample_size)
            doc_dirs = sorted(doc_dirs, key=_parse_document_num)

        summary = {
            "model": self.model,
            "input_dir": str(self.input_dir),
            "classification_dir": str(self.classification_dir),
            "output_dir": str(self.output_dir),
            "include_tokens": bool(self.include_tokens),
            "document_count": len(doc_dirs),
            "documents_processed": 0,
            "segments_processed": 0,
            "sentences_processed": 0,
            "statement_rows_written": 0,
            "segment_generosity_rows_written": 0,
            "sentence_type_counts": {k: 0 for k in SENTENCE_TYPES},
            "auth_category_counts": {k: 0 for k in AUTH_CATEGORIES},
            "agent_counts": {k: 0 for k in AGENT_BUCKETS},
            "agent_sentence_type_counts": {a: {k: 0 for k in SENTENCE_TYPES} for a in AGENT_BUCKETS},
            "agent_auth_category_counts": {a: {k: 0 for k in AUTH_CATEGORIES} for a in AGENT_BUCKETS},
            "document_composite_clause_scores": {},
            "documents": [],
        }

        global_sentence_type_counts: Counter = Counter()
        global_auth_category_counts: Counter = Counter()
        global_agent_counts: Counter = Counter()
        global_agent_sentence_type_counts = _init_nested_agent_counter()
        global_agent_auth_category_counts = _init_nested_agent_counter()
        global_worker_benefit_total = 0
        global_firm_benefit_total = 0
        segment_generosity_rows: list[dict[str, Any]] = []
        doc_summary_paths: dict[str, Path] = {}

        with (
            statement_rows_path.open("w", newline="", encoding="utf-8") as statement_file,
            segment_generosity_path.open("w", newline="", encoding="utf-8") as generosity_file,
        ):
            statement_writer = csv.DictWriter(statement_file, fieldnames=STATEMENT_ROW_COLUMNS)
            statement_writer.writeheader()
            segment_generosity_writer = csv.DictWriter(generosity_file, fieldnames=SEGMENT_GENEROSITY_COLUMNS)
            segment_generosity_writer.writeheader()

            for doc_dir in doc_dirs:
                doc_id = doc_dir.name
                doc_out_dir = self.output_dir / doc_id
                doc_out_dir.mkdir(parents=True, exist_ok=True)

                segment_files = self._list_segment_files(doc_dir)
                if max_segments is not None:
                    segment_files = segment_files[:max(0, int(max_segments))]

                doc_sentence_type_counts: Counter = Counter()
                doc_auth_category_counts: Counter = Counter()
                doc_agent_counts: Counter = Counter()
                doc_agent_sentence_type_counts = _init_nested_agent_counter()
                doc_agent_auth_category_counts = _init_nested_agent_counter()
                doc_sentences_processed = 0
                doc_statement_rows = 0
                doc_worker_benefit_total = 0
                doc_firm_benefit_total = 0
                doc_segment_generosity_rows = 0

                for segment_path in tqdm(segment_files, desc=f"04_generosity_ash {doc_id}", unit="segment"):
                    segment_number = _parse_segment_num(segment_path)
                    if segment_number == 10**12:
                        continue

                    segment_text = segment_path.read_text(encoding="utf-8", errors="replace")
                    sentence_payloads = _parse_segment(self.nlp, segment_text, self.include_tokens)
                    segment_totals = _segment_totals(sentence_payloads)
                    segment_clause_type = self._load_segment_clause_type(doc_id, segment_number)
                    segment_generosity = _compute_segment_generosity(segment_totals)
                    segment_generosity_row = {
                        "document_id": doc_id,
                        "segment_number": segment_number,
                        "clause_type": segment_clause_type,
                        **segment_generosity,
                    }

                    statement_rows: list[dict[str, Any]] = []
                    for sentence_payload in sentence_payloads:
                        classification = sentence_payload.get("classification", {})
                        if isinstance(classification, dict):
                            _update_counters_from_classification(
                                classification,
                                doc_sentence_type_counts,
                                doc_auth_category_counts,
                                doc_agent_counts,
                                doc_agent_sentence_type_counts,
                                doc_agent_auth_category_counts,
                            )
                            _update_counters_from_classification(
                                classification,
                                global_sentence_type_counts,
                                global_auth_category_counts,
                                global_agent_counts,
                                global_agent_sentence_type_counts,
                                global_agent_auth_category_counts,
                            )
                        statement_rows.extend(
                            _statement_rows_for_sentence(
                                document_id=doc_id,
                                segment_number=segment_number,
                                sentence_payload=sentence_payload,
                            )
                        )

                    statement_writer.writerows(statement_rows)
                    segment_generosity_writer.writerow(segment_generosity_row)
                    segment_generosity_rows.append(segment_generosity_row)

                    payload = {
                        "document_id": doc_id,
                        "segment_file": segment_path.name,
                        "segment_number": segment_number,
                        "clause_type": segment_clause_type,
                        "model": self.model,
                        "sentence_count": len(sentence_payloads),
                        "sentences": sentence_payloads,
                        "segment_generosity": segment_generosity,
                        "segment_totals": {
                            **segment_totals,
                            "statement_rows": len(statement_rows),
                        },
                    }
                    out_path = doc_out_dir / f"segment_{segment_number}.json"
                    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

                    summary["segments_processed"] += 1
                    summary["sentences_processed"] += len(sentence_payloads)
                    summary["statement_rows_written"] += len(statement_rows)
                    summary["segment_generosity_rows_written"] += 1
                    doc_sentences_processed += len(sentence_payloads)
                    doc_statement_rows += len(statement_rows)
                    doc_segment_generosity_rows += 1
                    doc_worker_benefit_total += _safe_int(segment_generosity.get("worker_benefit", 0) or 0)
                    doc_firm_benefit_total += _safe_int(segment_generosity.get("firm_benefit", 0) or 0)
                    global_worker_benefit_total += _safe_int(segment_generosity.get("worker_benefit", 0) or 0)
                    global_firm_benefit_total += _safe_int(segment_generosity.get("firm_benefit", 0) or 0)

                doc_summary = {
                    "document_id": doc_id,
                    "segments_processed": len(segment_files),
                    "sentences_processed": doc_sentences_processed,
                    "statement_rows_written": doc_statement_rows,
                    "segment_generosity_rows_written": doc_segment_generosity_rows,
                    "sentence_type_counts": _finalize_counter(doc_sentence_type_counts, SENTENCE_TYPES),
                    "auth_category_counts": _finalize_counter(doc_auth_category_counts, AUTH_CATEGORIES),
                    "agent_counts": _finalize_agent_counter(doc_agent_counts),
                    "agent_sentence_type_counts": _finalize_nested_agent_counter(doc_agent_sentence_type_counts, SENTENCE_TYPES),
                    "agent_auth_category_counts": _finalize_nested_agent_counter(doc_agent_auth_category_counts, AUTH_CATEGORIES),
                }
                doc_ratio, doc_ratio_status, _ = _ratio_details(doc_worker_benefit_total, doc_firm_benefit_total)
                doc_summary["segment_generosity"] = {
                    "worker_benefit_total": int(doc_worker_benefit_total),
                    "firm_benefit_total": int(doc_firm_benefit_total),
                    "worker_over_firm_ratio": doc_ratio,
                    "ratio_status": doc_ratio_status,
                }
                doc_summary_path = doc_out_dir / "document_summary.json"
                doc_summary_path.write_text(
                    json.dumps(doc_summary, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                doc_summary_paths[doc_id] = doc_summary_path
                summary["documents"].append(doc_summary)
                summary["documents_processed"] += 1

        summary["sentence_type_counts"] = _finalize_counter(global_sentence_type_counts, SENTENCE_TYPES)
        summary["auth_category_counts"] = _finalize_counter(global_auth_category_counts, AUTH_CATEGORIES)
        summary["agent_counts"] = _finalize_agent_counter(global_agent_counts)
        summary["agent_sentence_type_counts"] = _finalize_nested_agent_counter(global_agent_sentence_type_counts, SENTENCE_TYPES)
        summary["agent_auth_category_counts"] = _finalize_nested_agent_counter(global_agent_auth_category_counts, AUTH_CATEGORIES)
        overall_ratio, overall_ratio_status, _ = _ratio_details(global_worker_benefit_total, global_firm_benefit_total)
        summary["overall_segment_generosity"] = {
            "worker_benefit_total": int(global_worker_benefit_total),
            "firm_benefit_total": int(global_firm_benefit_total),
            "worker_over_firm_ratio": overall_ratio,
            "ratio_status": overall_ratio_status,
        }

        clause_type_document_rankings = _build_clause_type_document_ratio_rankings(segment_generosity_rows)
        document_composite_clause_scores = _build_document_composite_clause_scores(clause_type_document_rankings)
        summary["document_composite_clause_scores"] = document_composite_clause_scores

        for ranking_rows in clause_type_document_rankings.values():
            if not isinstance(ranking_rows, list):
                continue
            for row in ranking_rows:
                if not isinstance(row, dict):
                    continue
                document_id = str(row.get("document_id", "")).strip()
                composite_payload = document_composite_clause_scores.get(document_id, {})
                row["composite_clause_score"] = composite_payload.get("composite_clause_score")
                row["composite_clause_type_count"] = int(composite_payload.get("composite_clause_type_count", 0))

        for doc_summary in summary["documents"]:
            if not isinstance(doc_summary, dict):
                continue
            doc_id = str(doc_summary.get("document_id", "")).strip()
            composite_payload = document_composite_clause_scores.get(doc_id, {})
            doc_summary["composite_clause_score"] = composite_payload.get("composite_clause_score")
            doc_summary["composite_clause_type_count"] = int(composite_payload.get("composite_clause_type_count", 0))

            doc_summary_path = doc_summary_paths.get(doc_id)
            if doc_summary_path is not None:
                doc_summary_path.write_text(
                    json.dumps(doc_summary, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

        summary["clause_type_document_ratio_rankings"] = clause_type_document_rankings
        # Backward-compatible key name retained for existing consumers.
        summary["document_clause_type_ratio_rankings"] = clause_type_document_rankings

        clause_type_rankings_path = self.output_dir / "clause_type_document_ratio_rankings.json"
        clause_type_rankings_path.write_text(
            json.dumps(clause_type_document_rankings, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        legacy_clause_type_rankings_path = self.output_dir / "document_clause_type_ratio_rankings.json"
        legacy_clause_type_rankings_path.write_text(
            json.dumps(clause_type_document_rankings, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        summary_path = self.output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run Ash-style auth categorization (obligation/constraint/permission/entitlement) "
            "on segmented CBA text."
        )
    )
    parser.add_argument("--cache-dir", type=str, default=_default_cache_dir())
    parser.add_argument("--input-dir", type=Path, default=_default_input_dir())
    parser.add_argument("--classification-dir", type=Path, default=_default_classification_dir())
    parser.add_argument("--output-dir", type=Path, default=_default_output_dir())
    parser.add_argument("--model", type=str, default="en_core_web_sm")
    parser.add_argument("--document-id", type=str, default=None)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-segments", type=int, default=None)
    parser.add_argument(
        "--include-tokens",
        action="store_true",
        help="Include token-level dependency data in per-segment JSON outputs.",
    )
    args = parser.parse_args()

    runner = GenerosityAshRunner(
        cache_dir=args.cache_dir,
        input_dir=args.input_dir,
        classification_dir=args.classification_dir,
        output_dir=args.output_dir,
        model=args.model,
        include_tokens=args.include_tokens,
    )
    summary = runner.run(
        sample_size=args.sample_size,
        seed=args.seed,
        document_id=_normalize_document_id(args.document_id),
        max_segments=args.max_segments,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
