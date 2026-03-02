from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable
from tqdm import tqdm

from dotenv import load_dotenv

load_dotenv()


SENTENCE_TYPES = ("obligations", "rights", "permissions", "prohibitions", "other")
AGENT_TYPES = ("worker", "firm", "union", "manager")

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

OBLIGATION_SPECIAL_LEMMAS = {
    "require",
    "expect",
    "compel",
    "oblige",
    "obligate",
}
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
    "entitle",
    "reimburse",
    "grant",
}

PROHIBITION_PATTERNS = (
    re.compile(r"\b(?:shall|must|will|may|can)\s+not\b", re.IGNORECASE),
    re.compile(r"\b(?:cannot|can't|prohibit(?:ed|s)?|forbid(?:den|s)?)\b", re.IGNORECASE),
    re.compile(r"\bno\s+(?:employee|employees|worker|workers|union|company|employer)\s+shall\b", re.IGNORECASE),
)
RIGHTS_PATTERNS = (
    re.compile(r"\bentitled\s+to\b", re.IGNORECASE),
    re.compile(r"\bhave\s+the\s+right\s+to\b", re.IGNORECASE),
    re.compile(r"\b(?:shall|will|must)\s+(?:be\s+)?(?:paid|reimbursed|compensated|provided)\b", re.IGNORECASE),
)
PERMISSION_PATTERNS = (
    re.compile(r"\b(?:may|can)\b", re.IGNORECASE),
    re.compile(r"\b(?:permitted|permission|allowed)\b", re.IGNORECASE),
)
OBLIGATION_PATTERNS = (
    re.compile(r"\b(?:shall|must)\b", re.IGNORECASE),
    re.compile(r"\b(?:has|have|had)\s+to\b", re.IGNORECASE),
    re.compile(r"\bought\s+to\b", re.IGNORECASE),
    re.compile(r"\b(?:required|obligat(?:ed|ion)|agree(?:s|d)?\s+to)\b", re.IGNORECASE),
)
NEGATED_OBLIGATION_RIGHTS_PATTERNS = (
    re.compile(
        r"\b(?:may|can|shall|must|will|should)\s+not\s+be\s+"
        r"(?:required|expected|compelled|obliged|obligated)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bnot\s+(?:be\s+)?(?:required|expected|compelled|obliged|obligated)\s+to\b",
        re.IGNORECASE,
    ),
)

AGENT_KEYWORDS = {
    "worker": {
        "employee",
        "employees",
        "worker",
        "workers",
        "individual employee",
        "bargaining unit member",
        "members",
        "member",
    },
    "firm": {
        "company",
        "employer",
        "firm",
        "commission",
        "authority",
        "agency",
        "mta",
    },
    "union": {
        "union",
        "local union",
        "bargaining representative",
        "labor organization",
    },
    "manager": {
        "manager",
        "management",
        "supervisor",
        "director",
        "foreman",
        "chief",
        "department head",
    },
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_path(raw: str, base: Path) -> Path:
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p
    return (base / p).resolve()


def _default_segmentation_root() -> Path:
    cache_dir = os.environ.get("CACHE_DIR", "").strip()
    if cache_dir:
        return _resolve_path(cache_dir, _project_root()) / "02_segmentation_output"
    return (_project_root() / "outputs" / "02_segmentation_output").resolve()


def _parse_segment_num(path: Path) -> int:
    m = re.match(r"segment_(\d+)\.txt$", path.name)
    return int(m.group(1)) if m else 10**9


def _parse_segment_json_num(path: Path) -> int:
    m = re.match(r"segment_(\d+)\.json$", path.name)
    return int(m.group(1)) if m else 10**9


def _list_documents(segmentation_root: Path) -> list[Path]:
    if not segmentation_root.exists():
        return []
    return sorted([p for p in segmentation_root.iterdir() if p.is_dir()])


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


def _compile_agent_matchers() -> dict[str, list[re.Pattern[str]]]:
    compiled: dict[str, list[re.Pattern[str]]] = {}
    for agent_type, phrases in AGENT_KEYWORDS.items():
        compiled[agent_type] = [
            re.compile(rf"\b{re.escape(phrase)}\b", re.IGNORECASE) for phrase in sorted(phrases, key=len, reverse=True)
        ]
    return compiled


AGENT_MATCHERS = _compile_agent_matchers()


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _load_sentence_embedder(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as err:
        raise RuntimeError(
            "sentence-transformers is required for embedding clustering. "
            "Install it first (for example: pip install sentence-transformers)."
        ) from err

    try:
        return SentenceTransformer(model_name)
    except Exception as err:
        raise RuntimeError(
            f"Failed to load sentence-transformers model '{model_name}'."
        ) from err


def _cluster_document_sentences(
    sentence_entries: list[dict[str, Any]],
    *,
    embedder,
    embedding_model: str,
    dbscan_eps: float,
    dbscan_min_samples: int,
) -> dict[str, Any]:
    if not sentence_entries:
        return {
            "embedding_model": embedding_model,
            "dbscan": {
                "eps": dbscan_eps,
                "min_samples": dbscan_min_samples,
                "metric": "cosine",
            },
            "cluster_count": 0,
            "noise_count": 0,
            "cluster_sizes": {},
        }

    try:
        from sklearn.cluster import DBSCAN
    except Exception as err:
        raise RuntimeError(
            "scikit-learn is required for DBSCAN clustering. "
            "Install it first (for example: pip install scikit-learn)."
        ) from err

    texts = [str(entry.get("text", "")).strip() for entry in sentence_entries]
    embeddings = embedder.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    labels = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples, metric="cosine").fit_predict(embeddings)
    label_counts = Counter(int(label) for label in labels)

    for entry, label in zip(sentence_entries, labels):
        sentence_payload = entry["payload"]
        sentence_payload["embedding_cluster_id"] = int(label)
        sentence_payload["embedding_model"] = embedding_model

    cluster_sizes = {
        str(label): int(count)
        for label, count in sorted(label_counts.items(), key=lambda item: item[0])
        if label != -1
    }
    return {
        "embedding_model": embedding_model,
        "dbscan": {
            "eps": dbscan_eps,
            "min_samples": dbscan_min_samples,
            "metric": "cosine",
        },
        "cluster_count": len(cluster_sizes),
        "noise_count": int(label_counts.get(-1, 0)),
        "cluster_sizes": cluster_sizes,
    }


def _apply_embedding_clusters_to_document(
    doc_out_dir: Path,
    *,
    embedder,
    embedding_model: str,
    dbscan_eps: float,
    dbscan_min_samples: int,
) -> dict[str, Any]:
    segment_json_files = sorted(doc_out_dir.glob("segment_*.json"), key=_parse_segment_json_num)
    if not segment_json_files:
        return {
            "follow_up_step": True,
            "embedding_model": embedding_model,
            "dbscan": {
                "eps": dbscan_eps,
                "min_samples": dbscan_min_samples,
                "metric": "cosine",
            },
            "segments_updated": 0,
            "sentences_clustered": 0,
            "cluster_count": 0,
            "noise_count": 0,
            "cluster_sizes": {},
        }

    loaded_payloads: list[tuple[Path, dict[str, Any]]] = []
    sentence_entries: list[dict[str, Any]] = []
    for json_path in segment_json_files:
        payload = json.loads(json_path.read_text(encoding="utf-8", errors="replace"))
        loaded_payloads.append((json_path, payload))
        for sentence in payload.get("sentences", []):
            text = _normalize_space(str(sentence.get("text", "")))
            if not text:
                continue
            sentence_entries.append({"text": text, "payload": sentence})

    cluster_summary = _cluster_document_sentences(
        sentence_entries,
        embedder=embedder,
        embedding_model=embedding_model,
        dbscan_eps=dbscan_eps,
        dbscan_min_samples=dbscan_min_samples,
    )

    for json_path, payload in loaded_payloads:
        payload["embedding_clustering"] = {
            "follow_up_step": True,
            "embedding_model": cluster_summary["embedding_model"],
            "dbscan": cluster_summary["dbscan"],
        }
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "follow_up_step": True,
        **cluster_summary,
        "segments_updated": len(loaded_payloads),
        "sentences_clustered": len(sentence_entries),
    }


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

    # Passive fallback: "X shall be done by the Union."
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
            # Prepositional object fallback.
            if child.dep_ == "prep":
                for grandchild in child.children:
                    if grandchild.dep_ == "pobj":
                        objects.append(grandchild)
    return _expand_with_conjuncts(_dedupe_tokens(objects))


def _modal_features(sent, root) -> dict[str, Any]:
    modal_tokens = [
        tok for tok in sent
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

    # Also catch constructions like "have to" that can be parsed without a modal POS tag.
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
        "modal_verbs": modal_lemmas,
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

    return {
        "root_verb": root.lemma_.lower() if root is not None else "",
        "root_text": root.text if root is not None else "",
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

    # "Workers Union" should primarily count as union, not worker+union.
    if "union" in matches and "worker" in matches:
        if any(re.search(r"\bworkers?\s+union\b", text, re.IGNORECASE) for text in search_texts):
            matches = [m for m in matches if m != "worker"]
    return matches


def _is_negated_obligation_protection(sentence_text: str, clause: dict[str, Any]) -> bool:
    if not bool(clause.get("is_negative", False)):
        return False
    for pat in NEGATED_OBLIGATION_RIGHTS_PATTERNS:
        if pat.search(sentence_text):
            return True
    # Structured fallback for variants that are semantically equivalent:
    # e.g. "employees should not be expected to ..."
    return (
        str(clause.get("special_verb_type", "none")) == "obligation"
        and str(clause.get("voice", "active")) == "passive"
        and str(clause.get("modal_force", "none")) in {"restrictive", "permissive", "mixed"}
    )


def _classify_sentence_type(sentence_text: str, clause: dict[str, Any]) -> tuple[str, list[str]]:
    text = sentence_text.strip()
    evidence: list[str] = []

    def _match(patterns: tuple[re.Pattern[str], ...], label: str) -> bool:
        local = [p.pattern for p in patterns if p.search(text)]
        if local:
            evidence.extend([f"{label}:{p}" for p in local])
            return True
        return False

    modal_force = str(clause.get("modal_force", "none"))
    is_negative = bool(clause.get("is_negative", False))
    special_verb_type = str(clause.get("special_verb_type", "none"))
    root_verb = str(clause.get("root_verb", ""))

    if is_negative:
        evidence.append("dep:negated_modal")
    if modal_force != "none":
        evidence.append(f"modal_force:{modal_force}")
    if special_verb_type != "none":
        evidence.append(f"special_verb:{special_verb_type}")
    if clause.get("voice") == "passive":
        evidence.append("voice:passive")

    # Rights-style protection against imposed duties:
    # "workers may not be required ..." / "employees should not be expected ..."
    if _is_negated_obligation_protection(text, clause):
        evidence.append("rule:negated_obligation_protection")
        return "rights", evidence

    if special_verb_type == "rights" or _match(RIGHTS_PATTERNS, "rights"):
        return "rights", evidence

    # Prohibition has high precedence because it flips legal valence.
    if is_negative and modal_force in {"restrictive", "permissive", "mixed"}:
        return "prohibitions", evidence
    if special_verb_type == "prohibition" or _match(PROHIBITION_PATTERNS, "prohibition"):
        return "prohibitions", evidence

    # Permission only when not negated.
    if not is_negative and (
        special_verb_type == "permission"
        or modal_force == "permissive"
        or _match(PERMISSION_PATTERNS, "permission")
    ):
        return "permissions", evidence

    if (
        special_verb_type == "obligation"
        or modal_force in {"restrictive", "mixed"}
        or _match(OBLIGATION_PATTERNS, "obligation")
    ):
        return "obligations", evidence

    # Fallback for policy-style imperative constructions.
    if root_verb in {"agree", "require"}:
        evidence.append("dep:root_policy_verb")
        return "obligations", evidence

    return "other", evidence


def _classify_sentence(sent) -> dict:
    clause = _reduce_clause(sent)
    subject_phrases = list(clause.get("subject_phrases", []))
    sentence_type, evidence = _classify_sentence_type(sent.text, clause)
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
        "clause_reduction": clause,
    }


def _parse_segment(nlp, segment_text: str) -> list[dict]:
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

    sentence_payloads: list[dict] = []
    for sent_idx, (start_char, end_char, sent_text) in enumerate(spans, start=1):
        sent_doc = nlp(sent_text)
        sent = next((s for s in sent_doc.sents if s.text.strip()), sent_doc[:])
        tokens = []
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

        sentence_payloads.append(
            {
                "sentence_index": sent_idx,
                "start_char": start_char,
                "end_char": end_char,
                "text": sent.text,
                "sentence_source": sentence_source,
                "classification": _classify_sentence(sent),
                "tokens": tokens,
            }
        )
    return sentence_payloads


def run(
    segmentation_root: Path,
    output_root: Path,
    model: str,
    document_id: str | None = None,
    max_segments: int | None = None,
    max_render_sentences: int = 30,
    cluster_follow_up: bool = True,
    embedding_model: str = "all-mpnet-base-v2",
    dbscan_eps: float = 0.35,
    dbscan_min_samples: int = 2,
) -> dict:
    nlp = _load_nlp(model)
    embedder = _load_sentence_embedder(embedding_model) if cluster_follow_up else None

    from spacy import displacy

    docs = _list_documents(segmentation_root)
    if document_id:
        raw_id = str(document_id).strip()
        if re.fullmatch(r"\d+", raw_id):
            raw_id = f"document_{int(raw_id)}"
        docs = [d for d in docs if d.name == raw_id]

    output_root.mkdir(parents=True, exist_ok=True)

    summary = {
        "model": model,
        "segmentation_root": str(segmentation_root),
        "output_root": str(output_root),
        "document_count": len(docs),
        "embedding_clustering": {
            "follow_up_step": bool(cluster_follow_up),
            "embedding_model": embedding_model,
            "dbscan": {
                "eps": dbscan_eps,
                "min_samples": dbscan_min_samples,
                "metric": "cosine",
            },
        },
        "documents": [],
    }

    for doc_dir in docs:
        segments_dir = doc_dir / "segments"
        if not segments_dir.exists():
            continue

        segment_files = sorted(segments_dir.glob("segment_*.txt"), key=_parse_segment_num)
        if max_segments is not None:
            segment_files = segment_files[:max_segments]

        doc_out_dir = output_root / doc_dir.name
        doc_out_dir.mkdir(parents=True, exist_ok=True)

        doc_summary = {
            "document_id": doc_dir.name,
            "segments_processed": 0,
            "sentences_processed": 0,
        }

        for seg_file in tqdm(segment_files, desc=f"Processing {doc_dir.name}", unit="segment"):
            segment_text = seg_file.read_text(encoding="utf-8", errors="replace")
            sentence_payloads = _parse_segment(nlp, segment_text)

            seg_stem = seg_file.stem
            json_path = doc_out_dir / f"{seg_stem}.json"

            for sent_idx, sent_payload in enumerate(sentence_payloads, start=1):
                if sent_idx > max_render_sentences:
                    break
                sentence_html_name = f"{seg_stem}_sentence_{sent_idx}_dep.html"
                sentence_html_path = doc_out_dir / sentence_html_name
                sent_doc = nlp(str(sent_payload.get("text", "")))
                sentence_html = displacy.render(list(sent_doc.sents)[:1], style="dep", page=True)
                sentence_html_path.write_text(sentence_html, encoding="utf-8")
                if sent_idx - 1 < len(sentence_payloads):
                    sentence_payloads[sent_idx - 1]["dep_html_file"] = sentence_html_name

            payload = {
                "document_id": doc_dir.name,
                "segment_file": seg_file.name,
                "model": model,
                "sentence_count": len(sentence_payloads),
                "sentences": sentence_payloads,
            }
            json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

            doc_summary["segments_processed"] += 1
            doc_summary["sentences_processed"] += len(sentence_payloads)

        if cluster_follow_up and embedder is not None:
            doc_summary["embedding_clustering"] = _apply_embedding_clusters_to_document(
                doc_out_dir,
                embedder=embedder,
                embedding_model=embedding_model,
                dbscan_eps=dbscan_eps,
                dbscan_min_samples=dbscan_min_samples,
            )

        summary["documents"].append(doc_summary)

    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Parse segmentation segments into dependency structures and HTML renders."
    )
    parser.add_argument(
        "--segmentation-root",
        type=Path,
        default=_default_segmentation_root(),
        help="Root directory containing segmentation output document folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "output" / "spacy_parse",
        help="Directory where parse JSON and HTML renders will be written.",
    )
    parser.add_argument("--document-id", type=str, default=None, help="Optional document id filter.")
    parser.add_argument("--model", type=str, default="en_core_web_sm", help="spaCy model name.")
    parser.add_argument("--max-segments", type=int, default=None, help="Optional max segments per document.")
    parser.add_argument(
        "--disable-embedding-clustering",
        action="store_true",
        help="Disable the follow-up sentence embedding clustering step.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="all-mpnet-base-v2",
        help="SentenceTransformers model name used in follow-up clustering.",
    )
    parser.add_argument(
        "--dbscan-eps",
        type=float,
        default=0.35,
        help="DBSCAN eps value for sentence embedding clustering.",
    )
    parser.add_argument(
        "--dbscan-min-samples",
        type=int,
        default=2,
        help="DBSCAN min_samples value for sentence embedding clustering.",
    )
    parser.add_argument(
        "--max-render-sentences",
        type=int,
        default=1,
        help="Maximum number of sentences to include in each dependency HTML render.",
    )
    args = parser.parse_args()

    summary = run(
        segmentation_root=args.segmentation_root,
        output_root=args.output_root,
        model=args.model,
        document_id=args.document_id,
        max_segments=args.max_segments,
        max_render_sentences=args.max_render_sentences,
        cluster_follow_up=not args.disable_embedding_clustering,
        embedding_model=args.embedding_model,
        dbscan_eps=args.dbscan_eps,
        dbscan_min_samples=args.dbscan_min_samples,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
