import json
import random
import re
from pathlib import Path
from collections import Counter

import streamlit as st
import streamlit.components.v1 as components


APP_DIR = Path(__file__).resolve().parent
DEFAULT_PARSE_OUTPUT_DIR = APP_DIR.parent / "experiments" / "sentence_parse" / "output" / "spacy_parse"
TABLE2_CLAUSE_ORDER = ("obligations", "rights", "permissions", "prohibitions", "other")
TABLE2_AGENT_ORDER = ("worker", "firm", "union", "manager")
CLUSTER_VIEW_TABS = ("Sentence Review", "Cluster Word Clouds")
CLUSTER_NOISE_ID = -1


def _safe_read_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _safe_read_text(path: Path):
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return None


def _extract_sentence_render(html_doc: str, sentence_index: int) -> str | None:
    # Support both old and new displacy page wrappers.
    figures = re.findall(r"(<figure[^>]*>[\s\S]*?</figure>)", html_doc, flags=re.IGNORECASE)
    if not figures:
        # Fallback: some outputs may only expose top-level SVG blocks.
        figures = re.findall(
            r"(<svg[^>]*class=\"[^\"]*displacy[^\"]*\"[^>]*>[\s\S]*?</svg>)",
            html_doc,
            flags=re.IGNORECASE,
        )
    idx = sentence_index - 1
    if 0 <= idx < len(figures):
        return f"<html><body>{figures[idx]}</body></html>"
    return None


def _sentence_html_path(doc_dir: Path, seg_json_path: Path, sent_payload: dict, sent_num: int) -> Path:
    dep_html_file = sent_payload.get("dep_html_file") if isinstance(sent_payload, dict) else None
    if isinstance(dep_html_file, str) and dep_html_file.strip():
        return doc_dir / dep_html_file.strip()
    return seg_json_path.with_name(f"{seg_json_path.stem}_sentence_{sent_num}_dep.html")


def _doc_ids(root: Path) -> list[str]:
    if not root.exists():
        return []
    return sorted([p.name for p in root.iterdir() if p.is_dir() and p.name.startswith("document_")])


def _segment_json_paths(doc_dir: Path) -> list[Path]:
    return sorted(
        [p for p in doc_dir.glob("segment_*.json") if p.is_file()],
        key=lambda p: int(p.stem.split("_")[1]) if p.stem.split("_")[1].isdigit() else 10**9,
    )


def _segment_num_from_json(path: Path) -> str:
    stem = path.stem
    if "_" not in stem:
        return stem
    return stem.split("_", 1)[1]


def _sentence_classification(sent: dict) -> dict:
    raw = sent.get("classification", {}) if isinstance(sent, dict) else {}
    if not isinstance(raw, dict):
        raw = {}

    sentence_type = str(raw.get("sentence_type", "other") or "other")
    agent_types = raw.get("subject_agent_types", [])
    subject_phrases = raw.get("subject_phrases", [])
    evidence = raw.get("classification_evidence", [])

    if not isinstance(agent_types, list):
        agent_types = []
    if not isinstance(subject_phrases, list):
        subject_phrases = []
    if not isinstance(evidence, list):
        evidence = []

    return {
        "sentence_type": sentence_type,
        "subject_agent_types": [str(v) for v in agent_types if str(v).strip()],
        "subject_phrases": [str(v) for v in subject_phrases if str(v).strip()],
        "classification_evidence": [str(v) for v in evidence if str(v).strip()],
    }


def _matching_sentence_indices(sentences: list[dict], type_filter: str, agent_filter: str) -> list[int]:
    matched = []
    for i, sent in enumerate(sentences, start=1):
        cls = _sentence_classification(sent)
        if type_filter != "all" and cls["sentence_type"] != type_filter:
            continue
        if agent_filter != "all" and agent_filter not in cls["subject_agent_types"]:
            continue
        matched.append(i)
    return matched


def _segment_sentence_rows(sentences: list[dict], segment_label: str) -> list[dict]:
    rows = []
    for i, sent in enumerate(sentences, start=1):
        cls = _sentence_classification(sent)
        rows.append(
            {
                "segment": segment_label,
                "sentence_index": i,
                "sentence_type": cls["sentence_type"],
                "agent_types": ", ".join(cls["subject_agent_types"]) if cls["subject_agent_types"] else "",
                "text": str(sent.get("text", "")),
            }
        )
    return rows


@st.cache_data(show_spinner=False)
def _document_sentence_rows(parse_root_str: str, doc_id: str) -> list[dict]:
    parse_root = Path(parse_root_str)
    doc_dir = parse_root / doc_id
    rows = []
    for seg_json_path in _segment_json_paths(doc_dir):
        payload = _safe_read_json(seg_json_path)
        if not isinstance(payload, dict):
            continue
        sentences = payload.get("sentences", [])
        if not isinstance(sentences, list):
            continue
        seg_label = f"segment_{_segment_num_from_json(seg_json_path)}"
        rows.extend(_segment_sentence_rows(sentences, seg_label))
    return rows


@st.cache_data(show_spinner=False)
def _corpus_bucket_sentence_rows(parse_root_str: str) -> list[dict]:
    parse_root = Path(parse_root_str)
    rows: list[dict] = []

    for doc_id in _doc_ids(parse_root):
        doc_dir = parse_root / doc_id
        for seg_json_path in _segment_json_paths(doc_dir):
            payload = _safe_read_json(seg_json_path)
            if not isinstance(payload, dict):
                continue
            sentences = payload.get("sentences", [])
            if not isinstance(sentences, list):
                continue

            seg_label = f"segment_{_segment_num_from_json(seg_json_path)}"
            for sent_idx, sent in enumerate(sentences, start=1):
                if not isinstance(sent, dict):
                    continue
                text = str(sent.get("text", "")).strip()
                if not text:
                    continue

                cls = _sentence_classification(sent)
                sentence_type = str(cls.get("sentence_type", "other")).strip().lower()
                if sentence_type not in TABLE2_CLAUSE_ORDER:
                    sentence_type = "other"

                raw_agents = cls.get("subject_agent_types", [])
                if not isinstance(raw_agents, list):
                    raw_agents = []
                agents: list[str] = []
                seen = set()
                for agent in raw_agents:
                    agent = str(agent).strip().lower()
                    if agent in TABLE2_AGENT_ORDER and agent not in seen:
                        seen.add(agent)
                        agents.append(agent)

                for agent in agents:
                    rows.append(
                        {
                            "doc_id": doc_id,
                            "segment": seg_label,
                            "sentence_index": sent_idx,
                            "agent": agent,
                            "sentence_type": sentence_type,
                            "text": text,
                            "agent_types": agents,
                            "classification_evidence": cls.get("classification_evidence", []),
                            "subject_phrases": cls.get("subject_phrases", []),
                        }
                    )

    return rows


def _render_random_bucket_sentence_view(parse_root: Path):
    st.subheader("Random Bucket Sentence")
    st.caption("Sample a random sentence from a specific Agent x Sentence Type bucket across the full corpus.")

    bucket_rows = _corpus_bucket_sentence_rows(str(parse_root))
    if not bucket_rows:
        st.warning("No agent/sentence-type bucket assignments found in this parse output root.")
        return

    bucket_counts = Counter((row["agent"], row["sentence_type"]) for row in bucket_rows)
    c1, c2 = st.columns(2)
    with c1:
        selected_agent = st.selectbox(
            "Agent bucket",
            options=list(TABLE2_AGENT_ORDER),
            format_func=lambda x: x.title(),
            key="random_bucket_agent",
        )
    with c2:
        selected_clause = st.selectbox(
            "Sentence type bucket",
            options=list(TABLE2_CLAUSE_ORDER),
            format_func=lambda x: x.title(),
            key="random_bucket_clause",
        )

    matching_rows = [
        row
        for row in bucket_rows
        if row["agent"] == selected_agent and row["sentence_type"] == selected_clause
    ]
    st.caption(
        f"Bucket size: {len(matching_rows):,} sentence assignment(s) "
        f"(table count: {bucket_counts.get((selected_agent, selected_clause), 0):,})"
    )

    if not matching_rows:
        st.info("No sentences available for this bucket.")
        return

    selection_key = f"random_bucket_selection::{str(parse_root)}::{selected_agent}::{selected_clause}"
    if st.button("Draw random sentence", key="draw_random_bucket_sentence"):
        st.session_state[selection_key] = random.choice(matching_rows)

    selected_row = st.session_state.get(selection_key)
    if not isinstance(selected_row, dict):
        selected_row = random.choice(matching_rows)
        st.session_state[selection_key] = selected_row

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Document", selected_row.get("doc_id", ""))
    d2.metric("Segment", selected_row.get("segment", ""))
    d3.metric("Sentence index", selected_row.get("sentence_index", ""))
    d4.metric("Bucket", f"{selected_agent.title()} x {selected_clause.title()}")

    st.code(str(selected_row.get("text", "")))

    e1, e2 = st.columns(2)
    with e1:
        st.caption("Agent types on sentence")
        all_agents = selected_row.get("agent_types", [])
        if isinstance(all_agents, list) and all_agents:
            st.write(", ".join(str(x) for x in all_agents))
        else:
            st.write("None")
    with e2:
        st.caption("Classification evidence")
        evidence = selected_row.get("classification_evidence", [])
        if isinstance(evidence, list) and evidence:
            st.write("\n".join(f"- {str(x)}" for x in evidence))
        else:
            st.write("None")

def _safe_cluster_id(raw) -> int | None:
    try:
        if raw is None:
            return None
        return int(raw)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def _document_cluster_sentences(parse_root_str: str, doc_id: str) -> dict[int, list[dict]]:
    parse_root = Path(parse_root_str)
    doc_dir = parse_root / doc_id
    clusters: dict[int, list[dict]] = {}

    for seg_json_path in _segment_json_paths(doc_dir):
        payload = _safe_read_json(seg_json_path)
        if not isinstance(payload, dict):
            continue
        sentences = payload.get("sentences", [])
        if not isinstance(sentences, list):
            continue
        seg_label = f"segment_{_segment_num_from_json(seg_json_path)}"
        for idx, sent in enumerate(sentences, start=1):
            if not isinstance(sent, dict):
                continue
            cluster_id = _safe_cluster_id(sent.get("embedding_cluster_id"))
            if cluster_id is None:
                continue
            sentence_text = str(sent.get("text", "")).strip()
            if not sentence_text:
                continue
            cls = _sentence_classification(sent)
            clusters.setdefault(cluster_id, []).append(
                {
                    "text": sentence_text,
                    "segment": seg_label,
                    "sentence_index": idx,
                    "sentence_type": cls.get("sentence_type", "other"),
                    "agent_types": cls.get("subject_agent_types", []),
                }
            )
    return clusters


def _wordcloud_modules():
    try:
        from wordcloud import STOPWORDS, WordCloud
    except Exception:
        return None, None
    return WordCloud, STOPWORDS


def _cluster_term_frequencies(texts: list[str], stopwords: set[str]) -> Counter:
    counts = Counter()
    for text in texts:
        for token in re.findall(r"[A-Za-z][A-Za-z'-]{1,}", text.lower()):
            if token in stopwords:
                continue
            counts[token] += 1
    return counts


def _render_cluster_wordcloud(
    frequencies: Counter,
    *,
    width: int,
    height: int,
    max_words: int,
):
    WordCloud, _ = _wordcloud_modules()
    if WordCloud is None:
        return None
    if not frequencies:
        return None
    wc = WordCloud(
        width=width,
        height=height,
        max_words=max_words,
        background_color="white",
        collocations=False,
    ).generate_from_frequencies(frequencies)
    return wc.to_array()


def _render_cluster_wordcloud_view(parse_root: Path, doc_id: str):
    st.subheader("Cluster Word Clouds")
    cluster_map = _document_cluster_sentences(str(parse_root), doc_id)
    if not cluster_map:
        st.warning(
            "No embedding clusters found in this document. "
            "Run the spaCy parse pipeline with follow-up embedding clustering enabled."
        )
        return

    WordCloud, default_stopwords = _wordcloud_modules()
    if WordCloud is None or default_stopwords is None:
        st.error(
            "Word cloud rendering requires the `wordcloud` package. "
            "Install it, then rerun this app."
        )
        return

    with st.sidebar:
        st.subheader("Cluster Cloud Settings")
        include_noise = st.checkbox("Include DBSCAN noise (-1)", value=False)
        max_words = st.slider("Max words", min_value=20, max_value=300, value=120, step=10)
        min_word_freq = st.slider("Min token frequency", min_value=1, max_value=10, value=2, step=1)
        cloud_width = st.slider("Cloud width", min_value=500, max_value=1600, value=900, step=100)
        cloud_height = st.slider("Cloud height", min_value=250, max_value=900, value=420, step=50)

    extra_stopwords = {
        "shall",
        "may",
        "must",
        "will",
        "employee",
        "employees",
        "employer",
        "union",
        "agreement",
        "section",
        "article",
        "party",
        "parties",
    }
    stopwords = {str(x).lower() for x in default_stopwords}.union(extra_stopwords)

    ordered_cluster_ids = sorted(cluster_map.keys())
    if not include_noise:
        ordered_cluster_ids = [cid for cid in ordered_cluster_ids if cid != CLUSTER_NOISE_ID]

    if not ordered_cluster_ids:
        st.info("No non-noise clusters available with current filters.")
        return

    total_clustered = sum(len(v) for v in cluster_map.values())
    shown_clustered = sum(len(cluster_map[cid]) for cid in ordered_cluster_ids)
    s1, s2, s3 = st.columns(3)
    s1.metric("Document", doc_id)
    s2.metric("Clusters shown", len(ordered_cluster_ids))
    s3.metric("Sentences shown", shown_clustered)
    st.caption(f"Total clustered sentences (including noise): {total_clustered}")

    for cluster_id in ordered_cluster_ids:
        entries = cluster_map.get(cluster_id, [])
        texts = [str(e.get("text", "")).strip() for e in entries if str(e.get("text", "")).strip()]
        frequencies = _cluster_term_frequencies(texts, stopwords)
        filtered_freq = Counter({k: v for k, v in frequencies.items() if v >= min_word_freq})

        st.markdown(f"### Cluster {cluster_id}")
        st.caption(f"{len(entries)} sentence(s)")
        image_array = _render_cluster_wordcloud(
            filtered_freq,
            width=cloud_width,
            height=cloud_height,
            max_words=max_words,
        )
        if image_array is not None:
            st.image(image_array, use_container_width=True)
        else:
            st.info(
                "Not enough repeated tokens to render a cloud with current settings. "
                "Lower `Min token frequency` or increase sentence coverage."
            )

        top_terms = filtered_freq.most_common(12)
        if top_terms:
            st.dataframe(
                [{"term": term, "count": count} for term, count in top_terms],
                use_container_width=True,
                hide_index=True,
            )

        with st.expander(f"Sample sentences: cluster {cluster_id}", expanded=False):
            preview_count = min(20, len(entries))
            for row in entries[:preview_count]:
                sentence_type = str(row.get("sentence_type", "other"))
                segment = str(row.get("segment", "segment"))
                sent_idx = row.get("sentence_index", "?")
                st.write(f"- [{segment} #{sent_idx}] ({sentence_type}) {row.get('text', '')}")
            if len(entries) > preview_count:
                st.caption(f"... and {len(entries) - preview_count} more sentence(s)")

@st.cache_data(show_spinner=False)
def _table2_corpus_summary(parse_root_str: str) -> dict:
    parse_root = Path(parse_root_str)
    pair_counts = Counter()
    clause_totals = Counter()
    agent_totals = Counter()

    documents = _doc_ids(parse_root)
    segment_count = 0
    sentence_count = 0
    sentence_with_agent_count = 0
    sentence_without_agent_count = 0
    multi_agent_sentence_count = 0

    for doc_id in documents:
        doc_dir = parse_root / doc_id
        for seg_json_path in _segment_json_paths(doc_dir):
            payload = _safe_read_json(seg_json_path)
            if not isinstance(payload, dict):
                continue
            sentences = payload.get("sentences", [])
            if not isinstance(sentences, list):
                continue
            segment_count += 1
            for sent in sentences:
                sentence_count += 1
                cls = _sentence_classification(sent)
                clause = cls.get("sentence_type", "other")
                if clause not in TABLE2_CLAUSE_ORDER:
                    clause = "other"
                clause_totals[clause] += 1

                raw_agents = cls.get("subject_agent_types", [])
                if not isinstance(raw_agents, list):
                    raw_agents = []
                agents = []
                seen = set()
                for agent in raw_agents:
                    agent = str(agent).strip().lower()
                    if agent in TABLE2_AGENT_ORDER and agent not in seen:
                        seen.add(agent)
                        agents.append(agent)

                if not agents:
                    sentence_without_agent_count += 1
                    continue

                sentence_with_agent_count += 1
                if len(agents) > 1:
                    multi_agent_sentence_count += 1
                for agent in agents:
                    pair_counts[(agent, clause)] += 1
                    agent_totals[agent] += 1

    assignment_total = sum(pair_counts.values())

    def _pct(num: int, denom: int) -> float:
        if denom <= 0:
            return 0.0
        return (num * 100.0) / denom

    table2_rows = []
    for agent in TABLE2_AGENT_ORDER:
        row = {"Agent": agent.title()}
        row_total = 0
        for clause in TABLE2_CLAUSE_ORDER:
            count = int(pair_counts.get((agent, clause), 0))
            row_total += count
            row[clause.title()] = f"{count:,} ({_pct(count, assignment_total):.1f}%)"
        row["Total"] = f"{row_total:,} ({_pct(row_total, assignment_total):.1f}%)"
        table2_rows.append(row)

    total_row = {"Agent": "Total"}
    for clause in TABLE2_CLAUSE_ORDER:
        count = int(sum(pair_counts.get((agent, clause), 0) for agent in TABLE2_AGENT_ORDER))
        total_row[clause.title()] = f"{count:,} ({_pct(count, assignment_total):.1f}%)"
    total_row["Total"] = f"{assignment_total:,} (100.0%)" if assignment_total > 0 else "0 (0.0%)"
    table2_rows.append(total_row)

    counts_rows = []
    for agent in TABLE2_AGENT_ORDER:
        row = {"Agent": agent.title()}
        row_total = 0
        for clause in TABLE2_CLAUSE_ORDER:
            count = int(pair_counts.get((agent, clause), 0))
            row_total += count
            row[clause.title()] = count
        row["Total"] = row_total
        counts_rows.append(row)
    counts_total_row = {"Agent": "Total"}
    for clause in TABLE2_CLAUSE_ORDER:
        counts_total_row[clause.title()] = int(
            sum(pair_counts.get((agent, clause), 0) for agent in TABLE2_AGENT_ORDER)
        )
    counts_total_row["Total"] = assignment_total
    counts_rows.append(counts_total_row)

    return {
        "document_count": len(documents),
        "segment_count": segment_count,
        "sentence_count": sentence_count,
        "sentence_with_agent_count": sentence_with_agent_count,
        "sentence_without_agent_count": sentence_without_agent_count,
        "multi_agent_sentence_count": multi_agent_sentence_count,
        "assignment_total": assignment_total,
        "table2_rows": table2_rows,
        "counts_rows": counts_rows,
        "clause_sentence_totals": {k: int(clause_totals.get(k, 0)) for k in TABLE2_CLAUSE_ORDER},
        "agent_assignment_totals": {k: int(agent_totals.get(k, 0)) for k in TABLE2_AGENT_ORDER},
    }


def main():
    st.set_page_config(page_title="spaCy Parse Review", layout="wide")
    st.title("spaCy Parse Review")
    st.caption("Review sentence dependency parse outputs from experiments/sentence_parse/spacy_parse.")

    with st.sidebar:
        parse_root = Path(
            st.text_input("Parse output dir", value=str(DEFAULT_PARSE_OUTPUT_DIR))
        )

    if not parse_root.exists():
        st.error(f"Parse output directory does not exist: {parse_root}")
        return

    doc_ids = _doc_ids(parse_root)
    if not doc_ids:
        st.warning(f"No document folders found in {parse_root}")
        return

    st.subheader("Corpus Summary (Table 2 Style)")
    with st.spinner("Aggregating clause classifications across processed documents..."):
        corpus_summary = _table2_corpus_summary(str(parse_root))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Documents", corpus_summary["document_count"])
    m2.metric("Segments", corpus_summary["segment_count"])
    m3.metric("Sentences", corpus_summary["sentence_count"])
    m4.metric("Agent-Clause Assignments", corpus_summary["assignment_total"])

    st.caption(
        "Percentages are shares of all agent-clause assignments in this parse output root "
        "(mirroring the paper's Table 2 style)."
    )
    st.dataframe(corpus_summary["table2_rows"], use_container_width=True, hide_index=True)

    with st.expander("Coverage details", expanded=False):
        d1, d2, d3 = st.columns(3)
        d1.metric("Sentences with >=1 agent", corpus_summary["sentence_with_agent_count"])
        d2.metric("Sentences with no agent", corpus_summary["sentence_without_agent_count"])
        d3.metric("Sentences with multiple agents", corpus_summary["multi_agent_sentence_count"])
        st.caption("Raw assignment counts matrix")
        st.dataframe(corpus_summary["counts_rows"], use_container_width=True, hide_index=True)

    with st.expander("Random Sentence Sampler (Agent x Sentence Type)", expanded=False):
        _render_random_bucket_sentence_view(parse_root)

    with st.sidebar:
        doc_id = st.selectbox("Document", options=doc_ids)
        view_mode = st.radio("View", options=list(CLUSTER_VIEW_TABS))

    doc_dir = parse_root / doc_id
    segment_paths = _segment_json_paths(doc_dir)
    if not segment_paths:
        st.warning(f"No segment JSON files found for {doc_id} in {doc_dir}")
        return

    if view_mode == "Cluster Word Clouds":
        _render_cluster_wordcloud_view(parse_root, doc_id)
        return

    segment_labels = [f"segment_{_segment_num_from_json(p)}" for p in segment_paths]
    with st.sidebar:
        seg_label = st.selectbox("Segment", options=segment_labels)
    seg_idx = segment_labels.index(seg_label)
    seg_json_path = segment_paths[seg_idx]
    seg_html_path = seg_json_path.with_name(f"{seg_json_path.stem}_dep.html")

    payload = _safe_read_json(seg_json_path) or {}
    sentences = payload.get("sentences", []) if isinstance(payload.get("sentences"), list) else []
    sentence_count = len(sentences)

    sentence_type_counts = Counter()
    agent_type_counts = Counter()
    all_sentence_types = set()
    all_agent_types = set()
    for sent in sentences:
        cls = _sentence_classification(sent)
        sentence_type = cls["sentence_type"]
        sentence_type_counts[sentence_type] += 1
        all_sentence_types.add(sentence_type)
        for agent in cls["subject_agent_types"]:
            agent_type_counts[agent] += 1
            all_agent_types.add(agent)

    with st.sidebar:
        st.subheader("Categorization Filters")
        type_filter = st.selectbox("Sentence type", options=["all"] + sorted(all_sentence_types))
        agent_filter = st.selectbox("Agent type", options=["all"] + sorted(all_agent_types))

    c1, c2, c3 = st.columns(3)
    c1.metric("Document", doc_id)
    c2.metric("Segment", seg_label)
    c3.metric("Sentences", sentence_count)

    if not sentences:
        st.info("No parsed sentences found in this segment JSON.")
        return

    st.subheader("Segment Categorization Summary")
    s1, s2 = st.columns(2)
    with s1:
        st.caption("Sentence type counts")
        st.dataframe(
            [{"sentence_type": k, "count": v} for k, v in sorted(sentence_type_counts.items())],
            use_container_width=True,
            hide_index=True,
        )
    with s2:
        st.caption("Agent type counts")
        st.dataframe(
            [{"agent_type": k, "count": v} for k, v in sorted(agent_type_counts.items())],
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("All Parsed Sentences")
    segment_rows = _segment_sentence_rows(sentences, seg_label)
    st.caption(f"Selected segment: {len(segment_rows)} parsed sentence(s)")
    st.dataframe(segment_rows, use_container_width=True, hide_index=True, height=360)

    with st.expander("All parsed sentences in selected document", expanded=False):
        doc_rows = _document_sentence_rows(str(parse_root), doc_id)
        st.caption(f"{doc_id}: {len(doc_rows)} parsed sentence(s) across all segments")
        st.dataframe(doc_rows, use_container_width=True, hide_index=True, height=420)

    matching_indices = _matching_sentence_indices(sentences, type_filter, agent_filter)
    if matching_indices:
        st.caption(f"{len(matching_indices)} sentence(s) match current filters.")
    else:
        st.warning("No sentences match the current filters.")
        return

    sent_num = st.selectbox("Sentence index", options=matching_indices)
    sent = sentences[int(sent_num) - 1]
    classification = _sentence_classification(sent)

    st.subheader("Dependency Render")
    sentence_html_path = _sentence_html_path(doc_dir, seg_json_path, sent, int(sent_num))
    sentence_html = _safe_read_text(sentence_html_path)
    if sentence_html:
        components.html(sentence_html, height=360, scrolling=True)
    else:
        html_doc = _safe_read_text(seg_html_path)
        if html_doc:
            single_sentence_html = _extract_sentence_render(html_doc, int(sent_num))
            if single_sentence_html:
                components.html(single_sentence_html, height=360, scrolling=True)
            else:
                st.info(
                    "Could not isolate a single-sentence render from the HTML output. "
                    "Showing sentence text only below."
                )
        else:
            st.info(f"Missing HTML render: {sentence_html_path.name}")

    st.subheader("Sentence Details")
    st.code(str(sent.get("text", "")))

    c4, c5 = st.columns([2, 3])
    c4.metric("Sentence Type", classification["sentence_type"])
    c5.write(
        "**Agent Types:** "
        + (", ".join(classification["subject_agent_types"]) if classification["subject_agent_types"] else "None")
    )

    d1, d2 = st.columns(2)
    with d1:
        st.caption("Subject phrases")
        if classification["subject_phrases"]:
            st.write("\n".join([f"- {x}" for x in classification["subject_phrases"]]))
        else:
            st.write("None")
    with d2:
        st.caption("Classification evidence")
        if classification["classification_evidence"]:
            st.write("\n".join([f"- {x}" for x in classification["classification_evidence"]]))
        else:
            st.write("None")

    tokens = sent.get("tokens", [])
    if not isinstance(tokens, list) or not tokens:
        st.info("No token-level dependency records found for this sentence.")
        return

    table_rows = []
    for tok in tokens:
        if not isinstance(tok, dict):
            continue
        table_rows.append(
            {
                "i": tok.get("i"),
                "text": tok.get("text"),
                "lemma": tok.get("lemma"),
                "pos": tok.get("pos"),
                "tag": tok.get("tag"),
                "dep": tok.get("dep"),
                "head_i": tok.get("head_i"),
                "head_text": tok.get("head_text"),
            }
        )
    st.dataframe(table_rows, use_container_width=True, hide_index=True)

    with st.expander("Raw Segment Payload", expanded=False):
        st.json(payload)


if __name__ == "__main__":
    main()
