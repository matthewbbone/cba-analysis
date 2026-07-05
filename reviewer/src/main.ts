import "./style.css";

interface Extraction {
  extraction_class: string;
  extraction_text: string;
  span_start: number;
  span_end: number;
  grounding_status: string;
  attributes?: { dimensions?: string[] };
  document_id?: string;
}

interface DocInfo {
  source: string;
  model: string;
  documentId: string;
  extractionFile: string;
  extractionCount: number;
  hasPdf: boolean;
}

interface DocumentPayload {
  source: string;
  model: string;
  documentId: string;
  text: string;
  extractions: Extraction[];
  pdfUrl: string | null;
}

// Color encodes grounding status — the key review signal: how tightly the
// model's extraction is anchored to the source OCR text.
const STATUS_COLORS: Record<string, string> = {
  match_exact: "#2f9e44",
  match_lesser: "#f08c00",
  match_greater: "#1971c2",
  match_fuzzy: "#9c36b5",
  none: "#e03131",
  no_match: "#e03131",
  unknown: "#868e96",
};
const STATUS_LABELS: Record<string, string> = {
  match_exact: "Exact match",
  match_lesser: "Partial (span < text)",
  match_greater: "Span > text",
  match_fuzzy: "Fuzzy match",
  none: "No grounding",
  no_match: "No grounding",
  unknown: "Unknown",
};

function statusColor(status: string): string {
  return STATUS_COLORS[status] ?? STATUS_COLORS.unknown;
}

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const docSelect = $<HTMLSelectElement>("doc-select");
const prevBtn = $<HTMLButtonElement>("prev");
const nextBtn = $<HTMLButtonElement>("next");
const counter = $("counter");
const legend = $("legend");
const pdfFrame = $<HTMLIFrameElement>("pdf-frame");
const pdfEmpty = $("pdf-empty");
const detail = $("detail");
const textView = $("text-view");

let docs: DocInfo[] = [];
let current: DocumentPayload | null = null;
let active = -1;

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

/** Interval-flatten the extraction spans and render the OCR text with
 *  color-coded highlights. Handles overlaps by segmenting on every boundary. */
function renderText(text: string, extractions: Extraction[]): void {
  const spans = extractions
    .map((e, i) => ({ i, start: e.span_start, end: e.span_end, status: e.grounding_status }))
    .filter((s) => Number.isFinite(s.start) && Number.isFinite(s.end) && s.end > s.start);

  const bounds = new Set<number>([0, text.length]);
  for (const s of spans) {
    bounds.add(Math.max(0, Math.min(text.length, s.start)));
    bounds.add(Math.max(0, Math.min(text.length, s.end)));
  }
  const points = [...bounds].sort((a, b) => a - b);

  const parts: string[] = [];
  for (let k = 0; k < points.length - 1; k++) {
    const a = points[k];
    const b = points[k + 1];
    if (b <= a) continue;
    const chunk = escapeHtml(text.slice(a, b));
    const covering = spans.filter((s) => s.start <= a && s.end >= b);
    if (covering.length === 0) {
      parts.push(chunk);
      continue;
    }
    // Primary color = the most recently started (innermost) extraction.
    const primary = covering.reduce((p, c) => (c.start >= p.start ? c : p));
    const ids = covering.map((c) => c.i).join(",");
    parts.push(
      `<mark class="hl" data-ext="${ids}" data-primary="${primary.i}" ` +
        `style="--hl:${statusColor(primary.status)}">${chunk}</mark>`,
    );
  }
  textView.innerHTML = parts.join("");

  textView.querySelectorAll<HTMLElement>("mark.hl").forEach((el) => {
    el.addEventListener("click", () => {
      const primary = Number(el.dataset.primary);
      if (!Number.isNaN(primary)) select(primary);
    });
  });
}

function buildLegend(extractions: Extraction[]): void {
  const counts = new Map<string, number>();
  for (const e of extractions) counts.set(e.grounding_status, (counts.get(e.grounding_status) ?? 0) + 1);
  const items = [...counts.entries()].sort((a, b) => b[1] - a[1]);
  legend.innerHTML = items
    .map(
      ([status, n]) =>
        `<span class="legend-item"><span class="swatch" style="background:${statusColor(status)}"></span>` +
        `${STATUS_LABELS[status] ?? status} (${n})</span>`,
    )
    .join("");
}

function renderDetail(): void {
  if (!current || active < 0 || active >= current.extractions.length) {
    detail.innerHTML = `<div class="detail-empty">Select a highlight, or use ◀ ▶ to step through
      ${current ? current.extractions.length : 0} extraction(s).</div>`;
    return;
  }
  const e = current.extractions[active];
  const sourceSlice = current.text.slice(e.span_start, e.span_end);
  const dims = e.attributes?.dimensions ?? [];
  detail.innerHTML = `
    <div class="detail-head">
      <span class="badge class">${escapeHtml(e.extraction_class)}</span>
      <span class="badge status" style="background:${statusColor(e.grounding_status)}">
        ${STATUS_LABELS[e.grounding_status] ?? e.grounding_status}</span>
      ${dims.map((d) => `<span class="badge dim">${escapeHtml(d)}</span>`).join("")}
      <span class="offsets">chars ${e.span_start}–${e.span_end}</span>
    </div>
    <div class="detail-cols">
      <div class="detail-col">
        <div class="detail-label">Extracted text (model output)</div>
        <pre class="detail-text">${escapeHtml(e.extraction_text)}</pre>
      </div>
      <div class="detail-col">
        <div class="detail-label">Grounded source span (highlighted in OCR)</div>
        <pre class="detail-text src">${escapeHtml(sourceSlice)}</pre>
      </div>
    </div>`;
}

function select(index: number): void {
  if (!current) return;
  active = index;
  counter.textContent = `${index + 1} / ${current.extractions.length}`;
  textView.querySelectorAll<HTMLElement>("mark.hl.active").forEach((el) => el.classList.remove("active"));
  let first: HTMLElement | null = null;
  textView.querySelectorAll<HTMLElement>("mark.hl").forEach((el) => {
    const ids = (el.dataset.ext ?? "").split(",").map(Number);
    if (ids.includes(index)) {
      el.classList.add("active");
      if (!first) first = el;
    }
  });
  if (first) (first as HTMLElement).scrollIntoView({ behavior: "smooth", block: "center" });
  renderDetail();
}

async function loadDocument(doc: DocInfo): Promise<void> {
  const params = new URLSearchParams({
    source: doc.source,
    model: doc.model,
    doc: doc.documentId,
    file: doc.extractionFile,
  });
  const res = await fetch(`/api/document?${params}`);
  current = await res.json();
  if (!current) return;

  if (current.pdfUrl) {
    pdfFrame.src = current.pdfUrl;
    pdfFrame.style.display = "block";
    pdfEmpty.style.display = "none";
  } else {
    pdfFrame.removeAttribute("src");
    pdfFrame.style.display = "none";
    pdfEmpty.style.display = "flex";
  }

  buildLegend(current.extractions);
  renderText(current.text, current.extractions);
  active = -1;
  counter.textContent = `0 / ${current.extractions.length}`;
  renderDetail();
  if (current.extractions.length > 0) select(0);
}

function step(delta: number): void {
  if (!current || current.extractions.length === 0) return;
  const n = current.extractions.length;
  const next = active < 0 ? 0 : (active + delta + n) % n;
  select(next);
}

function labelFor(d: DocInfo): string {
  return `${d.source} / ${d.documentId} — ${d.extractionCount} extraction(s)${d.hasPdf ? "" : " (no PDF)"}`;
}

async function init(): Promise<void> {
  const res = await fetch("/api/documents");
  const data = await res.json();
  docs = data.documents ?? [];
  if (docs.length === 0) {
    detail.innerHTML = `<div class="detail-empty">No extractions found under cache/stg_02_extract.</div>`;
    return;
  }
  docSelect.innerHTML = docs
    .map((d, i) => `<option value="${i}">${escapeHtml(labelFor(d))}</option>`)
    .join("");
  docSelect.addEventListener("change", () => loadDocument(docs[Number(docSelect.value)]));
  prevBtn.addEventListener("click", () => step(-1));
  nextBtn.addEventListener("click", () => step(1));
  document.addEventListener("keydown", (e) => {
    if (e.target instanceof HTMLSelectElement) return;
    if (e.key === "ArrowLeft") step(-1);
    if (e.key === "ArrowRight") step(1);
  });
  setupDivider();
  await loadDocument(docs[0]);
}

function setupDivider(): void {
  const divider = $("divider");
  const workspace = $("workspace");
  const pdfPane = $("pdf-pane");
  let dragging = false;
  divider.addEventListener("mousedown", (e) => {
    dragging = true;
    e.preventDefault();
    document.body.style.userSelect = "none";
  });
  window.addEventListener("mousemove", (e) => {
    if (!dragging) return;
    const rect = workspace.getBoundingClientRect();
    const pct = Math.min(85, Math.max(15, ((e.clientX - rect.left) / rect.width) * 100));
    pdfPane.style.flex = `0 0 ${pct}%`;
  });
  window.addEventListener("mouseup", () => {
    dragging = false;
    document.body.style.userSelect = "";
  });
}

init();
