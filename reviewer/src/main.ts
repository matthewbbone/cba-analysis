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

type ComparisonChoice = "left_better" | "right_better" | "both_good" | "both_bad";

interface OcrComparisonPayload {
  sampleId: string;
  source: string;
  documentId: string;
  pageNumber: number;
  pdfUrl: string;
  left: { model: string; text: string };
  right: { model: string; text: string };
  minimumNormalizedEditDistance?: number;
  progress: { reviewed: number; candidateTotal?: number; total?: number; exhausted: boolean };
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

/** One entry per (source, document). The same document is often extracted by
 *  several models; those variants hang off `models` so the model dropdown can
 *  switch between them without changing which document is shown. */
interface DocGroup {
  source: string;
  documentId: string;
  hasPdf: boolean;
  models: DocInfo[]; // sorted by model name
}

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const docSelect = $<HTMLSelectElement>("doc-select");
const modelSelect = $<HTMLSelectElement>("model-select");
const prevBtn = $<HTMLButtonElement>("prev");
const nextBtn = $<HTMLButtonElement>("next");
const counter = $("counter");
const legend = $("legend");
const pdfFrame = $<HTMLIFrameElement>("pdf-frame");
const pdfEmpty = $("pdf-empty");
const detail = $("detail");
const textView = $("text-view");
const extractionTab = $<HTMLButtonElement>("extraction-tab");
const comparisonTab = $<HTMLButtonElement>("comparison-tab");
const extractionControls = $("extraction-controls");
const comparisonControls = $("comparison-controls");
const extractionWorkspace = $("workspace");
const comparisonWorkspace = $("comparison-workspace");
const comparisonMeta = $("comparison-meta");
const comparisonProgress = $("comparison-progress");
const comparisonMessage = $("comparison-message");
const comparisonPdfFrame = $<HTMLIFrameElement>("comparison-pdf-frame");
const leftText = $("left-text");
const rightText = $("right-text");
const newComparisonBtn = $<HTMLButtonElement>("new-comparison");
// Optional so an already-open page from before the filter was added can still
// hot-reload this script and fetch comparisons. A hard refresh will add it.
const distanceFilter = document.getElementById("distance-filter") as HTMLSelectElement | null;
const comparisonActionButtons = [
  ...document.querySelectorAll<HTMLButtonElement>("#comparison-actions button[data-choice]"),
];

let groups: DocGroup[] = [];
let current: DocumentPayload | null = null;
let active = -1;
let activeTab: "extraction" | "comparison" = "extraction";
let comparison: OcrComparisonPayload | null = null;
let comparisonLoading = false;

/** Collapse the flat /api/documents list into one entry per document. */
function buildGroups(docs: DocInfo[]): DocGroup[] {
  const byKey = new Map<string, DocGroup>();
  for (const d of docs) {
    const key = `${d.source}\u0000${d.documentId}`;
    let group = byKey.get(key);
    if (!group) {
      group = { source: d.source, documentId: d.documentId, hasPdf: d.hasPdf, models: [] };
      byKey.set(key, group);
    }
    group.hasPdf ||= d.hasPdf;
    group.models.push(d);
  }
  const result = [...byKey.values()];
  for (const g of result) g.models.sort((a, b) => a.model.localeCompare(b.model));
  result.sort((a, b) => a.source.localeCompare(b.source) || a.documentId.localeCompare(b.documentId));
  return result;
}

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

/** `keepIndex` holds the extraction cursor steady when swapping models on the
 *  same document, so you can flip between models at the same position. */
async function loadDocument(doc: DocInfo, keepIndex = -1): Promise<void> {
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
  if (current.extractions.length > 0) {
    select(Math.min(Math.max(keepIndex, 0), current.extractions.length - 1));
  }
}

function setComparisonActionsDisabled(disabled: boolean): void {
  comparisonActionButtons.forEach((button) => {
    button.disabled = disabled;
  });
}

async function loadComparison(): Promise<void> {
  if (comparisonLoading) return;
  comparisonLoading = true;
  comparison = null;
  setComparisonActionsDisabled(true);
  newComparisonBtn.disabled = true;
  if (distanceFilter) distanceFilter.disabled = true;
  comparisonMessage.textContent = "Loading a random page…";
  leftText.textContent = "";
  rightText.textContent = "";
  try {
    const minimumDistance = Number(distanceFilter?.value ?? "0.05");
    const res = await fetch(`/api/ocr-comparison?minDistance=${encodeURIComponent(minimumDistance)}`);
    const payload = await res.json();
    if (!res.ok) throw new Error(payload.error ?? `Request failed (${res.status})`);
    comparison = payload as OcrComparisonPayload;
    comparisonMeta.textContent =
      `${comparison.source} · ${comparison.documentId} · PDF page ${comparison.pageNumber}`;
    const progress = comparison.progress;
    const candidateTotal = progress.candidateTotal ?? progress.total;
    comparisonProgress.textContent = progress.exhausted
      ? `No unreviewed matches · sampling reviewed pairs again`
      : `${progress.reviewed} reviewed overall${candidateTotal === undefined ? "" : ` · ${candidateTotal} possible pairs`}`;
    comparisonPdfFrame.src = `${comparison.pdfUrl}#page=${comparison.pageNumber}&zoom=page-width`;
    leftText.textContent = comparison.left.text;
    rightText.textContent = comparison.right.text;
    comparisonMessage.textContent = "Compare both transcriptions to the PDF page, then choose one result.";
    setComparisonActionsDisabled(false);
  } catch (err) {
    comparisonMeta.textContent = "OCR comparison unavailable";
    comparisonProgress.textContent = "";
    comparisonMessage.textContent = `Could not load an OCR comparison. ${String(err)}`;
  } finally {
    comparisonLoading = false;
    newComparisonBtn.disabled = false;
    if (distanceFilter) distanceFilter.disabled = false;
  }
}

async function saveComparison(choice: ComparisonChoice): Promise<void> {
  if (!comparison || comparisonLoading) return;
  const reviewed = comparison;
  comparisonLoading = true;
  setComparisonActionsDisabled(true);
  newComparisonBtn.disabled = true;
  if (distanceFilter) distanceFilter.disabled = true;
  comparisonMessage.textContent = "Saving review…";
  try {
    const res = await fetch("/api/ocr-comparison/review", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        sampleId: reviewed.sampleId,
        source: reviewed.source,
        documentId: reviewed.documentId,
        pageNumber: reviewed.pageNumber,
        leftModel: reviewed.left.model,
        rightModel: reviewed.right.model,
        choice,
        minimumNormalizedEditDistance:
          reviewed.minimumNormalizedEditDistance ?? Number(distanceFilter?.value ?? "0.05"),
      }),
    });
    const payload = await res.json();
    if (!res.ok) throw new Error(payload.error ?? `Save failed (${res.status})`);
    comparisonMessage.textContent = "Review saved. Loading another random pair…";
    comparisonLoading = false;
    await loadComparison();
  } catch (err) {
    comparisonMessage.textContent = String(err);
    comparisonLoading = false;
    setComparisonActionsDisabled(false);
    newComparisonBtn.disabled = false;
    if (distanceFilter) distanceFilter.disabled = false;
  }
}

function showTab(tab: "extraction" | "comparison"): void {
  activeTab = tab;
  const showExtraction = tab === "extraction";
  extractionTab.classList.toggle("active", showExtraction);
  extractionTab.setAttribute("aria-selected", String(showExtraction));
  comparisonTab.classList.toggle("active", !showExtraction);
  comparisonTab.setAttribute("aria-selected", String(!showExtraction));
  extractionControls.hidden = !showExtraction;
  comparisonControls.hidden = showExtraction;
  extractionWorkspace.hidden = !showExtraction;
  comparisonWorkspace.hidden = showExtraction;
  if (!showExtraction && !comparison) void loadComparison();
}

function step(delta: number): void {
  if (!current || current.extractions.length === 0) return;
  const n = current.extractions.length;
  const next = active < 0 ? 0 : (active + delta + n) % n;
  select(next);
}

function labelForGroup(g: DocGroup): string {
  const models = g.models.length > 1 ? ` — ${g.models.length} models` : "";
  return `${g.source} / ${g.documentId}${models}${g.hasPdf ? "" : " (no PDF)"}`;
}

function labelForModel(d: DocInfo): string {
  return `${d.model} — ${d.extractionCount} extraction(s)`;
}

/** Repopulate the model dropdown for a document and load its first model. */
function selectGroup(group: DocGroup): void {
  modelSelect.innerHTML = group.models
    .map((d, i) => `<option value="${i}">${escapeHtml(labelForModel(d))}</option>`)
    .join("");
  modelSelect.value = "0";
  modelSelect.disabled = group.models.length <= 1;
  void loadDocument(group.models[0]);
}

async function init(): Promise<void> {
  const res = await fetch("/api/documents");
  const data = await res.json();
  groups = buildGroups(data.documents ?? []);
  if (groups.length === 0) {
    detail.innerHTML = `<div class="detail-empty">No extractions found under cache/stg_02_extract.</div>`;
    return;
  }
  docSelect.innerHTML = groups
    .map((g, i) => `<option value="${i}">${escapeHtml(labelForGroup(g))}</option>`)
    .join("");
  docSelect.addEventListener("change", () => selectGroup(groups[Number(docSelect.value)]));
  modelSelect.addEventListener("change", () => {
    const group = groups[Number(docSelect.value)];
    void loadDocument(group.models[Number(modelSelect.value)], active);
  });
  prevBtn.addEventListener("click", () => step(-1));
  nextBtn.addEventListener("click", () => step(1));
  document.addEventListener("keydown", (e) => {
    if (e.target instanceof HTMLSelectElement || e.target instanceof HTMLButtonElement) return;
    if (activeTab === "extraction") {
      if (e.key === "ArrowLeft") step(-1);
      if (e.key === "ArrowRight") step(1);
      return;
    }
    const choiceByKey: Partial<Record<string, ComparisonChoice>> = {
      "1": "left_better",
      "2": "right_better",
      "3": "both_good",
      "4": "both_bad",
    };
    const choice = choiceByKey[e.key];
    if (choice) void saveComparison(choice);
  });
  extractionTab.addEventListener("click", () => showTab("extraction"));
  comparisonTab.addEventListener("click", () => showTab("comparison"));
  newComparisonBtn.addEventListener("click", () => void loadComparison());
  distanceFilter?.addEventListener("change", () => void loadComparison());
  comparisonActionButtons.forEach((button) => {
    button.addEventListener("click", () => void saveComparison(button.dataset.choice as ComparisonChoice));
  });
  setupDivider("divider", "workspace", "pdf-pane");
  setupDivider("comparison-divider", "comparison-workspace", "comparison-pdf-pane");
  selectGroup(groups[0]);
}

function setupDivider(dividerId: string, workspaceId: string, pdfPaneId: string): void {
  const divider = $(dividerId);
  const workspace = $(workspaceId);
  const pdfPane = $(pdfPaneId);
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
