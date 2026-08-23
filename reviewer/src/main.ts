import "./style.css";

interface Extraction {
  extraction_class: string;
  extraction_text: string;
  span_start: number;
  span_end: number;
  grounding_status: string;
  span_reliable?: boolean;
  attributes?: { context?: string | null };
}

interface ExtractionComparisonPayload {
  sampleId: string;
  source: string;
  documentId: string;
  provision: string;
  ocrModel: string;
  text: string;
  pdfUrl: string | null;
  left: { model: string; extractions: Extraction[] };
  right: { model: string; extractions: Extraction[] };
  progress: { reviewed: number; candidateTotal?: number; exhausted: boolean };
}

type Side = "left" | "right";

/** One navigable entry: an extraction plus which model produced it. */
interface SideExtraction {
  side: Side;
  extraction: Extraction;
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

// Highlight color encodes *which model* found the span — the review question
// here is where the two models agree and where only one of them fired.
const SIDE_COLORS: Record<"left" | "both" | "right", string> = {
  left: "#1971c2",
  both: "#7048e8",
  right: "#f08c00",
};
const SIDE_LABELS: Record<"left" | "both" | "right", string> = {
  left: "Extraction A only",
  both: "Both models",
  right: "Extraction B only",
};
const ALIASES: Record<Side, string> = { left: "Extraction A", right: "Extraction B" };

// Grounding status survives as a per-extraction badge in the detail card: it
// says how tightly a model's text is anchored to the source OCR.
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
const provisionSelect = $<HTMLSelectElement>("provision-select");
const extractionMeta = $("extraction-meta");
const extractionProgress = $("extraction-progress");
const extractionMessage = $("extraction-message");
const newExtractionBtn = $<HTMLButtonElement>("new-extraction");
const extractionActionButtons = [
  ...document.querySelectorAll<HTMLButtonElement>("#extraction-actions button[data-choice]"),
];
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

let current: ExtractionComparisonPayload | null = null;
/** Both models' extractions merged into one document-order navigation list. */
let items: SideExtraction[] = [];
let active = -1;
let extractionLoading = false;
let activeTab: "extraction" | "comparison" = "extraction";
let comparison: OcrComparisonPayload | null = null;
let comparisonLoading = false;


function escapeHtml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function overlaps(a: Extraction, b: Extraction): boolean {
  return a.span_start < b.span_end && b.span_start < a.span_end;
}

/** Merge both models' extractions into one document-order list. Index into this
 *  list is the identity used by the highlights, the counter, and the detail card. */
function mergeItems(payload: ExtractionComparisonPayload): SideExtraction[] {
  const merged: SideExtraction[] = [
    ...payload.left.extractions.map((extraction) => ({ side: "left" as Side, extraction })),
    ...payload.right.extractions.map((extraction) => ({ side: "right" as Side, extraction })),
  ];
  return merged.sort(
    (a, b) =>
      a.extraction.span_start - b.extraction.span_start ||
      a.extraction.span_end - b.extraction.span_end ||
      a.side.localeCompare(b.side),
  );
}

/** Interval-flatten both models' spans over the one shared full.txt and render
 *  it with color-coded highlights. Handles overlaps by segmenting on every
 *  boundary, so a stretch both models claimed gets its own "both" segment. */
function renderOverlay(text: string, entries: SideExtraction[]): void {
  const spans = entries
    .map((item, i) => ({
      i,
      side: item.side,
      start: item.extraction.span_start,
      end: item.extraction.span_end,
    }))
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
    const sides = new Set(covering.map((s) => s.side));
    const bucket = sides.size > 1 ? "both" : covering[0].side;
    // Primary = the most recently started (innermost) extraction.
    const primary = covering.reduce((p, c) => (c.start >= p.start ? c : p));
    const ids = covering.map((c) => c.i).join(",");
    parts.push(
      `<mark class="hl" data-ext="${ids}" data-primary="${primary.i}" ` +
        `style="--hl:${SIDE_COLORS[bucket]}">${chunk}</mark>`,
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

function buildLegend(payload: ExtractionComparisonPayload): void {
  const counts = { left: 0, both: 0, right: 0 };
  for (const e of payload.left.extractions) {
    if (payload.right.extractions.some((other) => overlaps(e, other))) counts.both++;
    else counts.left++;
  }
  for (const e of payload.right.extractions) {
    if (!payload.left.extractions.some((other) => overlaps(e, other))) counts.right++;
  }
  legend.innerHTML = (["left", "both", "right"] as const)
    .map(
      (bucket) =>
        `<span class="legend-item"><span class="swatch" style="background:${SIDE_COLORS[bucket]}"></span>` +
        `${SIDE_LABELS[bucket]} (${counts[bucket]})</span>`,
    )
    .join("");
}

function renderExtractionColumn(side: Side, entry: Extraction | null, isActive: boolean): string {
  const alias = ALIASES[side];
  if (!entry) {
    return `
      <div class="detail-col">
        <div class="detail-label">${alias}</div>
        <div class="detail-empty">No overlapping extraction from ${alias}.</div>
      </div>`;
  }
  const context = entry.attributes?.context;
  // A span the pipeline could not re-anchor may highlight only a prefix of the
  // extracted text, so the reviewer should not read the highlight as ground truth.
  const unreliable =
    entry.span_reliable === false ? `<span class="badge warn">Span unreliable</span>` : "";
  return `
    <div class="detail-col${isActive ? " active" : ""}">
      <div class="detail-head">
        <span class="badge alias" style="background:${SIDE_COLORS[side]}">${alias}</span>
        <span class="badge status" style="background:${statusColor(entry.grounding_status)}">
          ${STATUS_LABELS[entry.grounding_status] ?? entry.grounding_status}</span>
        ${unreliable}
        <span class="offsets">chars ${entry.span_start}–${entry.span_end}</span>
      </div>
      <div class="detail-label">Extracted text</div>
      <pre class="detail-text">${escapeHtml(entry.extraction_text)}</pre>
      <div class="detail-label">Context</div>
      <div class="detail-context${context ? "" : " empty"}">${
        context ? escapeHtml(context) : "No context recorded."
      }</div>
    </div>`;
}

/** Show the selected extraction beside whatever the other model found at the
 *  same place — the direct A-vs-B question this tab exists to answer. */
function renderDetail(): void {
  if (!current || active < 0 || active >= items.length) {
    detail.innerHTML = `<div class="detail-empty">Select a highlight, or use ◀ ▶ to step through
      ${items.length} extraction(s).</div>`;
    return;
  }
  const { side, extraction } = items[active];
  const other: Side = side === "left" ? "right" : "left";
  const counterpart =
    current[other].extractions.find((candidate) => overlaps(extraction, candidate)) ?? null;
  const columns: Record<Side, string> = {
    left: renderExtractionColumn(
      "left",
      side === "left" ? extraction : counterpart,
      side === "left",
    ),
    right: renderExtractionColumn(
      "right",
      side === "right" ? extraction : counterpart,
      side === "right",
    ),
  };
  detail.innerHTML = `<div class="detail-cols">${columns.left}${columns.right}</div>`;
}

function select(index: number): void {
  if (!current) return;
  active = index;
  counter.textContent = `${index + 1} / ${items.length}`;
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

function setExtractionActionsDisabled(disabled: boolean): void {
  extractionActionButtons.forEach((button) => {
    button.disabled = disabled;
  });
}

async function loadExtractionComparison(): Promise<void> {
  if (extractionLoading) return;
  extractionLoading = true;
  current = null;
  items = [];
  active = -1;
  setExtractionActionsDisabled(true);
  newExtractionBtn.disabled = true;
  provisionSelect.disabled = true;
  extractionMessage.textContent = "Loading a random document…";
  textView.innerHTML = "";
  legend.innerHTML = "";
  counter.textContent = "0 / 0";
  try {
    const provision = provisionSelect.value;
    const res = await fetch(`/api/extraction-comparison?provision=${encodeURIComponent(provision)}`);
    const payload = await res.json();
    if (!res.ok) throw new Error(payload.error ?? `Request failed (${res.status})`);
    current = payload as ExtractionComparisonPayload;
    items = mergeItems(current);

    if (current.pdfUrl) {
      pdfFrame.src = current.pdfUrl;
      pdfFrame.style.display = "block";
      pdfEmpty.style.display = "none";
    } else {
      pdfFrame.removeAttribute("src");
      pdfFrame.style.display = "none";
      pdfEmpty.style.display = "flex";
    }

    extractionMeta.textContent = `${current.source} · ${current.documentId} · ${current.provision}`;
    const progress = current.progress;
    extractionProgress.textContent = progress.exhausted
      ? "No unreviewed pairs · sampling reviewed pairs again"
      : `${progress.reviewed} reviewed overall` +
        (progress.candidateTotal === undefined ? "" : ` · ${progress.candidateTotal} possible pairs`);

    buildLegend(current);
    renderOverlay(current.text, items);
    renderDetail();
    if (items.length > 0) select(0);
    extractionMessage.textContent =
      `${current.left.extractions.length} extraction(s) from Extraction A, ` +
      `${current.right.extractions.length} from Extraction B. ` +
      "Step through them, then choose one result.";
    setExtractionActionsDisabled(false);
  } catch (err) {
    extractionMeta.textContent = "Extraction comparison unavailable";
    extractionProgress.textContent = "";
    extractionMessage.textContent = `Could not load an extraction comparison. ${String(err)}`;
    renderDetail();
  } finally {
    extractionLoading = false;
    newExtractionBtn.disabled = false;
    provisionSelect.disabled = false;
  }
}

async function saveExtractionComparison(choice: ComparisonChoice): Promise<void> {
  if (!current || extractionLoading) return;
  const reviewed = current;
  extractionLoading = true;
  setExtractionActionsDisabled(true);
  newExtractionBtn.disabled = true;
  provisionSelect.disabled = true;
  extractionMessage.textContent = "Saving review…";
  try {
    const res = await fetch("/api/extraction-comparison/review", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        sampleId: reviewed.sampleId,
        source: reviewed.source,
        documentId: reviewed.documentId,
        provision: reviewed.provision,
        leftModel: reviewed.left.model,
        rightModel: reviewed.right.model,
        choice,
      }),
    });
    const payload = await res.json();
    if (!res.ok) throw new Error(payload.error ?? `Save failed (${res.status})`);
    extractionMessage.textContent = "Review saved. Loading another random pair…";
    extractionLoading = false;
    await loadExtractionComparison();
  } catch (err) {
    extractionMessage.textContent = String(err);
    extractionLoading = false;
    setExtractionActionsDisabled(false);
    newExtractionBtn.disabled = false;
    provisionSelect.disabled = false;
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
  if (showExtraction && !current) void loadExtractionComparison();
  if (!showExtraction && !comparison) void loadComparison();
}

function step(delta: number): void {
  if (items.length === 0) return;
  const n = items.length;
  const next = active < 0 ? 0 : (active + delta + n) % n;
  select(next);
}

const CHOICE_BY_KEY: Partial<Record<string, ComparisonChoice>> = {
  "1": "left_better",
  "2": "right_better",
  "3": "both_good",
  "4": "both_bad",
};

async function init(): Promise<void> {
  const res = await fetch("/api/extraction-provisions");
  const data = await res.json();
  const provisions: string[] = data.provisions ?? [];
  provisionSelect.innerHTML = provisions
    .map((p) => `<option value="${escapeHtml(p)}">${escapeHtml(p)}</option>`)
    .join("");
  provisionSelect.disabled = provisions.length <= 1;

  prevBtn.addEventListener("click", () => step(-1));
  nextBtn.addEventListener("click", () => step(1));
  document.addEventListener("keydown", (e) => {
    if (e.target instanceof HTMLSelectElement || e.target instanceof HTMLButtonElement) return;
    if (activeTab === "extraction" && e.key === "ArrowLeft") return step(-1);
    if (activeTab === "extraction" && e.key === "ArrowRight") return step(1);
    const choice = CHOICE_BY_KEY[e.key];
    if (!choice) return;
    if (activeTab === "extraction") void saveExtractionComparison(choice);
    else void saveComparison(choice);
  });
  extractionTab.addEventListener("click", () => showTab("extraction"));
  comparisonTab.addEventListener("click", () => showTab("comparison"));
  newExtractionBtn.addEventListener("click", () => void loadExtractionComparison());
  provisionSelect.addEventListener("change", () => void loadExtractionComparison());
  extractionActionButtons.forEach((button) => {
    button.addEventListener(
      "click",
      () => void saveExtractionComparison(button.dataset.choice as ComparisonChoice),
    );
  });
  newComparisonBtn.addEventListener("click", () => void loadComparison());
  distanceFilter?.addEventListener("change", () => void loadComparison());
  comparisonActionButtons.forEach((button) => {
    button.addEventListener("click", () => void saveComparison(button.dataset.choice as ComparisonChoice));
  });
  setupDivider("divider", "workspace", "pdf-pane");
  setupDivider("comparison-divider", "comparison-workspace", "comparison-pdf-pane");

  if (provisions.length === 0) {
    extractionMeta.textContent = "No extractions found";
    extractionMessage.textContent = "No *.jsonl extractions found under cache/stg_02_extract.";
    setExtractionActionsDisabled(true);
    return;
  }
  void loadExtractionComparison();
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
