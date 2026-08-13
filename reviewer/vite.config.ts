import {
  appendFileSync,
  createReadStream,
  existsSync,
  mkdirSync,
  readdirSync,
  readFileSync,
  statSync,
} from "node:fs";
import { randomUUID } from "node:crypto";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import type { Connect, Plugin } from "vite";
import { defineConfig } from "vite";
import { modelMatchupWeight, weightedOrder } from "./src/ocrSampling";

const __dirname = dirname(fileURLToPath(import.meta.url));
const CACHE_ROOT = resolve(__dirname, "..", "cache");
const STG01 = join(CACHE_ROOT, "stg_01_ocr");
const STG02 = join(CACHE_ROOT, "stg_02_extract");
const OCR_COMPARISON_FILE = process.env.OCR_COMPARISON_FILE
  ? resolve(process.env.OCR_COMPARISON_FILE)
  : join(CACHE_ROOT, "reviewer", "ocr_comparisons.jsonl");
const KNOWN_SOURCES = ["cornell_dol", "cornell_retail_educ", "dol_archive"];
const OCR_COMPARISON_MODELS = new Set([
  "Qwen_Qwen3.6-27B-FP8",
  "AIDC-AI_Ovis2.6-30B-A3B",
  "ATH-MaaS_OvisOCR2",
  "google_gemma-4-31B-it",
]);

// Mirrors pipeline/stg_02_extract/runner.py strip_think_blocks: spans are
// computed against the think-stripped text, so we must serve the same text.
const THINK_BLOCK = /<think\b[^>]*>[\s\S]*?<\/think>/gi;
function stripThinkBlocks(text: string): string {
  return text.replace(THINK_BLOCK, "");
}

function listDirs(path: string): string[] {
  if (!existsSync(path)) return [];
  return readdirSync(path, { withFileTypes: true })
    .filter((d) => d.isDirectory() && !d.name.startsWith("."))
    .map((d) => d.name)
    .sort();
}

interface DocInfo {
  source: string;
  model: string;
  documentId: string;
  extractionFile: string; // filename of the jsonl (e.g. wage_tables.jsonl)
  extractionCount: number;
  hasPdf: boolean;
}

interface OcrPage {
  source: string;
  documentId: string;
  pageNumber: number;
  models: string[];
}

interface OcrPair extends OcrPage {
  leftModel: string;
  rightModel: string;
  sampleKey: string;
}

interface OcrMatchup {
  modelA: string;
  modelB: string;
  pairs: OcrPair[];
}

interface OcrReviewHistory {
  reviewedPairs: Set<string>;
  comparisonCounts: Map<string, number>;
}

const OCR_CHOICES = new Set(["left_better", "right_better", "both_good", "both_bad"]);
const ocrTextCache = new Map<string, string>();
const distanceEligibilityCache = new Map<string, boolean>();

/** Resolve cache/<source>/<documentId>.pdf, tolerant of case differences. */
function findPdf(source: string, documentId: string): string | null {
  const sourceDir = join(CACHE_ROOT, source);
  if (!existsSync(sourceDir)) return null;
  const exact = join(sourceDir, `${documentId}.pdf`);
  if (existsSync(exact)) return exact;
  const lower = `${documentId}.pdf`.toLowerCase();
  const match = readdirSync(sourceDir).find((f) => f.toLowerCase() === lower);
  return match ? join(sourceDir, match) : null;
}

function countLines(file: string): number {
  try {
    return readFileSync(file, "utf-8")
      .split("\n")
      .filter((l) => l.trim().length > 0).length;
  } catch {
    return 0;
  }
}

function sampleKey(
  source: string,
  documentId: string,
  pageNumber: number,
  modelA: string,
  modelB: string,
): string {
  const models = [modelA, modelB].sort();
  return `${source}\u0000${documentId}\u0000${pageNumber}\u0000${models[0]}\u0000${models[1]}`;
}

/** Collapse layout-only whitespace differences before measuring OCR content. */
function normalizeOcrForDistance(text: string): string {
  return text.normalize("NFKC").replace(/\s+/g, " ").trim();
}

/** Return whether Levenshtein(a, b) <= maxDistance using a diagonal band.
 *  This avoids the quadratic work of calculating a full distance when all we
 *  need to know is whether a pair clears a normalized cutoff. */
function levenshteinWithin(a: string, b: string, maxDistance: number): boolean {
  if (Math.abs(a.length - b.length) > maxDistance) return false;
  if (a.length === 0 || b.length === 0) return Math.max(a.length, b.length) <= maxDistance;

  // Use the shorter string for columns to minimize the working arrays.
  if (a.length < b.length) [a, b] = [b, a];
  const rows = a.length;
  const columns = b.length;
  const outsideBand = maxDistance + 1;
  let previous = new Uint32Array(columns + 1);
  let current = new Uint32Array(columns + 1);
  const initialEnd = Math.min(columns, maxDistance);
  for (let j = 0; j <= initialEnd; j++) previous[j] = j;
  if (initialEnd < columns) previous[initialEnd + 1] = outsideBand;

  for (let i = 1; i <= rows; i++) {
    const start = Math.max(1, i - maxDistance);
    const end = Math.min(columns, i + maxDistance);
    if (start > end) return false;
    current[start - 1] = start === 1 ? i : outsideBand;
    let rowMinimum = outsideBand;
    for (let j = start; j <= end; j++) {
      const substitution = previous[j - 1] + (a.charCodeAt(i - 1) === b.charCodeAt(j - 1) ? 0 : 1);
      const distance = Math.min(previous[j] + 1, current[j - 1] + 1, substitution);
      current[j] = distance;
      rowMinimum = Math.min(rowMinimum, distance);
    }
    if (end < columns) current[end + 1] = outsideBand;
    if (rowMinimum > maxDistance) return false;
    [previous, current] = [current, previous];
  }
  return previous[columns] <= maxDistance;
}

/** Normalize by the longer string so the cutoff is comparable across pages. */
function meetsNormalizedEditDistance(a: string, b: string, minimum: number): boolean {
  if (minimum <= 0) return true;
  const normalizedA = normalizeOcrForDistance(a);
  const normalizedB = normalizeOcrForDistance(b);
  const longerLength = Math.max(normalizedA.length, normalizedB.length);
  if (longerLength === 0) return false;
  const requiredDistance = Math.ceil(minimum * longerLength);
  return !levenshteinWithin(normalizedA, normalizedB, requiredDistance - 1);
}

/** Find PDF pages from every review source with at least two OCR outputs. */
function discoverOcrPages(): OcrPage[] {
  const pages = new Map<string, OcrPage>();
  for (const source of KNOWN_SOURCES) {
    const comparisonModels = listDirs(join(STG01, source))
      .filter((model) => OCR_COMPARISON_MODELS.has(model));
    for (const model of comparisonModels) {
      const modelDir = join(STG01, source, model);
      for (const documentId of listDirs(modelDir)) {
        if (!findPdf(source, documentId)) continue;
        const documentDir = join(modelDir, documentId);
        for (const filename of readdirSync(documentDir)) {
          const match = /^page_(\d+)\.txt$/.exec(filename);
          if (!match) continue;
          const pageNumber = Number(match[1]);
          const text = readOcrPage(source, model, documentId, pageNumber);
          if (!text) continue;
          const key = `${source}\u0000${documentId}\u0000${pageNumber}`;
          const page = pages.get(key) ?? { source, documentId, pageNumber, models: [] };
          page.models.push(model);
          pages.set(key, page);
        }
      }
    }
  }
  return [...pages.values()]
    .filter((page) => page.models.length >= 2)
    .map((page) => ({ ...page, models: page.models.sort() }));
}

function allOcrPairs(): OcrPair[] {
  const pairs: OcrPair[] = [];
  for (const page of discoverOcrPages()) {
    for (let i = 0; i < page.models.length - 1; i++) {
      for (let j = i + 1; j < page.models.length; j++) {
        const [leftModel, rightModel] = Math.random() < 0.5
          ? [page.models[i], page.models[j]]
          : [page.models[j], page.models[i]];
        pairs.push({
          ...page,
          leftModel,
          rightModel,
          sampleKey: sampleKey(
            page.source,
            page.documentId,
            page.pageNumber,
            leftModel,
            rightModel,
          ),
        });
      }
    }
  }
  return pairs;
}

function ocrReviewHistory(): OcrReviewHistory {
  const reviewedPairs = new Set<string>();
  const comparisonCounts = new Map<string, number>();
  if (!existsSync(OCR_COMPARISON_FILE)) return { reviewedPairs, comparisonCounts };
  for (const line of readFileSync(OCR_COMPARISON_FILE, "utf-8").split("\n")) {
    if (!line.trim()) continue;
    try {
      const row = JSON.parse(line);
      const hasValidJudgment =
        typeof row.left_model === "string" &&
        row.left_model.length > 0 &&
        typeof row.right_model === "string" &&
        row.right_model.length > 0 &&
        row.left_model !== row.right_model &&
        typeof row.choice === "string" &&
        OCR_CHOICES.has(row.choice);
      if (hasValidJudgment) {
        comparisonCounts.set(
          row.left_model,
          (comparisonCounts.get(row.left_model) ?? 0) + 1,
        );
        comparisonCounts.set(
          row.right_model,
          (comparisonCounts.get(row.right_model) ?? 0) + 1,
        );
      }

      // Reconstruct the source-aware key so judgments saved by older versions
      // (whose sample_key omitted the source) remain marked as reviewed.
      if (
        typeof row.source === "string" &&
        typeof row.document_id === "string" &&
        typeof row.page_number === "number" &&
        typeof row.left_model === "string" &&
        typeof row.right_model === "string"
      ) {
        reviewedPairs.add(
          sampleKey(
            row.source,
            row.document_id,
            row.page_number,
            row.left_model,
            row.right_model,
          ),
        );
      } else if (typeof row.sample_key === "string") {
        reviewedPairs.add(row.sample_key);
      }
    } catch {
      // A partial/malformed line should not make the rest of the review file unusable.
    }
  }
  return { reviewedPairs, comparisonCounts };
}

function readOcrPage(source: string, model: string, documentId: string, pageNumber: number): string {
  const key = `${source}\u0000${model}\u0000${documentId}\u0000${pageNumber}`;
  const cached = ocrTextCache.get(key);
  if (cached !== undefined) return cached;
  const path = join(STG01, source, model, documentId, `page_${pageNumber}.txt`);
  const text = stripThinkBlocks(readFileSync(path, "utf-8")).trim();
  ocrTextCache.set(key, text);
  return text;
}

function shuffled<T>(items: T[]): T[] {
  const result = [...items];
  for (let i = result.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [result[i], result[j]] = [result[j], result[i]];
  }
  return result;
}

function pairMeetsDistance(pair: OcrPair, minimum: number): boolean {
  const cacheKey = `${pair.sampleKey}\u0000${minimum}`;
  const cached = distanceEligibilityCache.get(cacheKey);
  if (cached !== undefined) return cached;
  const eligible = meetsNormalizedEditDistance(
    readOcrPage(pair.source, pair.leftModel, pair.documentId, pair.pageNumber),
    readOcrPage(pair.source, pair.rightModel, pair.documentId, pair.pageNumber),
    minimum,
  );
  distanceEligibilityCache.set(cacheKey, eligible);
  return eligible;
}

function matchupKey(modelA: string, modelB: string): string {
  return [modelA, modelB].sort().join("\u0000");
}

/**
 * Pick sources in random order before model matchups, preventing a large
 * collection from dominating merely because it contains more pages. Within a
 * source, under-compared models receive more weight; a page is then sampled
 * uniformly from the selected unordered matchup.
 */
function preferredPairAcrossSources(
  pairs: OcrPair[],
  minimum: number,
  comparisonCounts: ReadonlyMap<string, number>,
): OcrPair | undefined {
  for (const source of shuffled(KNOWN_SOURCES)) {
    const matchups = new Map<string, OcrMatchup>();
    for (const pair of pairs) {
      if (pair.source !== source) continue;
      const [modelA, modelB] = [pair.leftModel, pair.rightModel].sort();
      const key = matchupKey(modelA, modelB);
      const matchup = matchups.get(key) ?? { modelA, modelB, pairs: [] };
      matchup.pairs.push(pair);
      matchups.set(key, matchup);
    }

    const orderedMatchups = weightedOrder(
      [...matchups.values()],
      (matchup) => modelMatchupWeight(
        matchup.modelA,
        matchup.modelB,
        comparisonCounts,
      ),
    );
    for (const matchup of orderedMatchups) {
      const pair = shuffled(matchup.pairs)
        .find((candidate) => pairMeetsDistance(candidate, minimum));
      if (pair) return pair;
    }
  }
  return undefined;
}

function randomOcrComparison(minimumNormalizedEditDistance: number) {
  const pairs = allOcrPairs();
  const history = ocrReviewHistory();
  const unreviewed = pairs.filter((pair) => !history.reviewedPairs.has(pair.sampleKey));
  let pair = preferredPairAcrossSources(
    unreviewed,
    minimumNormalizedEditDistance,
    history.comparisonCounts,
  );
  const exhausted = !pair;
  if (!pair) {
    pair = preferredPairAcrossSources(
      pairs,
      minimumNormalizedEditDistance,
      history.comparisonCounts,
    );
  }
  if (!pair) return null;
  return {
    sampleId: pair.sampleKey,
    source: pair.source,
    documentId: pair.documentId,
    pageNumber: pair.pageNumber,
    pdfUrl: `/api/pdf?source=${encodeURIComponent(pair.source)}&doc=${encodeURIComponent(pair.documentId)}`,
    left: {
      model: pair.leftModel,
      text: readOcrPage(pair.source, pair.leftModel, pair.documentId, pair.pageNumber),
    },
    right: {
      model: pair.rightModel,
      text: readOcrPage(pair.source, pair.rightModel, pair.documentId, pair.pageNumber),
    },
    minimumNormalizedEditDistance,
    progress: {
      reviewed: pairs.filter((candidate) => history.reviewedPairs.has(candidate.sampleKey)).length,
      candidateTotal: pairs.length,
      exhausted,
    },
  };
}

function saveOcrComparison(body: Record<string, unknown>): void {
  const sampleId = body.sampleId;
  const source = body.source;
  const documentId = body.documentId;
  const pageNumber = body.pageNumber;
  const leftModel = body.leftModel;
  const rightModel = body.rightModel;
  const choice = body.choice;
  // Default old/stale clients to the UI's standard cutoff. This matters when
  // Vite reloads the server config before an already-open browser tab reloads.
  const minimumNormalizedEditDistance = body.minimumNormalizedEditDistance ?? 0.05;
  if (
    typeof sampleId !== "string" ||
    typeof source !== "string" ||
    !KNOWN_SOURCES.includes(source) ||
    typeof documentId !== "string" ||
    typeof pageNumber !== "number" ||
    !Number.isInteger(pageNumber) ||
    typeof leftModel !== "string" ||
    typeof rightModel !== "string" ||
    typeof choice !== "string" ||
    !OCR_CHOICES.has(choice) ||
    typeof minimumNormalizedEditDistance !== "number" ||
    !Number.isFinite(minimumNormalizedEditDistance) ||
    minimumNormalizedEditDistance < 0 ||
    minimumNormalizedEditDistance > 1 ||
    sampleId !== sampleKey(source, documentId, pageNumber, leftModel, rightModel)
  ) {
    throw new Error("Invalid OCR comparison review");
  }

  const validPair = allOcrPairs().some((pair) => pair.sampleKey === sampleId);
  if (!validPair) throw new Error("OCR comparison sample no longer exists");

  const preferredModel = choice === "left_better" ? leftModel : choice === "right_better" ? rightModel : null;
  mkdirSync(dirname(OCR_COMPARISON_FILE), { recursive: true });
  appendFileSync(
    OCR_COMPARISON_FILE,
    `${JSON.stringify({
      review_id: randomUUID(),
      reviewed_at: new Date().toISOString(),
      sample_key: sampleId,
      source,
      document_id: documentId,
      page_number: pageNumber,
      left_model: leftModel,
      right_model: rightModel,
      choice,
      preferred_model: preferredModel,
      minimum_normalized_edit_distance: minimumNormalizedEditDistance,
    })}\n`,
    "utf-8",
  );
}

/** Walk stg_02_extract for every document that has a *.jsonl extraction file. */
function discoverDocuments(): DocInfo[] {
  const docs: DocInfo[] = [];
  for (const source of listDirs(STG02)) {
    const sourceDir = join(STG02, source);
    for (const model of listDirs(sourceDir)) {
      const modelDir = join(sourceDir, model);
      for (const documentId of listDirs(modelDir)) {
        const docDir = join(modelDir, documentId);
        const jsonls = readdirSync(docDir).filter((f) => f.endsWith(".jsonl"));
        if (jsonls.length === 0) continue;
        const extractionFile = jsonls[0];
        docs.push({
          source,
          model,
          documentId,
          extractionFile,
          extractionCount: countLines(join(docDir, extractionFile)),
          hasPdf: findPdf(source, documentId) !== null,
        });
      }
    }
  }
  return docs;
}

function readExtractions(source: string, model: string, documentId: string, file: string) {
  const path = join(STG02, source, model, documentId, file);
  if (!existsSync(path)) return [];
  return readFileSync(path, "utf-8")
    .split("\n")
    .filter((l) => l.trim().length > 0)
    .map((l) => JSON.parse(l));
}

/** Resolve the OCR full.txt for a document.
 *
 *  Deliberately ignores the extraction model: stg_02_extract is keyed by the
 *  *extraction* model (which varies), while stg_01_ocr is keyed by the *OCR*
 *  model (a single one for the whole cache). Scanning the OCR model dirs keeps
 *  the two independent, so extracting with a new model doesn't 404 here. */
function readText(source: string, documentId: string): string | null {
  for (const ocrModel of listDirs(join(STG01, source))) {
    const path = join(STG01, source, ocrModel, documentId, "full.txt");
    if (existsSync(path)) return stripThinkBlocks(readFileSync(path, "utf-8"));
  }
  return null;
}

function sendJson(res: Parameters<Connect.NextHandleFunction>[1], data: unknown, code = 200) {
  res.statusCode = code;
  res.setHeader("Content-Type", "application/json");
  res.end(JSON.stringify(data));
}

const apiHandler: Connect.NextHandleFunction = (req, res, next) => {
  const url = new URL(req.url ?? "", "http://localhost");
  const q = url.searchParams;
  const path = url.pathname; // relative to /api

  try {
    if (path === "/documents") {
      return sendJson(res, { documents: discoverDocuments(), sources: KNOWN_SOURCES });
    }

    if (path === "/ocr-comparison" && req.method === "GET") {
      const minimumNormalizedEditDistance = Number(q.get("minDistance") ?? "0.05");
      if (
        !Number.isFinite(minimumNormalizedEditDistance) ||
        minimumNormalizedEditDistance < 0 ||
        minimumNormalizedEditDistance > 1
      ) {
        return sendJson(res, { error: "minDistance must be between 0 and 1" }, 400);
      }
      const comparison = randomOcrComparison(minimumNormalizedEditDistance);
      if (!comparison) {
        return sendJson(res, { error: "No OCR pairs meet the selected minimum text difference" }, 404);
      }
      return sendJson(res, comparison);
    }

    if (path === "/ocr-comparison/review" && req.method === "POST") {
      let raw = "";
      req.setEncoding("utf-8");
      req.on("data", (chunk) => {
        raw += chunk;
        if (raw.length > 65_536) req.destroy(new Error("Request body too large"));
      });
      req.on("end", () => {
        try {
          saveOcrComparison(JSON.parse(raw));
          sendJson(res, { saved: true }, 201);
        } catch (err) {
          sendJson(res, { error: String(err) }, 400);
        }
      });
      return;
    }

    if (path === "/document") {
      const source = q.get("source")!;
      const model = q.get("model")!;
      const doc = q.get("doc")!;
      const file = q.get("file") ?? "wage_tables.jsonl";
      const text = readText(source, doc);
      if (text === null) return sendJson(res, { error: "OCR text not found" }, 404);
      const extractions = readExtractions(source, model, doc, file);
      const pdf = findPdf(source, doc);
      return sendJson(res, {
        source,
        model,
        documentId: doc,
        text,
        extractions,
        pdfUrl: pdf ? `/api/pdf?source=${encodeURIComponent(source)}&doc=${encodeURIComponent(doc)}` : null,
      });
    }

    if (path === "/pdf") {
      const source = q.get("source")!;
      const doc = q.get("doc")!;
      const pdf = findPdf(source, doc);
      if (!pdf) return sendJson(res, { error: "PDF not found" }, 404);
      res.statusCode = 200;
      res.setHeader("Content-Type", "application/pdf");
      res.setHeader("Content-Length", statSync(pdf).size);
      res.setHeader("Content-Disposition", `inline; filename="${doc}.pdf"`);
      return createReadStream(pdf).pipe(res);
    }
  } catch (err) {
    return sendJson(res, { error: String(err) }, 500);
  }
  next();
};

// Register on BOTH the dev server and the preview (built) server so the API
// is available whether you run `npm run dev` or `npm run build && npm run preview`.
function apiPlugin(): Plugin {
  return {
    name: "cba-reviewer-api",
    configureServer(server) {
      server.middlewares.use("/api", apiHandler);
    },
    configurePreviewServer(server) {
      server.middlewares.use("/api", apiHandler);
    },
  };
}

export default defineConfig({
  plugins: [apiPlugin()],
  server: { port: 5178, open: true },
});
