import { createReadStream, existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import type { Connect, Plugin } from "vite";
import { defineConfig } from "vite";

const __dirname = dirname(fileURLToPath(import.meta.url));
const CACHE_ROOT = resolve(__dirname, "..", "cache");
const STG01 = join(CACHE_ROOT, "stg_01_ocr");
const STG02 = join(CACHE_ROOT, "stg_02_extract");
const KNOWN_SOURCES = ["cornell_dol", "cornell_retail_educ", "dol_archive"];

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
