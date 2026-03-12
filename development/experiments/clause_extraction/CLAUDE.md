# Clause Extraction Experiment

## Goal
Identify legal clauses and extract the relevant text spans from OCR output of Collective Bargaining Agreement (CBA) documents. Compare three different extraction approaches.

## Metric
Pairwise comparison of clause-specific span overlap between methods. For each pair of methods, measure the overlap of the text spans they identify for each clause type across the same documents.

## Methods

### 1. summary_segmentation
- **Description:** Adapts the PIC (Pseudo-Instruction for Chunking) method from "Document Segmentation Matters for Retrieval-Augmented Generation" (ACL 2025 Findings). Generates a summary of the document, then computes semantic similarity between sentences and the summary to dynamically group sentences into clause-aligned segments. Segments are then classified against the taxonomy.
- **Paper:** https://aclanthology.org/2025.findings-acl.422/

### 2. langextract
- **Description:** Uses Google's [langextract](https://github.com/google/langextract) library with `lx.extract()`, structured prompts, and few-shot `ExampleData`/`Extraction` examples. Processes OCR text page-by-page with built-in source grounding and span alignment. Adapted from `clause_extraction/extraction/runner.py`.

### 3. llm_segmentation
- **Description:** Direct LLM-based segmentation via OpenRouter. Feeds OCR text to a language model with a prompt asking it to identify clause boundaries, classify each segment against the taxonomy, and return the exact text spans.

## Data
- **Input:** OCR text files from `ocr_output/document_*/page_*.txt` (15 documents, ~1500 pages)
- **Taxonomy:** 44 clause types from `references/feature_taxonomy_final.md`

## Output Format (per method)
Each method returns a list of extractions per document page:
```json
{
  "document_id": "document_1",
  "page": 1,
  "extractions": [
    {
      "clause_label": "Recognition Clause",
      "extraction_text": "recognizes the Union as the exclusive bargaining representative",
      "start_pos": 42,
      "end_pos": 99
    }
  ]
}
```

## Folder Structure
```
clause_extraction/
├── CLAUDE.md                    # This file
├── results.csv                  # Aggregated experiment results
├── run_experiment.py            # Runner script for all methods
├── analyze_results.py           # Analysis and reporting script
├── summary_segmentation/
│   ├── CLAUDE.md                # Method documentation
│   └── method.py                # PIC-based implementation
├── langextract/
│   ├── CLAUDE.md                # Method documentation
│   └── method.py                # LangExtract implementation
└── llm_segmentation/
    ├── CLAUDE.md                # Method documentation
    └── method.py                # Direct LLM implementation
```
