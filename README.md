# CBA Dataset Construction Methodology

## Purpose and Scope
This repository implements a document-transformation pipeline that converts archival collective bargaining agreements (CBAs) into a structured, document-linked dataset for clause-level and document-level generosity analysis.

The canonical methodology is:

`01_ocr -> 02_segment -> 03_classification -> 04_generosity_gab`

The `04_generosity_ash` method is retained as a baseline comparison and is documented in Appendix A.

## End-to-End Transformation Path
The pipeline transforms each source contract through the following representation sequence:

`PDF document -> page text -> document text -> segment text -> clause-typed segment -> clause-type generosity score -> document composite score`

At each stage, `document_id` is preserved so intermediate and final records remain linkable across tables.

## Stage 01: OCR Transformation (`01_ocr`)
**Input:** Scanned CBA PDFs.

**Transformation:** Page images are converted from non-machine-readable document scans into machine-readable text.

**Output:**
- Document-aligned page text units.
- Assembled document-level text.

**Dataset contribution:** Establishes the base textual corpus required for all downstream structuring and scoring.

## Stage 02: Segmentation Transformation (`02_segment`)
**Input:** OCR-derived document text.

**Transformation:** Each CBA is partitioned into ordered top-level contractual units (segments/articles), producing standardized text units for later labeling and scoring.

**Output:**
- Ordered segment text units per document.
- Document-relative structural segmentation.

**Dataset contribution:** Creates comparable segment-level records so semantic labels and generosity scores are attached to consistent contract units.

## Stage 03: Clause Classification Transformation (`03_classification`)
**Input:** Segmented contract text units.

**Transformation:** Each segment is assigned a clause type from a fixed taxonomy, with `OTHER` used when no taxonomy class is appropriate.

**Output:**
- Segment-level clause labels with one primary assigned class per segment.

**Dataset contribution:** Adds semantic clause typing, enabling within-clause comparisons across different documents.

## Stage 04: Gabriel Generosity Transformation (Primary) (`04_generosity_gab`)
**Input:** Segment text with clause-type labels.

**Transformation:**
1. Segments are ranked by generosity within each clause type across documents.
2. Segment ranks are converted to within-clause percentiles.
3. A clause-type score is computed per document as the mean segment percentile for that clause type.
4. A document composite score is computed as the mean of available clause-type scores.

**Coverage rule for stability:**
- If a clause type appears in fewer than 5 documents, no ranking is computed for that clause type.
- The corresponding clause-type score is treated as `NA` and excluded from the document composite mean.

**Output:**
- Segment-level generosity table.
- Clause-type-by-document generosity table.
- Document-level composite ranking table.

**Dataset contribution:** Produces comparable continuous generosity measures at segment, clause-type, and document levels.

## Final Dataset Structure
The final dataset is a linked set of tables keyed by `document_id` (and `segment_number` where applicable).

Core analytical views:

1. **Segment view**
- Segment text unit.
- Clause type.
- Segment-level generosity outputs.

2. **Clause-document view**
- Clause-type score for each document.

3. **Document view**
- Composite generosity score.
- Document rank.

Interpretation:
- Higher Gabriel values indicate language that is more worker-generous relative to peer segments in the same clause type.
- Composite values summarize average clause-type standing for each document using available non-NA clause-type scores.

## Appendix A: ASH Baseline Comparison Method (`04_generosity_ash`)
`04_generosity_ash` is a baseline comparator, not the primary scoring method.

Baseline transformation logic:

1. Parse segment text into sentences and classify statement/agent roles.
2. Derive worker-benefit and firm-benefit counts from classified statements.
3. Compute worker-over-firm ratios at segment and clause-document levels.
4. Compute a document composite from clause-type ratios.

Comparison role:
- ASH provides an alternative, rule-based generosity signal.
- It is used to benchmark and triangulate Gabriel-based rankings.
