# Summary Segmentation Method

## Approach
Adapts the PIC (Pseudo-Instruction for Chunking) method from:
> "Document Segmentation Matters for Retrieval-Augmented Generation"
> ACL 2025 Findings — https://aclanthology.org/2025.findings-acl.422/

## How It Works
1. **Summarize:** Generate a summary of the full document (or page) using an LLM via OpenRouter.
2. **Embed:** Compute sentence embeddings for both the summary and each sentence in the OCR text.
3. **Segment:** Compute semantic similarity between each sentence and the summary. Group consecutive sentences with similar topic alignment into segments (clause candidates).
4. **Classify:** For each identified segment, classify it against the 44-clause taxonomy using an LLM.

## Key Insight
By using the document summary as a "pseudo-instruction," the segmentation aligns chunks with the document's key themes without requiring costly per-sentence LLM calls for boundary detection. The summary acts as a semantic anchor that guides where clause boundaries should fall.

## Dependencies
- `sentence-transformers` — for computing sentence embeddings
- `openai` or `httpx` — for LLM calls via OpenRouter
- `numpy` — for similarity computation
