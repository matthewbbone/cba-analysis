# spacy_parse method

## Method summary
Use spaCy dependency parsing to convert each segmentation output segment into sentence-level dependency structures and visual dependency renders.

## User-provided references
- https://spacy.io/api/dependencyparser
- https://medium.com/@alshargi.usa/dependency-parsing-and-visualization-with-spacy-b419b9eda169

## Expected inputs
- Segment text files from `pipeline/02_segment/runner.py` output:
  - `<segmentation_root>/<document_id>/segments/segment_*.txt`

## Expected outputs
- JSON parse payloads per segment and sentence
- HTML dependency visualizations for manual review

