# sentence_parse

## Goal
Parse sentences in each segment produced by `pipeline/02_segment/runner.py` and convert them into dependency parse structures.

Reference provided:
- https://medium.com/@alshargi.usa/dependency-parsing-and-visualization-with-spacy-b419b9eda169

## KPI / Success Metric
No quantitative KPI provided. Success is visual reviewability of parse output, similar to spaCy render output.

## Methods
### spacy_parse
Use spaCy to extract dependency trees from sentences.

