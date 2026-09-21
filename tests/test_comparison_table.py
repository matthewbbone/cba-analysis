import json

import pytest

from pipeline.stg_02_extract import comparison_table as ct
from pipeline.stg_02_extract.span_grounding import GroundedAnswer, ground_answer


def test_evaluate_paired_cases_and_global_winners():
    gold = {(name, 'governing_law'): ct.GoldCase('NY', [(0, 2)] if name != 'negative' else [])
            for name in ['one', 'two', 'negative']}
    perfect = {case: GroundedAnswer(item.spans.copy()) for case, item in gold.items()}
    imperfect = {case: GroundedAnswer([(0, 2)] if case[0] == 'one' else [])
                 for case in gold}
    runs = [ct.Run('a', 'ContractEval', 'A <model>', perfect),
            ct.Run('a', 'Harness', 'A <model>', imperfect),
            ct.Run('b', 'Harness', 'B', dict(imperfect))]  # No paired ContractEval run.
    report = ct.evaluate(runs, gold)
    assert len(report['cases']) == 3 and report['positive_count'] == 2
    assert len(report['rows']) == 1
    assert report['rows'][0]['n_types'] == 1
    assert report['rows'][0]['ContractEval']['precision'] == 1
    assert report['rows'][0]['Harness']['recall'] == 0.5
    assert report['rows'][0]['ContractEval']['f1'] == 1
    assert report['rows'][0]['Harness']['f1'] == pytest.approx(2 / 3)
    assert report['rows'][0]['Harness']['jaccard_mean'] == 1
    html = ct.render_html(report, ct.DEFAULT_CUAD_JSON)
    assert html.count('<strong>1.000</strong>') == 6
    assert 'colspan="6" scope="colgroup">ContractEval' in html
    assert 'colspan="6" scope="colgroup">Harness' in html
    assert html.count('>Corpus retained</th>') == 2
    assert '<td>66.67%</td>' in html and '<td>33.33%</td>' in html
    assert 'A &lt;model&gt;' in html and '<td>—</td>' not in html
    assert '<th rowspan="2" scope="col">N Types</th>' in html
    assert '>Precision</th>' in html and html.count('>F1</th>') == 2
    assert html.count('<summary>Examples</summary>') == 2
    assert html.count('<table') == 1
    assert '<p>' not in html and '<h1>' not in html


@pytest.mark.parametrize('context,spans,characters,retained', [
    ('abcdefghij', [(0, 4), (0, 4), (2, 6), (8, 10)], 8, 0.8),
    ('é 中\n', [(0, 4)], 4, 1.0),
    ('abcdefghij', [], 0, 0.0),
    ('', [], 0, None),
])
def test_corpus_retained_counts_unique_source_characters(context, spans, characters, retained):
    case = ('one', 'governing_law')
    # All extractions are false positives here; they still retain source text.
    gold = {case: ct.GoldCase(context, [])}
    predictions = {case: GroundedAnswer(spans, ['invented text'])}
    report = ct.evaluate([ct.Run('a', method, 'A', predictions) for method in ct.METHODS], gold)
    for method in ct.METHODS:
        metrics = report['rows'][0][method]
        assert metrics['corpus_characters'] == len(context)
        assert metrics['extracted_characters'] == characters
        assert metrics['corpus_retained'] == retained


def test_all_types_are_pooled_per_model_and_jaccard_weights_true_positives():
    first, second, third = 'governing_law', 'joint_ip_ownership', 'warranty_duration'
    gold = {
        ('one', first): ct.GoldCase('abcdefgh', [(0, 4)]),
        ('two', first): ct.GoldCase('abcdefgh', [(0, 4)]),
        ('one', second): ct.GoldCase('abcdefgh', [(0, 2), (2, 4), (4, 6)]),
        ('two', second): ct.GoldCase('abcdefgh', []),
        ('one', third): ct.GoldCase('abcdefgh', [(0, 4)]),
        ('two', third): ct.GoldCase('abcdefgh', []),
    }
    a_cases = {case: GroundedAnswer(item.spans.copy()) for case, item in gold.items()
               if case[1] in (first, second)}
    a_ce = {
        ('one', first): GroundedAnswer([(0, 6)]),  # TP Jaccard 2/3.
        ('two', first): GroundedAnswer(),  # FN does not enter Jaccard's denominator.
        ('one', second): GroundedAnswer([(0, 2), (2, 4), (4, 6), (6, 8)]),  # 3 TP, 1 FP.
        ('two', second): GroundedAnswer(),
    }
    b_cases = {case: GroundedAnswer(item.spans.copy()) for case, item in gold.items() if case[1] == third}
    b_ce = {**b_cases, **{case: a_ce[case] for case in a_ce if case[1] == first}}
    b_harness = {**b_cases, ('one', first): a_cases[('one', first)]}  # Missing doc two disqualifies type.
    report = ct.evaluate([
        ct.Run('a', 'ContractEval', 'A', a_ce), ct.Run('a', 'Harness', 'A', a_cases),
        ct.Run('b', 'ContractEval', 'B', b_ce), ct.Run('b', 'Harness', 'B', b_harness),
    ], gold)
    a, b = report['rows']
    assert (a['n_types'], b['n_types']) == (2, 1)
    assert a['clause_types'] == [first, second] and b['clause_types'] == [third]
    assert (a['case_count'], b['case_count']) == (4, 2)
    assert a['ContractEval']['precision'] == a['ContractEval']['recall'] == 4 / 5
    assert a['ContractEval']['f1'] == pytest.approx(4 / 5)
    assert a['ContractEval']['corpus_characters'] == 16
    assert a['ContractEval']['extracted_characters'] == 8
    assert a['ContractEval']['corpus_retained'] == 0.5
    assert a['Harness']['extracted_characters'] == 10
    assert a['ContractEval']['jaccard_mean'] == pytest.approx((2 / 3 + 3) / 4)
    assert a['ContractEval']['TP'] == 4 and a['ContractEval']['FN'] == a['ContractEval']['FP'] == 1
    assert all(a['ContractEval']['examples'][kind] is not None for kind in ('TP', 'FP', 'TN', 'FN'))
    assert a['ContractEval']['examples']['TP']['gold_coverage'] == 1
    assert a['ContractEval']['examples']['FP']['extraction_text'] == 'gh'
    assert a['ContractEval']['examples']['FN']['gold_text'] == 'abcd'
    assert len(report['cases']) == 6  # Union of model-specific populations, not their intersection.
    assert all(d['clause_type'] == third for d in report['diagnostics'] if d['model_name'] == 'B')


@pytest.mark.parametrize('predicted,expected', [
    ([(0, 3)], (0, 1, 1)),  # Exactly 75% of gold is NOT a match.
    ([(1, 4)], (0, 1, 1)),  # Coverage is independent of which end is omitted.
    ([(0, 8)], (1, 0, 0)),  # Fully included gold matches despite Jaccard being .5.
    ([(0, 7)], (1, 0, 0)),
    ([(0, 4)], (1, 0, 0)),  # Exact gold span.
    ([(0, 2), (2, 4)], (0, 2, 1)),  # Separate extractions cannot combine into a match.
    ([(4, 8)], (0, 1, 1)),  # No overlap.
    ([(0, 4), (0, 4)], (1, 1, 0)),  # Duplicate predictions remain FPs.
])
def test_strict_threshold_and_one_to_one(predicted, expected):
    case = ('one', 'governing_law')
    report = ct.evaluate([ct.Run('a', 'Harness', 'A', {case: GroundedAnswer(predicted)}),
                          ct.Run('a', 'ContractEval', 'A', {case: GroundedAnswer()})],
                         {case: ct.GoldCase('abcdefgh', [(0, 4)])})
    metrics = report['rows'][0]['Harness']
    assert tuple(metrics[k] for k in ['TP', 'FP', 'FN']) == expected


@pytest.mark.parametrize('prediction,matched', [
    ((1, 10), False),  # 80% coverage is insufficient.
    ((0, 10), True),   # Surrounding text is allowed.
])
def test_full_containment_preserves_actual_jaccard(prediction, matched):
    case = ('one', 'governing_law')
    gold = {case: ct.GoldCase('abcdefghij', [(0, 5)])}
    report = ct.evaluate([
        ct.Run('a', 'Harness', 'A', {case: GroundedAnswer([prediction])}),
        ct.Run('a', 'ContractEval', 'A', {case: GroundedAnswer()}),
    ], gold)
    metrics = report['rows'][0]['Harness']
    assert metrics['recall'] == int(matched)
    if matched:
        assert metrics['jaccard_mean'] == pytest.approx(0.5)
        match = report['diagnostics'][0]['matches'][0]
        assert match['gold_coverage'] == 1
        assert match['jaccard'] == pytest.approx(0.5)
    else:
        assert metrics['jaccard_mean'] is None
        assert metrics['FP'] == metrics['FN'] == 1
        assert report['diagnostics'][0]['matches'] == []
    assert '100% character coverage' in ct.render_html(report, ct.DEFAULT_CUAD_JSON)


def test_merged_gold_spans_and_ungrounded_output():
    case = ('one', 'governing_law')
    gold = {case: ct.GoldCase('abcdefgh', [(0, 4), (4, 8)])}
    prediction = GroundedAnswer([(0, 6)], ['invented clause'])
    report = ct.evaluate([ct.Run('a', 'ContractEval', 'A', {case: prediction}),
                          ct.Run('a', 'Harness', 'A', {case: GroundedAnswer()})], gold)
    metrics = report['rows'][0]['ContractEval']
    assert (metrics['TP'], metrics['FP'], metrics['FN']) == (1, 0, 1)
    assert metrics['precision'] == 1 and metrics['recall'] == 0.5
    assert metrics['f1'] == pytest.approx(2 / 3)
    assert metrics['examples']['FP'] is None
    assert metrics['ungrounded_predictions'] == 1
    assert metrics['jaccard_mean'] == pytest.approx(4 / 6)
    assert report['diagnostics'][0]['ungrounded_passages'] == ['invented clause']


@pytest.mark.parametrize('gold_spans', [[], [(0, 2)]])
def test_ungrounded_only_output_is_treated_as_no_extractions(gold_spans):
    case = ('one', 'governing_law')
    predictions = {case: GroundedAnswer([], ['invented clause'])}
    report = ct.evaluate([ct.Run('a', method, 'A', predictions) for method in ct.METHODS],
                         {case: ct.GoldCase('NY', gold_spans)})
    for method in ct.METHODS:
        metrics = report['rows'][0][method]
        assert (metrics['TP'], metrics['FP'], metrics['FN']) == (0, 0, len(gold_spans))
        assert metrics['precision'] == metrics['recall'] == metrics['f1'] == 0
        assert metrics['corpus_retained'] == 0
        assert metrics['examples']['FP'] is None
        assert (metrics['examples']['TN'] is not None) == (not gold_spans)
        assert metrics['ungrounded_predictions'] == 1


def test_no_positive_cases_and_disjoint_coverage():
    case = ('one', 'governing_law')
    run = ct.Run('a', 'Harness', 'A', {case: GroundedAnswer()})
    paired = ct.Run('a', 'ContractEval', 'A', {case: GroundedAnswer()})
    report = ct.evaluate([run, paired], {case: ct.GoldCase('NY', [])})
    assert report['rows'][0]['Harness']['jaccard_mean'] is None
    assert '<strong>0.000</strong>' in ct.render_html(report, ct.DEFAULT_CUAD_JSON)
    assert ct.evaluate([run, paired], {case: ct.GoldCase('NY', [(0, 2)])})['rows'][0]['Harness']['jaccard_mean'] is None
    with pytest.raises(ValueError, match='complete test coverage in both methods'):
        ct.evaluate([run, ct.Run('b', 'ContractEval', 'B', {('two', 'governing_law'): GroundedAnswer()})], {})
    with pytest.raises(ValueError, match='No matching'):
        ct.evaluate([], {})


def test_discovery_gold_offsets_empty_files_and_cli(tmp_path):
    title = ' Test Document '
    gold_path = tmp_path / 'CUADv1.json'
    gold_path.write_text(json.dumps({'data': [{'title': title, 'paragraphs': [{
        'context': 'prefix shared IP', 'qas': [{'id': f'{title}__Joint Ip Ownership',
        'answers': [{'text': 'shared IP', 'answer_start': 7}]}]}]}]}))
    root = tmp_path / 'cuad'
    harness = root / 'vendor_model/Test Document/joint_ip_ownership.jsonl'
    harness.parent.mkdir(parents=True)
    harness.write_text('')
    baseline = root / 'contracteval/vendor_model/Test Document/joint_ip_ownership.json'
    baseline.parent.mkdir(parents=True)
    baseline.write_text(json.dumps({'model_name': 'vendor/model', 'answer': 'shared IP',
                                    'labels': [], 'scores': {'classification': 'TN'}}))
    metrics = root / 'contracteval/vendor_model/metrics/joint_ip_ownership.json'
    metrics.parent.mkdir()
    metrics.write_text('{}')
    unknown = root / 'vendor_model/unknown/joint_ip_ownership.jsonl'
    unknown.parent.mkdir()
    unknown.write_text('')
    gold = ct.load_gold(gold_path)
    assert gold[(title, 'joint_ip_ownership')].spans == [(7, 16)]
    runs, unmatched = ct.discover_runs(root, gold)
    assert len(runs) == 2 and unmatched == [str(unknown)]
    report = ct.evaluate(runs, gold)
    assert report['rows'][0]['model_name'] == 'vendor/model'
    assert report['rows'][0]['ContractEval']['TP'] == 1
    assert report['rows'][0]['Harness']['FN'] == 1
    assert len(ct.discover_runs(root, gold, model_names=['vendor/model'])[0]) == 2
    with pytest.raises(ValueError, match='No matching outputs'):
        ct.discover_runs(root, gold, model_names=['other/model'])
    output = tmp_path / 'table.html'
    assert ct.main(['--input-root', str(root), '--cuad-json', str(gold_path),
                    '--test-json', str(gold_path),
                    '--clause-type', 'joint_ip_ownership', '--output', str(output)]) == 0
    assert 'Harness' in output.read_text()
    audit = json.loads(output.with_suffix('.spans.json').read_text())
    assert audit['diagnostics'][0]['predicted_spans'] == [[7, 16]]
    assert audit['test_document_count'] == 1
    assert audit['target'] == 'all'
    baseline.write_text(json.dumps({'answer': None}))
    with pytest.raises(ValueError, match='missing final answer'):
        ct.discover_runs(root, gold)


@pytest.mark.parametrize('legacy', [False, True])
def test_failed_contracteval_requests_remain_in_shared_evaluation(tmp_path, legacy):
    gold = {(title, 'governing_law'): ct.GoldCase('NY', [(0, 2)] if title == 'positive' else [])
            for title in ['positive', 'negative']}
    for title, clause in gold:
        harness = tmp_path / 'model' / title / f'{clause}.jsonl'
        harness.parent.mkdir(parents=True)
        harness.write_text('')
        if not legacy:
            failed = tmp_path / 'contracteval/model' / title / f'{clause}.json'
            failed.parent.mkdir(parents=True)
            failed.write_text(json.dumps({'status': 'failed', 'answer': None, 'error': 'timeout'}))
    if legacy:
        summary = tmp_path / 'contracteval/model/metrics/governing_law.json'
        summary.parent.mkdir(parents=True)
        summary.write_text(json.dumps({'failures': {'positive': 'timeout', 'negative': 'timeout'}}))
    runs, _ = ct.discover_runs(tmp_path, gold)
    report = ct.evaluate(runs, gold)
    assert len(report['cases']) == 2
    metrics = report['rows'][0]['ContractEval']
    assert metrics['FN'] == 1 and metrics['FP'] == metrics['TP'] == 0
    assert metrics['jaccard_mean'] is None
    assert metrics['recall'] == metrics['precision'] == 0
    assert metrics['failure_count'] == 2
    assert all(item['predicted_spans'] == [] for item in report['diagnostics'])


def test_complete_test_coverage_excludes_partial_runs_before_reading(tmp_path, capsys):
    titles = {f'doc{i:03}' for i in range(102)}
    clause = 'joint_ip_ownership'
    gold = {(title, clause): ct.GoldCase('shared IP', [(0, 9)])
            for title in titles | {'training'}}
    for title in sorted(titles | {'training'}):
        harness = tmp_path / 'complete' / title / f'{clause}.jsonl'
        harness.parent.mkdir(parents=True)
        harness.write_text('unreadable training output' if title == 'training' else '')
        if title == 'doc101':
            continue
        partial = tmp_path / 'contracteval/partial' / title / f'{clause}.json'
        partial.parent.mkdir(parents=True)
        partial.write_text('unreadable incomplete run')
        if title != 'training':
            complete = tmp_path / 'contracteval/complete' / title / f'{clause}.json'
            complete.parent.mkdir(parents=True)
            complete.write_text(json.dumps({'answer': 'shared IP'}))
    # A recorded failure completes coverage and remains an empty extraction.
    summary = tmp_path / 'contracteval/complete/metrics' / f'{clause}.json'
    summary.parent.mkdir(parents=True)
    summary.write_text(json.dumps({'failures': {'doc101': 'timeout'}}))
    runs, unmatched = ct.discover_runs(tmp_path, gold, clause, test_titles=titles)
    assert unmatched == []
    assert {run.model_key for run in runs} == {'complete'}
    report = ct.evaluate(runs, gold, test_titles=titles)
    assert len(report['cases']) == 102
    assert {title for title, _ in report['cases']} == titles
    assert report['rows'][0]['ContractEval']['failure_count'] == 1
    assert report['coverage'][1]['excluded'] == 1  # Training input was not read.
    assert 'partial/joint_ip_ownership: 101/102 test documents' in capsys.readouterr().err
    with pytest.raises(ValueError, match='complete test coverage'):
        ct.discover_runs(tmp_path, gold, clause, model_names=['partial'], test_titles=titles)


def test_harness_offsets_and_title_override(tmp_path):
    from references.cuad.compare_extractions import DOCUMENT_ID_TITLE_OVERRIDES
    document_id, title = next(iter(DOCUMENT_ID_TITLE_OVERRIDES.items()))
    file = tmp_path / 'vendor_model' / document_id / 'joint_ip_ownership.jsonl'
    file.parent.mkdir(parents=True)
    record = {'extraction_text': 'first', 'span_start': 0, 'span_end': 5, 'model_name': 'vendor/model'}
    file.write_text(json.dumps(record))
    baseline = tmp_path / 'contracteval/vendor_model' / document_id / 'joint_ip_ownership.json'
    baseline.parent.mkdir(parents=True)
    baseline.write_text(json.dumps({'answer': 'first', 'model_name': 'vendor/model'}))
    gold = {(title, 'joint_ip_ownership'): ct.GoldCase('first second', [(0, 5)])}
    runs, unmatched = ct.discover_runs(tmp_path, gold)
    assert unmatched == []
    assert runs[1].predictions[(title, 'joint_ip_ownership')].spans == [(0, 5)]
    assert ct.evaluate(runs, gold)['rows'][0]['Harness']['TP'] == 1
    record['span_start'], record['span_end'] = 1, 6
    file.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='do not match original CUAD'):
        ct.discover_runs(tmp_path, gold)


@pytest.mark.parametrize('answer', ['No related clause.', 'NO RELATED CLAUSE', '"No related clause."'])
def test_abstention(answer):
    result = ground_answer(answer, 'Some unrelated contract.')
    assert result.spans == result.ungrounded == []


def test_grounding_whitespace_wrappers_and_original_offsets():
    context = 'Prefix. First  clause\ncontinues. Unrelated. Second clause. End.'
    result = ground_answer('```text\n- "First clause continues."\n\n2. Second clause.\n```', context)
    assert [context[a:b] for a, b in result.spans] == ['First  clause\ncontinues.', 'Second clause.']
    assert result.ungrounded == []
    assert ground_answer('```First clause```', 'First clause').spans == [(0, 12)]


def test_discontiguous_sentences_merged_only_across_whitespace():
    context = 'First clause. Second clause. OMITTED. Last clause.'
    result = ground_answer('First clause.\nSecond clause. Last clause.', context)
    assert [context[a:b] for a, b in result.spans] == ['First clause. Second clause.', 'Last clause.']
    assert result.ungrounded == []


def test_ungrounded_and_altered_text_not_silently_dropped():
    result = ground_answer('First clause. Invented clause. Last clause.', 'First clause. Last clause.')
    assert result.spans == [(0, 13), (14, 26)]
    assert result.ungrounded == ['Invented clause.']
    assert ground_answer('FIRST clause.', 'First clause.').ungrounded == ['FIRST clause.']
    assert ground_answer('NY', 'COMPANY').spans == []


def test_repeated_occurrences_and_duplicates_are_not_expanded_or_deduplicated():
    assert ground_answer('Repeat.', 'Repeat. Repeat.').spans == [(0, 7)]
    result = ground_answer('Repeat.\n\nRepeat.', 'Repeat. Repeat.')
    assert result.spans == [(0, 7), (8, 15)] and result.ambiguous_passages == 2
    result = ground_answer('Repeat.\n\nRepeat.', 'Repeat.')
    assert result.spans == [(0, 7), (0, 7)]


def test_adjacent_list_items_stay_separate_predictions():
    context = 'First clause. Second clause.'
    result = ground_answer('- First clause.\n- Second clause.', context)
    assert [context[a:b] for a, b in result.spans] == ['First clause.', 'Second clause.']
    assert result.ungrounded == []


def test_discovery_only_reads_shared_cases(tmp_path):
    gold = {(title, 'joint_ip_ownership'): ct.GoldCase('shared IP', [(0, 9)])
            for title in ['shared', 'training']}
    for title in gold:
        path = tmp_path / 'model' / title[0] / 'joint_ip_ownership.jsonl'
        path.parent.mkdir(parents=True)
        path.write_text('invalid unused output' if title[0] == 'training' else '')
    path = tmp_path / 'contracteval/model/shared/joint_ip_ownership.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'answer': 'shared IP'}))
    runs, _ = ct.discover_runs(tmp_path, gold, test_titles={'shared'})
    report = ct.evaluate(runs, gold, test_titles={'shared'})
    assert report['cases'] == [('shared', 'joint_ip_ownership')]
    assert report['coverage'][1]['available'] == 2
    assert report['coverage'][1]['excluded'] == 1


def test_discovery_requires_both_methods_for_same_model_and_type(tmp_path):
    titles = {'one', 'two'}
    types = ['governing_law', 'joint_ip_ownership']
    gold = {(title, clause): ct.GoldCase('NY', [(0, 2)]) for title in titles for clause in types}
    for model in ['a', 'b', 'unpaired']:
        for title, clause in gold:
            ce = tmp_path / 'contracteval' / model / title / f'{clause}.json'
            ce.parent.mkdir(parents=True, exist_ok=True)
            include = model == 'a' or (model == 'b' and clause == types[1])
            ce.write_text(json.dumps({'answer': 'NY'}) if include else 'invalid unpaired output')
            if include:
                harness = tmp_path / model / title / f'{clause}.jsonl'
                harness.parent.mkdir(parents=True, exist_ok=True)
                harness.write_text('')
    # Another model's Harness output must not complete the unpaired model.
    for title, clause in gold:
        path = tmp_path / 'harness_only' / title / f'{clause}.jsonl'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('invalid unpaired output')
    runs, _ = ct.discover_runs(tmp_path, gold, test_titles=titles)
    report = ct.evaluate(runs, gold, test_titles=titles)
    assert [(row['model_name'], row['n_types']) for row in report['rows']] == [('a', 2), ('b', 1)]
    assert len(runs) == 4


def test_min_5_target_filters_outputs_and_metrics_before_reading(tmp_path):
    from pipeline.stg_02_extract import orchestrator

    assert ct.MIN_5_CLAUSES == orchestrator.MIN_5_CLAUSES
    clauses = [*ct.MIN_5_CLAUSES, 'governing_law']
    gold_path = tmp_path / 'gold.json'
    gold_path.write_text(json.dumps({'data': [
        {'title': title, 'paragraphs': [{'context': 'NY', 'qas': [
            {'id': title + '__' + ct.CATEGORY_BY_CLAUSE_TYPE[clause],
             'answers': [{'text': 'NY', 'answer_start': 0}]} for clause in clauses
        ]}]} for title in ['one', 'two']
    ]}))
    root = tmp_path / 'cuad'
    for title in ['one', 'two']:
        for clause in clauses:
            baseline = root / 'contracteval/model' / title / f'{clause}.json'
            baseline.parent.mkdir(parents=True, exist_ok=True)
            baseline.write_text(json.dumps({'answer': 'NY'}))
            harness = root / 'model' / title / f'{clause}.jsonl'
            harness.parent.mkdir(parents=True, exist_ok=True)
            harness.write_text('')
    output = tmp_path / 'table.html'
    args = ['--input-root', str(root), '--cuad-json', str(gold_path),
            '--test-json', str(gold_path), '--output', str(output)]
    assert ct.main(args) == 0
    assert json.loads(output.with_suffix('.spans.json').read_text())['rows'][0]['n_types'] == 6

    # An excluded clause must not be opened, including its legacy metrics.
    outside = root / 'contracteval/model/one/governing_law.json'
    outside.write_text('invalid excluded output')
    metrics = root / 'contracteval/model/metrics/governing_law.json'
    metrics.parent.mkdir()
    metrics.write_text('invalid excluded metrics')
    assert ct.main(args + ['--target', 'min_5']) == 0
    report = json.loads(output.with_suffix('.spans.json').read_text())
    assert report['target'] == 'min_5'
    assert report['requested_clause_types'] == sorted(ct.MIN_5_CLAUSES)
    assert report['rows'][0]['n_types'] == 5
    assert {clause for _, clause in report['cases']} == set(ct.MIN_5_CLAUSES)
    assert 'Target: min_5' not in output.read_text()

    (root / 'model/two/anti_assignment.jsonl').unlink()
    assert ct.main(args + ['--target', 'min_5']) == 0
    report = json.loads(output.with_suffix('.spans.json').read_text())
    assert report['rows'][0]['n_types'] == 4
    assert 'anti_assignment' not in report['rows'][0]['clause_types']
    assert ct.main(args + ['--target', 'min_5', '--clause-type', 'audit_rights']) == 0
    assert json.loads(output.with_suffix('.spans.json').read_text())['rows'][0]['n_types'] == 1


@pytest.mark.parametrize('args', [
    ['--target', 'unknown'],
    ['--target', 'min_5', '--clause-type', 'joint_ip_ownership'],
])
def test_comparison_target_rejects_invalid_selection(args):
    with pytest.raises(SystemExit):
        ct.main(args)
