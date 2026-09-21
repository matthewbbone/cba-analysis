import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from pipeline.stg_02_extract import contracteval as ce
from pipeline.stg_02_extract.structure_provision import load_provision, resolve_provisions_dir
from references.cuad.compare_extractions import DOCUMENT_ID_TITLE_OVERRIDES


@pytest.mark.parametrize('answer,labels,classification', [
    ('The law is NY.', ['NY'], 'TP'),
    ('The law is NY.', ['NY', 'Delaware'], 'FN'),
    ('the law is ny.', ['NY'], 'FN'),
    ('` NY \n`', [' `NY\n'], 'TP'),
    ('No related clause.', ['NY'], 'FN'),
    ('Explanation: NO RELATED CLAUSE.', [], 'TN'),
    ('New York', [], 'FP'),
])
def test_reference_classification(answer, labels, classification):
    assert ce.score_answer(answer, labels)['classification'] == classification


def test_reference_jaccard_and_positive_denominator():
    # Literal-space splitting preserves empty tokens and embedded newlines.
    assert ce.score_answer('A B', ['a/b.'])['jaccard'] == 1
    assert ce.score_answer('a b', ['a  b'])['jaccard'] == 2 / 3
    assert ce.score_answer('a b', ['a\nb'])['jaccard'] == 0
    scores = {
        'a': ce.score_answer('No related clause.', ['NY']),
        'b': ce.score_answer('NY', ['NY']),
        'c': ce.score_answer('No related clause.', []),
        'd': ce.score_answer('NY', []),
    }
    result = ce.aggregate(scores)
    assert [result[k] for k in ['TP', 'TN', 'FP', 'FN']] == [1, 1, 1, 1]
    assert result['false_abstention_rate'] == 0.5
    assert result['jaccard_mean'] == 0.5
    assert result['f1'] == result['f2'] == 0.5
    missed_positive = ce.aggregate({'a': scores['a'], 'b': scores['b']})
    assert missed_positive['f1'] == 2 / 3
    assert missed_positive['f2'] == 5 / 9
    assert ce.aggregate({})['jaccard_mean'] is None
    assert ce.aggregate({'a': scores['c']})['false_abstention_rate'] is None
    assert ce.aggregate({'a': scores['c']})['f1'] == 0


def test_local_question_and_reference_prompt():
    provision = load_provision('governing_law', provisions_dir=resolve_provisions_dir('cuad'))
    assert provision.clause_description == "Which state/country's law governs the interpretation of the contract?"
    assert ce.build_question(provision) == (
        'Highlight the parts (if any) of this contract related to "Governing Law" '
        'that should be reviewed by a lawyer. Details: ' + provision.clause_description
    )
    context = 'a\n  page 3\n``` unmodified\n'
    assert ce.PROMPT.format(context=context, question='Q') == (
        'Context:\n```\n' + context + '\n```\nQuestion:\n```\nQ\n```\n'
    )
    assert 'Do not rephrase or summarize' in ce.SYSTEM_PROMPT
    assert 'chunks' not in ce.SYSTEM_PROMPT


@pytest.fixture
def setup(tmp_path, monkeypatch):
    entries = []
    for title, labels in [('doc1', ['NY']), ('doc2', [])]:
        entries.append({'title': title, 'paragraphs': [{
            'context': 'original context',
            'qas': [{'id': title + '__Governing Law',
                     'answers': [{'text': label, 'answer_start': 0} for label in labels]}],
        }]})
        path = tmp_path / 'input/cuad/ocr' / title / 'full.txt'
        path.parent.mkdir(parents=True)
        path.write_text('OCR NY\n ')
    test_json = tmp_path / 'test.json'
    test_json.write_text(json.dumps({'data': entries}))
    monkeypatch.setattr(ce, 'TEST_JSON', test_json)
    monkeypatch.setenv('OPENROUTER_API_KEY', 'test-key')
    monkeypatch.setattr(ce.runner, 'validate_cuda_device_selection', lambda device, num_gpus: device)
    args = [
        '--provision', 'governing_law', '--model-name', 'vendor/model',
        '--endpoint', 'openrouter', '--input-root', str(tmp_path / 'input'),
        '--output-root', str(tmp_path / 'out'), '--ocr-model-name', 'ocr', '--no-progress',
    ]
    server = Mock()
    factory = Mock(return_value=server)
    monkeypatch.setattr(ce, 'VLLMServer', factory)
    client = Mock()
    client.chat.completions.create.return_value = response('NY')
    import openai
    client_factory = Mock(return_value=client)
    monkeypatch.setattr(openai, 'OpenAI', client_factory)
    return SimpleNamespace(args=args, root=tmp_path, server=server, factory=factory,
                           client=client, client_factory=client_factory, entries=entries,
                           test_json=test_json)


def response(answer, finish='stop'):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=answer, reasoning='private reasoning'),
                                 finish_reason=finish)],
        model='returned/model', usage=SimpleNamespace(model_dump=lambda: {'completion_tokens': 10}),
    )


def read_summary(setup):
    return json.loads((setup.root / 'out/vendor_model/metrics/governing_law.json').read_text())


def test_full_run_cache_and_offline_rescore(setup):
    assert ce.main(setup.args) == 0
    assert setup.client.chat.completions.create.call_count == 2
    call = setup.client.chat.completions.create.call_args.kwargs
    assert call['messages'][1]['content'].startswith('Context:\n```\nOCR NY\n ')
    assert call['max_tokens'] == 5000 and call['temperature'] == 0
    assert 'response_format' not in call
    assert call['top_p'] == 0.9
    assert setup.client_factory.call_args.kwargs['base_url'] == 'https://openrouter.ai/api/v1'
    setup.server.start.assert_called_once()
    setup.server.close.assert_called_once()
    setup.client.close.assert_called_once()
    path = setup.root / 'out/vendor_model/doc1/governing_law.json'
    record = json.loads(path.read_text())
    assert record['labels'] == ['NY'] and record['answer'] == 'NY'
    assert record['returned_model'] == 'returned/model'
    assert record['reasoning'] == 'private reasoning'
    assert record['scores']['classification'] == 'TP'
    assert not list(path.parent.glob('.*.json.*'))
    setup.factory.reset_mock()
    assert ce.main(setup.args) == 0
    assert ce.main(setup.args + ['--evaluate-only', '--temperature', 'default']) == 0
    setup.factory.assert_not_called()
    assert read_summary(setup)['metrics']['count'] == 2


def test_borrowed_server_preserves_requests_cache_and_ownership(setup):
    args = ce.parse_args(setup.args + ['--endpoint', 'vllm'])
    borrowed = Mock(endpoint=args.endpoint, model_name=args.model_name,
                    port=args.port, max_model_len=args.max_model_len)
    summary = ce.run(args, server=borrowed)
    assert summary['execution'] == {'completed': 2, 'reused': 0, 'failed': 0}
    original_request = setup.client.chat.completions.create.call_args.kwargs
    original = json.loads((setup.root / 'out/vendor_model/doc1/governing_law.json').read_text())
    assert ce.run(args, server=borrowed)['execution']['reused'] == 2
    assert setup.client.chat.completions.create.call_count == 2
    setup.factory.assert_not_called()
    borrowed.start.assert_not_called()
    borrowed.close.assert_not_called()
    assert ce.main(setup.args + ['--endpoint', 'vllm', '--force']) == 0
    assert setup.client.chat.completions.create.call_args.kwargs == original_request
    repeated = json.loads((setup.root / 'out/vendor_model/doc1/governing_law.json').read_text())
    assert repeated['fingerprint'] == original['fingerprint']
    assert repeated['generation'] == original['generation']


def test_borrowed_server_failure_does_not_close_owner(setup):
    args = ce.parse_args(setup.args + ['--endpoint', 'vllm'])
    borrowed = Mock(endpoint=args.endpoint, model_name=args.model_name,
                    port=args.port, max_model_len=args.max_model_len)
    setup.client.chat.completions.create.side_effect = RuntimeError('timeout')
    summary = ce.run(args, server=borrowed)
    assert summary['execution']['failed'] == 2
    assert summary['metrics']['count'] == 2
    borrowed.close.assert_not_called()
    borrowed.start.assert_not_called()
    setup.client.close.assert_called_once()


def test_borrowed_server_identity_mismatch_is_rejected(setup):
    args = ce.parse_args(setup.args + ['--endpoint', 'vllm'])
    borrowed = Mock(endpoint=args.endpoint, model_name='another/model',
                    port=args.port, max_model_len=args.max_model_len)
    with pytest.raises(ValueError, match='Borrowed server model_name'):
        ce.run(args, server=borrowed)
    setup.client.chat.completions.create.assert_not_called()


@pytest.mark.parametrize('change', ['input', 'settings', 'prompt'])
def test_cache_mismatch_before_inference(setup, monkeypatch, change):
    ce.main(setup.args)
    setup.factory.reset_mock()
    args = setup.args
    if change == 'input':
        (setup.root / 'input/cuad/ocr/doc1/full.txt').write_text('changed')
    elif change == 'settings':
        args = args + ['--max-tokens', '6000']
    else:
        monkeypatch.setattr(ce, 'SYSTEM_PROMPT', ce.SYSTEM_PROMPT + 'changed')
    with pytest.raises(ValueError, match='mismatch'):
        ce.main(args)
    setup.factory.assert_not_called()
    assert ce.main(args + ['--force']) == 0
    setup.factory.assert_called_once()


def test_failed_request_and_truncated_answer(setup):
    setup.client.chat.completions.create.side_effect = [RuntimeError('request failed'), response('No related clause.', 'length')]
    assert ce.main(setup.args) == 1
    result = read_summary(setup)
    assert result['metrics']['document_ids'] == ['doc1', 'doc2']
    assert result['metrics']['FN'] == result['metrics']['TN'] == 1
    assert result['metrics']['jaccard_mean'] == 0
    assert result['metrics']['false_abstention_rate'] == 1
    assert result['failures'] == {'doc1': 'request failed'}
    assert result['truncation_count'] == 1
    record = json.loads((setup.root / 'out/vendor_model/doc1/governing_law.json').read_text())
    assert record['status'] == 'failed' and record['answer'] is None
    assert record['error'] == 'request failed'
    setup.server.close.assert_called_once()
    setup.factory.reset_mock()
    assert ce.main(setup.args + ['--evaluate-only']) == 1
    setup.factory.assert_not_called()
    assert read_summary(setup)['metrics'] == result['metrics']
    assert read_summary(setup)['failures'] == result['failures']
    assert read_summary(setup)['missing_outputs'] == []

    setup.client.chat.completions.create.side_effect = [response('NY')]
    assert ce.main(setup.args) == 0
    assert read_summary(setup)['failures'] == {}
    assert read_summary(setup)['metrics']['TP'] == 1


def test_missing_final_answer_is_failure(setup):
    setup.client.chat.completions.create.return_value = response(None, 'length')
    assert ce.main(setup.args) == 1
    result = read_summary(setup)
    assert result['metrics']['count'] == 2
    assert result['metrics']['FN'] == result['metrics']['TN'] == 1
    assert result['truncation_count'] == 2
    assert len(result['failures']) == 2


def test_cleanup_on_start_failure(setup):
    setup.server.start.side_effect = RuntimeError('cannot start')
    with pytest.raises(RuntimeError, match='cannot start'):
        ce.main(setup.args)
    setup.server.close.assert_called_once()
    setup.client_factory.assert_not_called()


def test_vllm_settings_and_thinking(setup):
    args = setup.args + ['--endpoint', 'vllm', '--model-name', 'Qwen/Qwen3-8B',
                         '--thinking', 'off', '--temperature', 'default', '--device', '0']
    assert ce.main(args) == 0
    served = setup.factory.call_args.kwargs
    assert served['max_model_len'] == 131072
    assert served['extra_serve_args'] == ['--generation-config', 'vllm', '--reasoning-parser', 'qwen3']
    request = setup.client.chat.completions.create.call_args.kwargs
    assert request['extra_body'] == {'chat_template_kwargs': {'enable_thinking': False}}
    assert 'temperature' not in request
    assert setup.client_factory.call_args.kwargs['base_url'] == 'http://localhost:8123/v1'
    settings = ce.inference_settings(ce.parse_args(setup.args + ['--thinking', 'on']))
    assert settings['request']['extra_body'] == {'reasoning': {'enabled': True}}
    assert 'extra_body' not in ce.inference_settings(ce.parse_args(setup.args))['request']


def test_discovery_test_split_override_and_missing_input(setup):
    override_id, title = next(iter(DOCUMENT_ID_TITLE_OVERRIDES.items()))
    setup.entries[0]['title'] = title
    setup.entries[0]['paragraphs'][0]['qas'][0]['id'] = title + '__Governing Law'
    setup.test_json.write_text(json.dumps({'data': setup.entries}))
    (setup.root / 'input/cuad/ocr/doc1').rename(setup.root / 'input/cuad/ocr' / override_id)
    # An unrelated cached training document must not be queried.
    unrelated = setup.root / 'input/cuad/ocr/train/full.txt'
    unrelated.parent.mkdir()
    unrelated.write_text('training text')
    (setup.root / 'input/cuad/ocr/doc2/full.txt').unlink()
    jobs, labels, missing = ce.discover(ce.parse_args(setup.args), 'governing_law')
    assert len(jobs) == 1 and jobs[0].document_id == override_id
    assert labels[override_id] == ['NY'] and missing == ['doc2']
    assert jobs[0].output_path == setup.root / 'out/vendor_model' / override_id / 'governing_law.json'
    with pytest.raises(ValueError, match='outside CUAD test split'):
        ce.discover(ce.parse_args(setup.args + ['--document-id', 'train']), 'governing_law')


def test_sampling_and_document_selection(setup):
    args = ce.parse_args(setup.args + ['--sample', '1', '--seed', '5'])
    assert ce.discover(args, 'governing_law')[0] == ce.discover(args, 'governing_law')[0]
    assert len(ce.discover(args, 'governing_law')[0]) == 1
    assert ce.main(setup.args + ['--document-id', 'doc2']) == 0
    assert read_summary(setup)['metrics']['document_ids'] == ['doc2']


def test_evaluate_only_missing_outputs(setup):
    assert ce.main(setup.args + ['--evaluate-only']) == 1
    setup.factory.assert_not_called()
    assert read_summary(setup)['missing_outputs'] == ['doc1', 'doc2']
    assert read_summary(setup)['metrics']['document_ids'] == ['doc1', 'doc2']
    assert read_summary(setup)['metrics']['FN'] == read_summary(setup)['metrics']['TN'] == 1


def test_failed_requests_in_paired_comparison(setup):
    setup.client.chat.completions.create.side_effect = RuntimeError('timeout')
    baseline = setup.root / 'baseline/cuad/old_model/doc1/governing_law.jsonl'
    baseline.parent.mkdir(parents=True)
    baseline.write_text(json.dumps({'extraction_text': 'NY', 'ocr_model_name': 'ocr'}))
    args = setup.args + ['--compare-runner-model', 'old/model',
                         '--runner-output-root', str(setup.root / 'baseline')]
    assert ce.main(args) == 1
    comp = read_summary(setup)['comparison']
    assert comp['contracteval']['document_ids'] == comp['runner']['document_ids'] == ['doc1']
    assert comp['contracteval']['FN'] == comp['runner']['TP'] == 1
    assert comp['missing_runner_outputs'] == ['doc2']


def test_empty_extraction_has_zero_jaccard_even_with_empty_gold_tokens():
    scores = ce.score_empty_extraction(['a  b'])
    assert scores['jaccard'] == 0
    assert scores['classification'] == 'FN'
    assert ce.aggregate({'negative': ce.score_empty_extraction([])})['jaccard_mean'] is None


def test_comparison_uses_common_documents_and_empty_file(setup):
    ce.main(setup.args)
    baseline = setup.root / 'baseline/cuad/old_model/doc1/governing_law.jsonl'
    baseline.parent.mkdir(parents=True)
    baseline.write_text('')
    args = setup.args + ['--evaluate-only', '--compare-runner-model', 'old/model',
                         '--runner-output-root', str(setup.root / 'baseline')]
    ce.main(args)
    comp = read_summary(setup)['comparison']
    assert comp['runner']['document_ids'] == comp['contracteval']['document_ids'] == ['doc1']
    assert comp['runner']['FN'] == 1
    assert comp['runner']['false_abstention_rate'] == 1
    assert comp['missing_runner_outputs'] == ['doc2']
    baseline.write_text(json.dumps({'extraction_text': 'NY', 'ocr_model_name': 'ocr'}) + '\n')
    ce.main(args)
    assert read_summary(setup)['comparison']['runner']['TP'] == 1
    baseline.write_text(json.dumps({'extraction_text': 'NY', 'ocr_model_name': 'different'}) + '\n')
    assert ce.main(args) == 1
    assert read_summary(setup)['comparison']['runner']['count'] == 0


def test_atomic_write_preserves_previous_result_on_error(tmp_path):
    target = tmp_path / 'result.json'
    ce.atomic_json(target, {'old': True})
    with pytest.raises(TypeError):
        ce.atomic_json(target, {'invalid': object()})
    assert json.loads(target.read_text()) == {'old': True}
    assert list(tmp_path.iterdir()) == [target]
