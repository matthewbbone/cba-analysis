"""Sentence-ID grounding and mocked execution; never starts an inference server."""
from dataclasses import replace
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from openai.types.chat import ChatCompletion

from pipeline.stg_02_extract import runner, runner2 as r
from pipeline.stg_02_extract.orchestrator import MODELS
from pipeline.stg_02_extract.structure_provision import load_provision
from pipeline.utils.paths import path_safe_model_name


def response(content='{"extractions": []}', finish='stop'):
    return ChatCompletion.model_validate({
        'id': 'mock-request', 'object': 'chat.completion', 'created': 0, 'model': 'returned-model',
        'usage': {'prompt_tokens': 15, 'completion_tokens': 5, 'total_tokens': 20},
        'choices': [{'index': 0, 'finish_reason': finish, 'message': {
            'role': 'assistant', 'content': content, 'reasoning_content': 'separate reasoning'}}]})


def selection(a, b, p=1, c='P1C1'):
    return dict(start_sentence=f'S{a}', end_sentence=f'S{b}', pass_number=p, chunk_id=c)


@pytest.fixture
def provision():
    return replace(load_provision('technology'), extraction_passes=2,
                   max_char_buffer=[60, 120], langextract_max_workers=2, langextract_batch_length=3)


@pytest.fixture
def args(tmp_path):
    return r.parse_args(['--provision', 'technology', '--source', 'negotiating_tech',
                         '--input-root', str(tmp_path / 'input'), '--output-root', str(tmp_path / 'output'),
                         '--ocr-model-name', 'ocr/model', '--model-name', 'extract/model',
                         '--device', '0', '--no-progress'])


def input_file(args, document='doc', text='First sentence. Second sentence.', source=None):
    path = args.input_root / (source or args.source) / path_safe_model_name(args.ocr_model_name) / document / 'full.txt'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode('utf-8'))
    return path


def job_for(args, document='doc', text='First sentence. Second sentence.'):
    path = input_file(args, document, text)
    return runner.ExtractionJob(args.source, document, args.ocr_model_name, args.model_name,
                                args.provision, path, args.output_root / args.source /
                                path_safe_model_name(args.model_name) / document / f'{args.provision}.jsonl')


def assert_coverage(text, units):
    cursor = 0
    for n, unit in enumerate(units, 1):
        assert unit.ordinal == n and unit.id == f'S{n}'
        assert cursor <= unit.start < unit.end <= len(text)
        assert not text[cursor:unit.start].strip()
        cursor = unit.end
    assert not text[cursor:].strip()


def test_stable_numeric_ids_across_passes_and_markers(provision):
    text = '# Technology\n\n' + ' '.join(f'Notice number {i} must be given.' for i in range(105))
    units = r.build_sentence_map(text)
    original = list(units)
    assert len(units) == 106
    for p, budget in enumerate((100, 350), 1):
        chunks = r.build_chunks(text, units, budget, p)
        primary = [i for c in chunks for i in range(c.primary[0], c.primary[1] + 1)]
        assert primary == list(range(1, 107))
        for chunk in chunks:
            assert chunk.displayed == tuple(sorted(chunk.displayed))
            assert 1 in chunk.displayed  # enclosing heading
            a, b = chunk.primary
            assert max(1, a - 1) in chunk.displayed
            assert min(106, b + 1) in chunk.displayed
        assert units == original
    chunk = r.Chunk(1, 'P1C1', (9, 100), tuple(range(1, 107)))
    prompt = r.build_prompt(provision, text, units, chunk)
    assert '[S9]' in prompt and '[S10]' in prompt and '[S99]' in prompt and '[S100]' in prompt
    assert '[S01]' not in prompt
    assert provision.clause_description in prompt and 'Clause type:\ntechnology' in prompt
    payload = json.dumps({'extractions': [{'start_sentence': 'S9', 'end_sentence': 'S100'}]})
    assert r.parse_ranges(payload, chunk)[0] == selection(9, 100)
    endpoints = r.response_schema(chunk)['properties']['extractions']['items']['properties']
    assert endpoints['start_sentence']['enum'][9] == 'S10'


def test_structures_unicode_repeated_text_crlf_and_ocr_line_wraps():
    text = ('  # Rights\r\n\r\nCafé workers 🤝 receive\r\nnotice. Same sentence. Same sentence.\r\n\r\n'
            '1. Notify the union. Meet promptly.\r\n- Preserve benefits!\r\n\r\n'
            '| Kind | Value. Another sentence. |\r\n| --- | --- |\r\n\r\n'
            'New heading\n===========\nDr. Smith attends.\n\n'
            'Kind | Value\n--- | ---\nNotice. More notice. | Required\n')
    units = r.build_sentence_map(text)
    assert_coverage(text, units)
    slices = [text[u.start:u.end] for u in units]
    assert slices[:4] == ['# Rights', 'Café workers 🤝 receive\r\nnotice.', 'Same sentence.', 'Same sentence.']
    assert '1. Notify the union.' in slices and 'Meet promptly.' in slices
    assert '| Kind | Value. Another sentence. |' in slices
    assert 'New heading\n===========' in slices
    assert 'Dr. Smith attends.' in slices
    assert 'Notice. More notice. | Required' in slices
    repeated = [u for u in units if text[u.start:u.end] == 'Same sentence.']
    assert repeated[0].start != repeated[1].start


@pytest.mark.parametrize('text', ['', ' \n\t\r\n'])
def test_empty_map_and_chunks(text):
    assert r.build_sentence_map(text) == []
    assert r.build_chunks(text, [], 20, 1) == []


def test_oversized_units_and_whitespace_are_not_truncated():
    text = '# Header\n\n' + 'á' * 600 + '\n\n| ' + 'row. ' * 80 + '|'
    units = r.build_sentence_map(text)
    assert len(units) == 3
    chunks = r.build_chunks(text, units, 15, 1)
    assert [i for c in chunks for i in range(c.primary[0], c.primary[1] + 1)] == [1, 2, 3]
    assert_coverage(text, units)


@pytest.mark.parametrize('prepared', [
    [SimpleNamespace(start_index=1, end_index=4, text='ext')],
    [SimpleNamespace(start_index=0, end_index=4, text='fake')],
    [SimpleNamespace(start_index=0, end_index=2, text='te')],
])
def test_adapter_rejects_misalignment_or_lost_content(prepared):
    with patch('chonkie.SentenceChunker._prepare_sentences', return_value=prepared):
        with pytest.raises(ValueError, match='source'):
            r.sentence_spans('text')


@pytest.mark.parametrize('payload', [
    'not json', '[]', '{}', '{"extractions": null}',
    '{"extractions": [], "extra": 1}',
    '{"extractions": [{"start_sentence": "[S9]", "end_sentence": "S10"}]}',
    '{"extractions": [{"start_sentence": "S09", "end_sentence": "S10"}]}',
    '{"extractions": [{"start_sentence": "s9", "end_sentence": "S10"}]}',
    '{"extractions": [{"start_sentence": "S10", "end_sentence": "S9"}]}',
    '{"extractions": [{"start_sentence": "S1", "end_sentence": "S10"}]}',
    '{"extractions": [{"start_sentence": [], "end_sentence": "S10"}]}',
    '{"extractions": [{"start_sentence": "S9", "end_sentence": "S11"}]}',
])
def test_strict_range_validation(payload):
    with pytest.raises(ValueError):
        r.parse_ranges(payload, r.Chunk(1, 'P1C1', (9, 10), (1, 8, 9, 10)))


def test_corrective_retry_and_response_metadata(provision):
    client = Mock()
    client.chat.completions.create.side_effect = [response('bad'), response(
        '{"extractions": [{"start_sentence": "S1", "end_sentence": "S2"}]}')]
    text = 'First sentence. Second sentence.'
    logs = []
    got = r.select_ranges(client, {'model': 'test', 'max_tokens': 5000}, provision, text,
                          r.build_sentence_map(text), r.Chunk(1, 'P1C1', (1, 2), (1, 2)), logs)
    assert got == [selection(1, 2)]
    assert len(logs) == 2 and 'validation_error' in logs[0]
    assert logs[1]['response']['model'] == 'returned-model'
    assert logs[1]['response']['usage']['total_tokens'] == 20
    assert logs[1]['response']['choices'][0]['message']['reasoning_content'] == 'separate reasoning'
    request = client.chat.completions.create.call_args.kwargs
    assert request['response_format']['json_schema']['strict']
    assert 'corrected JSON' in request['messages'][-1]['content']


@pytest.mark.parametrize('responses,expected_calls', [
    ([response('bad'), response('bad again')], 2),
    ([response(None)], 1), ([response('')], 1),
    ([response('{"extractions": []}', 'length')], 1),
    ([response('{"extractions": []}', 'content_filter')], 1),
    ([RuntimeError('context window exceeded')], 1),
])
def test_bad_responses_never_become_abstentions(provision, responses, expected_calls):
    client = Mock()
    client.chat.completions.create.side_effect = responses
    with pytest.raises((ValueError, RuntimeError)):
        r.select_ranges(client, {}, provision, 'Text.', r.build_sentence_map('Text.'),
                        r.Chunk(1, 'P1C1', (1, 1), (1,)), [])
    assert client.chat.completions.create.call_count == expected_calls


def test_overlap_resolution_and_exact_record_contract(args):
    text = ' '.join(f'Notice {i} applies.' for i in range(14))
    job = job_for(args, text=text)
    items = [selection(2, 4), selection(2, 4), selection(3, 3), selection(4, 6, 2),
             selection(6, 8), selection(9, 10), selection(13, 14)]
    resolved = r.resolve_ranges(list(reversed(items)))
    assert [(x['first'], x['last']) for x in resolved] == [(2, 8), (9, 10), (13, 14)]
    assert len(resolved[0]['selections']) == 4
    units = r.build_sentence_map(text)
    records = r.make_records(job, text, units, items)
    for record in records:
        assert record['extraction_text'] == text[record['span_start']:record['span_end']]
        assert record['generated_extraction_text'] == record['extraction_text']
        assert record['span_reliable'] is True and record['grounding_status'] == 'sentence_ids'
        assert record['extraction_class'] == 'technology'
    assert records[0]['end_sentence'] == 'S8'
    assert records[1]['start_sentence'] == 'S9'


def test_document_manifest_cache_retries_and_atomic_completion(args, provision):
    text = '# Heading\r\n\r\nNotice is due.  Meet soon.\r\n'
    job = job_for(args, text=text)
    runner.validate_args(args)
    metadata = r.fingerprint(job, text, provision, args, r.request_settings(args))
    client = Mock()
    client.chat.completions.create.return_value = response(
        '{"extractions": [{"start_sentence": "S2", "end_sentence": "S3"}]}')
    assert r.read_source(job.input_path) == text
    result = r.process_document(job, text, metadata, provision, args, client)
    assert result.extraction_count == 1
    manifest = json.loads(r.manifest_path(job).read_text())
    assert manifest['status'] == 'completed' and manifest['chonkie_version'] == '1.7.0'
    assert len(manifest['sentence_map']) == 3
    assert {s['pass_number'] for s in manifest['resolution'][0]['selections']} == {1, 2}
    record = json.loads(job.output_path.read_text())
    assert record['extraction_text'] == 'Notice is due.  Meet soon.'
    assert r.cache_matches(job, metadata, False)
    assert not r.cache_matches(job, metadata, True)
    for changed in (replace(provision, clause_description='Different'), replace(provision, max_char_buffer=10)):
        with pytest.raises(ValueError, match='mismatch'):
            r.cache_matches(job, r.fingerprint(job, text, changed, args, r.request_settings(args)), False)
    with pytest.raises(ValueError, match='mismatch'):
        r.cache_matches(job, r.fingerprint(job, text + 'changed', provision, args, r.request_settings(args)), False)
    job.output_path.write_text('broken')
    assert not r.cache_matches(job, metadata, False)
    client.chat.completions.create.side_effect = RuntimeError('request failed')
    with pytest.raises(RuntimeError):
        r.process_document(job, text, metadata, provision, args, client)
    assert json.loads(r.manifest_path(job).read_text())['status'] == 'failed'
    assert not r.cache_matches(job, metadata, False)
    client.chat.completions.create.side_effect = None
    client.chat.completions.create.return_value = response()
    r.process_document(job, text, metadata, provision, args, client)
    assert job.output_path.read_bytes() == b'' and r.cache_matches(job, metadata, False)
    r.manifest_path(job).unlink()
    assert not r.cache_matches(job, metadata, False)
    r.manifest_path(job).write_text('{partial')
    assert not r.cache_matches(job, metadata, False)


def test_incomplete_publication_does_not_reuse_old_completion(args, provision):
    job = job_for(args)
    runner.validate_args(args)
    text = r.read_source(job.input_path)
    metadata = r.fingerprint(job, text, provision, args, r.request_settings(args))
    client = Mock()
    client.chat.completions.create.return_value = response()
    original = r.save_manifest
    statuses = []
    def interrupted_save(job, manifest):
        statuses.append(manifest['status'])
        if manifest['status'] == 'completed':
            raise OSError('interrupted publish')
        original(job, manifest)
    with patch.object(r, 'save_manifest', side_effect=interrupted_save):
        with pytest.raises(OSError):
            r.process_document(job, text, metadata, provision, args, client)
    assert statuses == ['running', 'completed', 'failed']
    assert job.output_path.exists() and not r.cache_matches(job, metadata, False)
    with patch.object(r.os, 'replace', side_effect=OSError('interrupted replace')):
        with pytest.raises(OSError):
            r.atomic_write(job.output_path, b'new')
    assert not list(job.output_path.parent.glob('.*'))


def test_empty_document_completes_without_requests(args, provision):
    job = job_for(args, text=' \r\n')
    client = Mock()
    result = r.process_document(job, ' \r\n', {'request_settings': {}}, provision, args, client)
    assert result.extraction_count == 0 and job.output_path.read_bytes() == b''
    client.chat.completions.create.assert_not_called()


def test_passes_sequential_concurrency_and_deterministic_resolution(args, provision):
    text = ' '.join(f'Notice number {i} applies.' for i in range(22))
    job = job_for(args, text=text)
    lock = threading.Lock()
    active, peak = 0, 0
    observed = []
    def select(client, settings, spec, text, units, chunk, logs):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
            observed.append(chunk.pass_number)
        time.sleep(0.005 * (3 - chunk.primary[0] % 3))
        with lock:
            active -= 1
        return [selection(*chunk.primary, chunk.pass_number, chunk.chunk_id)]
    for workers, batch in [(2, 3), (4, 1)]:
        active, peak = 0, 0
        observed.clear()
        spec = replace(provision, langextract_max_workers=workers, langextract_batch_length=batch)
        with patch.object(r, 'select_ranges', side_effect=select):
            r.process_document(job, text, {'request_settings': {}}, spec, args, Mock())
        assert 1 <= peak <= min(workers, batch)
        assert observed == sorted(observed) and set(observed) == {1, 2}
        records = [json.loads(line) for line in job.output_path.read_text().splitlines()]
        assert [x['span_start'] for x in records] == sorted(x['span_start'] for x in records)


@pytest.mark.parametrize('endpoint', ['vllm', 'openrouter'])
def test_endpoint_routing_cache_and_owned_cleanup(args, provision, endpoint):
    args.endpoint = endpoint
    input_file(args)
    client = Mock()
    client.chat.completions.create.return_value = response()
    with (patch.object(r, 'load_provision', return_value=provision), patch.object(r, 'VLLMServer') as server,
          patch('openai.OpenAI', return_value=client) as factory,
          patch.dict('os.environ', {'OPENROUTER_API_KEY': 'mock-key'})):
        results = r.run(args)
        assert [x.status for x in results] == ['completed']
        expected_host = 'openrouter.ai' if endpoint == 'openrouter' else 'localhost'
        assert expected_host in factory.call_args.kwargs['base_url']
        assert factory.call_args.kwargs['max_retries'] == 0
        sent = client.chat.completions.create.call_args.kwargs['extra_body']
        if endpoint == 'vllm':
            assert sent['chat_template_kwargs']['enable_thinking'] is False
            options = server.call_args.kwargs['extra_serve_args']
            template = json.loads(options[options.index('--default-chat-template-kwargs') + 1])
            assert template['enable_thinking'] is False
        else:
            assert sent['reasoning'] == {'enabled': False}
        server.return_value.start.assert_called_once()
        server.return_value.close.assert_called_once()
        client.close.assert_called_once()
        server.reset_mock()
        assert [x.status for x in r.run(args)] == ['skipped']
        server.assert_not_called()
        args.max_tokens += 1
        assert [x.status for x in r.run(args)] == ['failed']
        server.assert_not_called()
        args.force = True
        assert [x.status for x in r.run(args)] == ['completed']
        server.return_value.start.assert_called_once()


def test_document_failure_continues_and_is_retried(args, provision):
    input_file(args, 'a_bad', 'Bad input.')
    input_file(args, 'b_good', 'Good input.')
    client = Mock()
    def complete(**kwargs):
        if 'Bad input.' in kwargs['messages'][1]['content']:
            raise RuntimeError('request rejected')
        return response()
    client.chat.completions.create.side_effect = complete
    with (patch.object(r, 'load_provision', return_value=provision), patch.object(r, 'VLLMServer'),
          patch('openai.OpenAI', return_value=client)):
        assert [x.status for x in r.run(args)] == ['failed', 'completed']
        client.chat.completions.create.side_effect = None
        client.chat.completions.create.return_value = response()
        assert [x.status for x in r.run(args)] == ['completed', 'skipped']
    with patch.object(r, 'run', return_value=[runner.ExtractionResult(job_for(args), 'failed')]):
        assert r.main(['--provision', 'technology']) == 1


@pytest.mark.parametrize('failure', [RuntimeError('queue failed'), KeyboardInterrupt()])
@pytest.mark.parametrize('borrowed', [False, True])
def test_server_ownership_and_cleanup_on_exception(args, provision, failure, borrowed):
    input_file(args)
    runner.validate_args(args)
    server = SimpleNamespace(**{k: getattr(args, k) for k in (
        'endpoint', 'model_name', 'port', 'max_model_len', 'num_gpus', 'gpu_memory_utilization', 'max_num_seqs')},
        extra_serve_args=['--generation-config', 'vllm'], start=Mock(), close=Mock())
    with (patch.object(r, 'load_provision', return_value=provision), patch.object(r, 'VLLMServer', return_value=server) as factory,
          patch('openai.OpenAI') as client, patch.object(runner, 'run_extraction_queue', side_effect=failure)):
        with pytest.raises(type(failure)):
            r.run(args, server=server if borrowed else None)
        client.return_value.close.assert_called_once()
        if borrowed:
            factory.assert_not_called()
            server.start.assert_not_called()
            server.close.assert_not_called()
        else:
            server.start.assert_called_once()
            server.close.assert_called_once()


def test_start_failure_closes_owned_server(args):
    input_file(args)
    with patch.object(r, 'VLLMServer') as server:
        server.return_value.start.side_effect = RuntimeError('startup failed')
        with pytest.raises(RuntimeError):
            r.run(args)
        server.return_value.close.assert_called_once()


def test_discovery_cuad_override_filter_and_seed(args, tmp_path, provision):
    from references.cuad.compare_extractions import DOCUMENT_ID_TITLE_OVERRIDES
    alias, title = next(iter(DOCUMENT_ID_TITLE_OVERRIDES.items()))
    args.source, args.cuad_test = 'cuad', True
    for source, doc in [('cuad', alias), ('cuad', 'test_doc'), ('cuad', 'train_doc'), ('other', 'test_doc')]:
        input_file(args, doc, source=source)
    split = tmp_path / 'test.json'
    split.write_text(json.dumps({'data': [
        {'title': title, 'paragraphs': [{'qas': []}]},
        {'title': 'test_doc', 'paragraphs': [{'qas': []}]}]}))
    client = Mock()
    client.chat.completions.create.return_value = response()
    with (patch.object(runner, 'CUAD_TEST_JSON', split), patch.object(r, 'load_provision', return_value=provision),
          patch.object(r, 'VLLMServer'), patch('openai.OpenAI', return_value=client)):
        results = r.run(args)
        assert {x.job.document_id for x in results} == {alias, 'test_doc'}
        args.sample, args.seed, args.force = 1, 42, True
        first = r.run(args)[0].job
        assert r.run(args)[0].job == first
        args.sample, args.document_id = None, ['test_doc']
        assert [x.job.document_id for x in r.run(args)] == ['test_doc']


def test_cli_defaults_and_model_generation_overrides():
    args = r.parse_args(['--provision', 'technology'])
    assert args.endpoint == 'vllm' and args.max_tokens == 5000
    assert args.output_root.name == 'stg_02_extract_runner2'
    assert runner.parse_args(['--provision', 'technology']).output_root.name == 'stg_02_extract'
    assert r.parse_args(['--provision', 'governing_law', '--cuad_test']).source == 'cuad'
    args.harness_request_defaults = {'temperature': 0.7, 'top_p': 0.8, 'max_tokens': 900,
                                     'extra_body': {'chat_template_kwargs': {'enable_thinking': True}}}
    settings = r.request_settings(args)
    assert settings['temperature'] == 0 and settings['max_tokens'] == 5000 and settings['top_p'] == 0.8
    args.model_name = 'Qwen/Qwen3.8-Flash'
    assert r.request_settings(args)['temperature'] == 1.0
    assert r.request_settings(args)['extra_body']['chat_template_kwargs']['enable_thinking'] is False
    with pytest.raises(SystemExit):
        r.parse_args(['--provision', 'technology', '--max-tokens', '0'])


@pytest.mark.parametrize('model', MODELS)
def test_thinking_disabled_for_all_sweep_models_despite_shared_defaults(model):
    args = r.parse_args(['--provision', 'technology', '--model-name', model])
    defaults = {'top_p': .9, 'extra_body': {'top_k': 20, 'chat_template_kwargs': {
        'enable_thinking': True, 'preserve_thinking': True}}}
    args.harness_request_defaults = defaults
    settings = r.request_settings(args)
    assert settings['extra_body']['chat_template_kwargs'] == {
        'enable_thinking': False, 'preserve_thinking': False}
    assert settings['top_p'] == .9 and settings['extra_body']['top_k'] == 20
    # Shared inputs remain usable by the original Harness with thinking enabled.
    assert defaults['extra_body']['chat_template_kwargs'] == {
        'enable_thinking': True, 'preserve_thinking': True}
    options = r.serving_args(args)
    template = json.loads(options[options.index('--default-chat-template-kwargs') + 1])
    assert template['enable_thinking'] is False
    original = runner.reasoning_serve_args(model, None)
    assert json.loads(original[1])['enable_thinking'] is True


def test_openrouter_model_profile_cannot_reenable_reasoning():
    args = r.parse_args(['--provision', 'technology', '--model-name', 'Qwen/Qwen3.8-Flash',
                         '--endpoint', 'openrouter'])
    args.harness_request_defaults = {'extra_body': {'reasoning': {'enabled': True, 'effort': 'high'}}}
    settings = r.request_settings(args)
    assert settings['extra_body']['reasoning'] == {'enabled': False}
    assert settings['extra_body']['provider']['only'] == ['alibaba']
    assert args.harness_request_defaults['extra_body']['reasoning']['enabled'] is True


def test_thinking_setting_invalidates_existing_cache(args, provision):
    job = job_for(args)
    runner.validate_args(args)
    text = r.read_source(job.input_path)
    settings = r.request_settings(args)
    current = r.fingerprint(job, text, provision, args, settings)
    settings['extra_body']['chat_template_kwargs']['enable_thinking'] = True
    previous = r.fingerprint(job, text, provision, args, settings)
    r.save_manifest(job, {**previous, 'status': 'completed'})
    with pytest.raises(ValueError, match='mismatch'):
        r.cache_matches(job, current, False)
    assert r.cache_matches(job, current, True) is False


def test_sentence_endings_crlf_and_tabs_preserve_intervening_whitespace(args):
    text = 'First notice.\r\nSecond notice!\tThird notice?\rFinal notice.'
    units = r.build_sentence_map(text)
    assert [text[u.start:u.end] for u in units] == ['First notice.', 'Second notice!', 'Third notice?', 'Final notice.']
    record = r.make_records(job_for(args, text=text), text, units, [selection(1, 4)])[0]
    assert record['extraction_text'] == text


def test_scalar_budget_repeats_and_context_is_selectable(args, provision):
    text = ' '.join(f'Notice number {i} applies.' for i in range(12))
    job = job_for(args, text=text)
    spec = replace(provision, max_char_buffer=40)
    with patch.object(r, 'build_chunks', wraps=r.build_chunks) as chunker:
        client = Mock()
        client.chat.completions.create.return_value = response()
        r.process_document(job, text, {'request_settings': {}}, spec, args, client)
    assert [(call.args[2], call.args[3]) for call in chunker.call_args_list] == [(40, 1), (40, 2)]
    chunk = r.Chunk(1, 'P1C1', (9, 10), (1, 8, 9, 10, 11))
    assert r.parse_ranges('{"extractions": [{"start_sentence":"S8","end_sentence":"S11"}]}', chunk) == [selection(8, 11)]


def test_duplicate_proposed_boundaries_preserve_primary_coverage():
    text = 'First sentence. Second sentence. Third sentence.'
    units = r.build_sentence_map(text)
    proposals = [SimpleNamespace(end_index=i) for i in (3, 4, 5, 22, 23, len(text))]
    with patch('chonkie.RecursiveChunker.chunk', return_value=proposals):
        chunks = r.build_chunks(text, units, 5, 1)
    assert [c.primary for c in chunks] == [(1, 1), (2, 2), (3, 3)]


def test_discovery_full_real_test_split_and_overrides(args):
    from references.cuad.compare_extractions import DOCUMENT_ID_TITLE_OVERRIDES, load_gold_documents
    titles = load_gold_documents(runner.CUAD_TEST_JSON)
    reverse = {title: alias for alias, title in DOCUMENT_ID_TITLE_OVERRIDES.items()}
    expected = {reverse.get(title, title.strip()) for title in titles}
    assert len(expected) == 102
    for doc in expected | {'outside_test'}:
        input_file(args, doc, source='cuad')
    jobs = runner.discover_full_texts(args.input_root, args.output_root, args.ocr_model_name,
                                     args.model_name, args.provision)
    kept = runner.filter_cuad_test_jobs(jobs)
    assert {j.document_id for j in kept} == expected
    assert all(j.output_path.parent.name == j.document_id for j in kept)


def test_borrowed_success_settings_and_actual_serving_manifest(args, provision):
    input_file(args)
    runner.validate_args(args)
    standalone = r.request_settings(args)
    server = SimpleNamespace(**{k: getattr(args, k) for k in (
        'endpoint', 'model_name', 'port', 'max_model_len', 'num_gpus', 'gpu_memory_utilization', 'max_num_seqs')},
        extra_serve_args=['--generation-config', 'vllm'], start=Mock(), close=Mock())
    args.harness_request_defaults = {'top_p': 0.75, 'max_tokens': 9000}
    client = Mock()
    client.chat.completions.create.return_value = response()
    with (patch.object(r, 'load_provision', return_value=provision), patch.object(r, 'VLLMServer') as factory,
          patch('openai.OpenAI', return_value=client)):
        results = r.run(args, server=server)
    assert [x.status for x in results] == ['completed']
    factory.assert_not_called()
    server.start.assert_not_called()
    server.close.assert_not_called()
    sent = client.chat.completions.create.call_args.kwargs
    for key in ('temperature', 'max_tokens', 'extra_body'):
        assert sent[key] == standalone[key]
    assert sent['top_p'] == 0.75
    manifest = json.loads(r.manifest_path(results[0].job).read_text())
    assert manifest['serving']['extra_serve_args'] == ['--generation-config', 'vllm']
    server.model_name = 'another/model'
    with pytest.raises(ValueError, match='Borrowed server'):
        r.run(args, server=server)
