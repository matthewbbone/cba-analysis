import json
import signal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from pipeline.stg_02_extract import orchestrator as orch
from pipeline.stg_02_extract.orchestrator import execute_job
from pipeline.stg_02_extract import runner, contracteval as ce
from references.cuad.compare_extractions import CATEGORY_BY_CLAUSE_TYPE, DOCUMENT_ID_TITLE_OVERRIDES


def read_report(root):
    return json.loads((root / 'out/cuad/orchestration/latest.json').read_text())


@pytest.fixture
def sweep(tmp_path, monkeypatch):
    clauses = sorted(CATEGORY_BY_CLAUSE_TYPE)
    documents = [f'doc{i}' for i in range(102)]
    monkeypatch.setattr(orch, 'preflight', lambda args: (clauses, documents))
    defaults = Mock(return_value={'extra_body': {'chat_template_kwargs': {'enable_thinking': True}}})
    monkeypatch.setattr(orch, 'load_generation_defaults', defaults)
    events, servers = [], []

    class FakeServer:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.server = Mock()
            self.server.poll.return_value = None
            self.start = Mock(side_effect=lambda: events.append(('start', self.model_name)))
            self.close = Mock(side_effect=lambda: events.append(('close', self.model_name)))
            servers.append(self)

    factory = Mock(side_effect=FakeServer)
    monkeypatch.setattr(orch, 'VLLMServer', factory)

    def execute(args, task, document_ids, server, request_defaults):
        assert document_ids == documents
        assert server is servers[-1]
        events.append((task['method'], task['model_name'], task['clause_type']))
        return {'status': 'completed', 'counts': {'completed': 102, 'reused': 0, 'failed': 0}, 'errors': {}}

    executor = Mock(side_effect=execute)
    monkeypatch.setattr(orch, 'execute_job', executor)
    return SimpleNamespace(root=tmp_path, args=['--output-root', str(tmp_path / 'out'), '--no-progress'],
                           clauses=clauses, documents=documents, servers=servers, events=events,
                           factory=factory, defaults=defaults, executor=executor, execute=execute)


@pytest.mark.parametrize('device,expected', [('0', '0'), ('cuda:0', '0'), ('cuda:2', '2')])
def test_device_alias(device, expected):
    assert orch.parse_args(['--device', device]).device == expected


def test_target_defaults_and_validation():
    assert orch.parse_args([]).target == 'all'
    assert orch.parse_args([]).method == 'both'
    assert orch.parse_args(['--target', 'min_5']).target == 'min_5'
    assert orch.parse_args(['--model-name', orch.MODELS[2]]).model_name == [orch.MODELS[2]]
    assert orch.parse_args(['--model-name', orch.MODELS[0], '--model-name', orch.MODELS[2]]).model_name == [orch.MODELS[0], orch.MODELS[2]]
    assert orch.parse_args(['--model-name', 'thomsonreuters/Thomson-1.0-Small']).model_name == ['thomsonreuters/Thomson-1.0-Small']
    with pytest.raises(SystemExit):
        orch.parse_args(['--target', 'unknown'])
    with pytest.raises(SystemExit):
        orch.parse_args(['--method', 'unknown'])


@pytest.mark.parametrize('args', [
    ['--device', '0,1'], ['--device', 'cuda:0,1'], ['--device', '-1'],
    ['--port', '0'], ['--max-model-len', '0'], ['--max-num-seqs', '0'],
    ['--gpu-memory-utilization', '1.1'],
])
def test_invalid_settings(args):
    with pytest.raises(SystemExit):
        orch.parse_args(args)


def test_full_schedule_single_server_per_model(sweep):
    old_handler = signal.getsignal(signal.SIGTERM)
    assert orch.main(sweep.args) == 0
    expected = []
    for model in orch.MODELS:
        expected.append(('start', model))
        for clause in sweep.clauses:
            expected.extend([('ContractEval', model, clause), ('Harness', model, clause)])
        expected.append(('close', model))
    assert sweep.events == expected
    assert sweep.executor.call_count == 410
    assert [call.args[0] for call in sweep.defaults.call_args_list] == list(orch.MODELS)
    for server in sweep.servers:
        server.start.assert_called_once()
        server.close.assert_called_once()
        assert server.num_gpus == 1 and server.device == '0'
        assert server.max_model_len == 131072
        assert server.extra_serve_args[:2] == ['--generation-config', 'vllm']
        assert '--default-chat-template-kwargs' not in server.extra_serve_args
    assert sweep.servers[0].extra_serve_args[-1] == 'qwen3'
    assert sweep.servers[1].extra_serve_args[-1] == 'gemma4'
    assert sweep.servers[-1].extra_serve_args == ['--generation-config', 'vllm']
    report = read_report(sweep.root)
    assert report['status'] == 'completed'
    assert len(report['jobs']) == 410
    assert all(task['counts']['completed'] == 102 for task in report['jobs'])
    assert signal.getsignal(signal.SIGTERM) == old_handler


def test_dry_run_has_no_model_or_output_side_effects(sweep, capsys):
    assert orch.main(sweep.args + ['--dry-run']) == 0
    sweep.factory.assert_not_called()
    sweep.defaults.assert_not_called()
    sweep.executor.assert_not_called()
    assert not (sweep.root / 'out').exists()
    assert '410 sequential jobs' in capsys.readouterr().out


def test_single_model_filter(sweep):
    model = orch.MODELS[2]
    assert orch.main(sweep.args + ['--model-name', model]) == 0
    assert len(sweep.servers) == 1
    assert sweep.servers[0].model_name == model
    assert sweep.executor.call_count == 82
    sweep.defaults.assert_called_once_with(model)
    report = read_report(sweep.root)
    assert {task['model_name'] for task in report['jobs']} == {model}


@pytest.mark.parametrize('method,label', [('runner', 'Harness'), ('contracteval', 'ContractEval')])
def test_single_method_runs_only_selected_module(sweep, monkeypatch, method, label):
    # Restore the real dispatcher so this checks which extraction module runs.
    monkeypatch.setattr(orch, 'execute_job', execute_job)
    runner_run = Mock(return_value=[
        runner.ExtractionResult(SimpleNamespace(document_id=doc), 'completed')
        for doc in sweep.documents
    ])
    contracteval_run = Mock(return_value={
        'failures': {}, 'missing_inputs': [], 'missing_outputs': [],
        'execution': {'completed': 102, 'reused': 0, 'failed': 0},
        'metrics': {'count': 102},
    })
    monkeypatch.setattr(runner, 'run', runner_run)
    monkeypatch.setattr(ce, 'run', contracteval_run)
    models = [orch.MODELS[0], 'thomsonreuters/Thomson-1.0-Small']
    argv = sweep.args + ['--method', method, '--force']
    for model in models:
        argv.extend(['--model-name', model])
    assert orch.main(argv) == 0
    selected, excluded = (runner_run, contracteval_run) if method == 'runner' else (contracteval_run, runner_run)
    assert selected.call_count == 82
    excluded.assert_not_called()
    assert [server.model_name for server in sweep.servers] == models
    for call in selected.call_args_list:
        assert call.args[0].force
        assert call.kwargs['server'] in sweep.servers
    if method == 'contracteval':
        sweep.defaults.assert_not_called()
    report = read_report(sweep.root)
    assert report['settings']['method'] == method
    assert {task['method'] for task in report['jobs']} == {label}
    assert len(report['jobs']) == 82


def test_runner_only_dry_run(sweep, capsys):
    assert orch.main(sweep.args + ['--method', 'runner', '--dry-run']) == 0
    output = capsys.readouterr().out
    assert '205 sequential jobs' in output
    assert 'ContractEval' not in output
    sweep.factory.assert_not_called()
    sweep.executor.assert_not_called()


def test_clause_failures_continue_and_progress_is_saved(sweep):
    def execute(*args):
        if sweep.executor.call_count == 1:
            raise ValueError('Cached input/prompt/settings mismatch')
        report = read_report(sweep.root)
        assert report['jobs'][0]['error'] == 'Cached input/prompt/settings mismatch'
        return sweep.execute(*args)

    sweep.executor.side_effect = execute
    assert orch.main(sweep.args) == 1
    report = read_report(sweep.root)
    assert report['jobs'][0]['status'] == 'failed'
    assert report['jobs'][-1]['status'] == 'completed'
    assert sweep.executor.call_count == 410


def test_startup_failure_skips_model_and_continues(sweep):
    create = sweep.factory.side_effect

    def factory(**kwargs):
        server = create(**kwargs)
        if len(sweep.servers) == 1:
            server.start.side_effect = RuntimeError('out of memory')
        return server

    sweep.factory.side_effect = factory
    assert orch.main(sweep.args) == 1
    report = read_report(sweep.root)
    assert all(task['status'] == 'unavailable' for task in report['jobs'][:82])
    assert report['jobs'][82]['status'] == 'completed'
    assert sweep.executor.call_count == 328
    sweep.servers[0].close.assert_called_once()


def test_dead_server_is_not_restarted(sweep):
    def execute(*args):
        result = sweep.execute(*args)
        if sweep.executor.call_count == 1:
            sweep.servers[0].server.poll.return_value = 1
        return result

    sweep.executor.side_effect = execute
    assert orch.main(sweep.args) == 1
    report = read_report(sweep.root)
    assert report['jobs'][0]['status'] == 'completed'
    assert all(task['status'] == 'unavailable' for task in report['jobs'][1:82])
    assert len(sweep.servers) == 5
    sweep.servers[0].start.assert_called_once()


@pytest.mark.parametrize('sigterm', [False, True])
def test_interruption_closes_server_and_stops_sweep(sweep, sigterm):
    def interrupt(*args):
        if sigterm:
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        raise KeyboardInterrupt

    sweep.executor.side_effect = interrupt
    assert orch.main(sweep.args) == 130
    assert len(sweep.servers) == 1
    sweep.servers[0].close.assert_called_once()
    assert read_report(sweep.root)['status'] == 'interrupted'
    assert all(task['status'] == 'interrupted' for task in read_report(sweep.root)['jobs'])


def test_cleanup_failure_does_not_load_another_model(sweep):
    create = sweep.factory.side_effect

    def factory(**kwargs):
        server = create(**kwargs)
        server.close.side_effect = RuntimeError('cannot terminate')
        return server

    sweep.factory.side_effect = factory
    with pytest.raises(RuntimeError, match='Unable to close'):
        orch.main(sweep.args)
    assert len(sweep.servers) == 1
    assert read_report(sweep.root)['status'] == 'failed'


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    alias, title = next(iter(DOCUMENT_ID_TITLE_OVERRIDES.items()))
    pairs = [(alias, title)] + [(f'doc{i:03}', f' doc{i:03} ') for i in range(1, 102)]
    entries = []
    for doc, title in pairs:
        full = tmp_path / 'input/cuad/ocr' / doc / 'full.txt'
        full.parent.mkdir(parents=True)
        full.write_text('contract text')
        entries.append({'title': title, 'paragraphs': [{'qas': [
            {'id': title + '__' + category, 'answers': []}
            for category in CATEGORY_BY_CLAUSE_TYPE.values()
        ]}]})
    test_json = tmp_path / 'test.json'
    test_json.write_text(json.dumps({'data': entries}))
    monkeypatch.setattr(ce, 'TEST_JSON', test_json)
    args = orch.parse_args(['--input-root', str(tmp_path / 'input'), '--ocr-model-name', 'ocr'])
    return SimpleNamespace(args=args, root=tmp_path, pairs=pairs)


def test_preflight_full_split_title_overrides_and_training_exclusion(inputs):
    full = inputs.root / 'input/cuad/ocr/training/full.txt'
    full.parent.mkdir(parents=True)
    full.write_text('unused training text')
    clauses, ids = orch.preflight(inputs.args)
    assert clauses == sorted(CATEGORY_BY_CLAUSE_TYPE)
    assert ids == sorted(doc for doc, _ in inputs.pairs)


def test_min_5_preflight_and_schedule(inputs, monkeypatch, capsys):
    expected = {'change_of_control', 'audit_rights', 'post_termination_services',
                'rofr_rofo_rofn', 'anti_assignment'}
    args = ['--target', 'min_5', '--input-root', str(inputs.root / 'input'),
            '--ocr-model-name', 'ocr', '--output-root', str(inputs.root / 'out')]
    clauses, ids = orch.preflight(orch.parse_args(args))
    assert clauses == sorted(expected)
    assert len(ids) == 102
    factory = Mock()
    monkeypatch.setattr(orch, 'VLLMServer', factory)
    assert orch.main(args + ['--dry-run']) == 0
    factory.assert_not_called()
    assert not (inputs.root / 'out').exists()
    lines = capsys.readouterr().out.splitlines()
    assert '50 sequential jobs' in lines[0]
    assert len(lines[1:]) == 50
    assert {line.split(' / ')[1] for line in lines[1:]} == expected
    assert [line.split(' / ')[-1] for line in lines[1:]] == ['ContractEval', 'Harness'] * 25


@pytest.mark.parametrize('problem', ['missing', 'unreadable', 'duplicate'])
def test_preflight_errors_are_explicit(inputs, problem):
    full = inputs.root / 'input/cuad/ocr/doc001/full.txt'
    if problem == 'missing':
        full.unlink()
        expected = 'Missing OCR input:  doc001 '
    elif problem == 'unreadable':
        full.write_bytes(b'\xff')
        expected = 'Unreadable input'
    else:
        duplicate = inputs.root / 'input/cuad/ocr/ doc001 /full.txt'
        duplicate.parent.mkdir(parents=True)
        duplicate.write_text('duplicate')
        expected = 'Multiple cached document IDs'
    with pytest.raises(ValueError, match=expected):
        orch.preflight(inputs.args)


def test_clause_settings_and_output_paths(tmp_path):
    args = orch.parse_args(['--output-root', str(tmp_path), '--device', 'cuda:0', '--force',
                           '--gpu-memory-utilization', '.9', '--max-model-len', '65536', '--no-progress'])
    for method in orch.METHODS:
        selected = orch.clause_args(args, orch.MODELS[0], 'governing_law', method, ['doc'])
        assert selected.document_id == ['doc']
        assert selected.endpoint == 'vllm' and selected.num_gpus == 1
        assert selected.device == '0' and selected.concurrency == 1
        assert selected.max_model_len == 65536 and selected.force and selected.no_progress
        assert selected.gpu_memory_utilization == .9
        if method == 'ContractEval':
            assert selected.output_root == tmp_path / 'cuad/contracteval'
            assert selected.temperature == 0 and selected.max_tokens == 5000
            assert selected.thinking == 'default'
            assert 'extra_body' not in ce.inference_settings(selected)['request']
        else:
            assert selected.output_root == tmp_path and selected.cuad_test


def test_generation_config_loader_uses_only_nondefault_values(monkeypatch):
    import sys
    config = Mock()
    config.to_diff_dict.return_value = {'temperature': .7, 'top_k': 30}
    loader = Mock(return_value=config)
    monkeypatch.setitem(sys.modules, 'vllm.transformers_utils.config',
                        SimpleNamespace(try_get_generation_config=loader))
    settings = orch.load_generation_defaults(orch.MODELS[0])
    loader.assert_called_once_with(orch.MODELS[0], trust_remote_code=True)
    config.to_diff_dict.assert_called_once()
    assert settings['temperature'] == .7 and settings['extra_body']['top_k'] == 30
    loader.return_value = None
    settings = orch.load_generation_defaults(orch.MODELS[0])
    assert settings == {'extra_body': {'chat_template_kwargs': {
        'enable_thinking': True, 'preserve_thinking': False,
    }}}


def test_execute_job_reports_document_failures_and_borrows_server(tmp_path, monkeypatch):
    args = orch.parse_args(['--output-root', str(tmp_path)])
    task = {'model_name': orch.MODELS[0], 'clause_type': 'governing_law', 'method': 'Harness'}
    borrowed = object()
    defaults = {'temperature': .8}
    fake = Mock(return_value=[
        runner.ExtractionResult(SimpleNamespace(document_id='doc1'), 'skipped'),
        runner.ExtractionResult(SimpleNamespace(document_id='doc2'), 'failed', error='timeout'),
    ])
    monkeypatch.setattr(runner, 'run', fake)
    result = orch.execute_job(args, task, ['doc1', 'doc2'], borrowed, defaults)
    assert result['status'] == 'failed' and result['errors'] == {'doc2': 'timeout'}
    assert result['counts'] == {'completed': 0, 'reused': 1, 'failed': 1}
    assert fake.call_args.kwargs['server'] is borrowed
    assert fake.call_args.args[0].harness_request_defaults == defaults
    task['method'] = 'ContractEval'
    fake = Mock(return_value={'failures': {'doc2': 'timeout'}, 'missing_inputs': [],
                             'missing_outputs': [], 'execution': result['counts'], 'metrics': {'count': 2}})
    monkeypatch.setattr(ce, 'run', fake)
    assert orch.execute_job(args, task, ['doc1', 'doc2'], borrowed, defaults)['status'] == 'failed'
    assert fake.call_args.kwargs['server'] is borrowed
    assert not hasattr(fake.call_args.args[0], 'harness_request_defaults')


def test_harness_borrowed_server_cache_retry_and_ownership(tmp_path, monkeypatch):
    full = tmp_path / 'input/cuad/ocr/doc/full.txt'
    full.parent.mkdir(parents=True)
    full.write_text('New York')
    args = runner.parse_args([
        '--provision', 'governing_law', '--source', 'cuad', '--model-name', 'test/model',
        '--input-root', str(tmp_path / 'input'), '--output-root', str(tmp_path / 'out'),
        '--ocr-model-name', 'ocr', '--max-model-len', '131072', '--no-progress',
    ])
    args.harness_request_defaults = {'temperature': .7}
    server = Mock(endpoint='vllm', model_name='test/model', port=8123, max_model_len=131072)
    factory = Mock()
    monkeypatch.setattr(runner, 'VLLMServer', factory)
    extractor = Mock(side_effect=RuntimeError('request failed'))
    extractor_factory = Mock(return_value=extractor)
    monkeypatch.setattr(runner, 'make_langextract_extractor', extractor_factory)
    result = runner.run(args, server=server)
    assert result[0].status == 'failed'
    assert not result[0].job.output_path.exists()
    extractor.side_effect = None
    extractor.return_value = []
    result = runner.run(args, server=server)
    assert result[0].status == 'completed'
    assert result[0].job.output_path == tmp_path / 'out/cuad/test_model/doc/governing_law.jsonl'
    assert extractor_factory.call_args.kwargs['request_defaults'] == {'temperature': .7}
    assert runner.run(args, server=server)[0].status == 'skipped'
    assert extractor.call_count == 2
    args.force = True
    assert runner.run(args, server=server)[0].status == 'completed'
    extractor_factory.side_effect = ValueError('cannot create extractor')
    with pytest.raises(ValueError, match='cannot create extractor'):
        runner.run(args, server=server)
    factory.assert_not_called()
    server.start.assert_not_called()
    server.close.assert_not_called()


def test_interrupted_queue_cancels_pending_documents(monkeypatch):
    from threading import Event, Timer

    started, release, finished = Event(), Event(), Event()
    calls = []
    jobs = [SimpleNamespace(document_id=str(i)) for i in range(3)]

    def process(job):
        calls.append(job.document_id)
        started.set()
        release.wait(5)
        finished.set()
        return runner.ExtractionResult(job, 'completed')

    def interrupt(futures):
        assert started.wait(1)
        raise KeyboardInterrupt

    monkeypatch.setattr(runner, 'as_completed', interrupt)
    # A fallback keeps a regression from hanging the test suite.
    timer = Timer(2, release.set)
    timer.start()
    try:
        with pytest.raises(KeyboardInterrupt):
            runner.run_extraction_queue(jobs, 1, process)
        assert not release.is_set()
        assert calls == ['0']
    finally:
        release.set()
        timer.cancel()
        assert finished.wait(2)
