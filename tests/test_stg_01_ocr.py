import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from pipeline.stg_01_ocr import common
from pipeline.stg_01_ocr.common import (
    DocumentJob,
    PageJob,
    PageResult,
    ProgressReporter,
    PROJECT_ROOT,
    build_page_jobs,
    combine_document_pages,
    default_input_root,
    default_stage_output_root,
    discover_documents,
    path_safe_model_name,
    resolve_project_path,
    process_page_job,
    run_page_queue,
    write_page_markdown,
)
from pipeline.stg_01_ocr.general import runner as general
from pipeline.stg_01_ocr.render import render_page_isolated


class OcrDiscoveryTests(unittest.TestCase):
    def test_discovers_source_pdfs_and_ignores_output_root(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "cache"
            output_root = root / "stg_01_ocr"
            model_name = "org/model"
            (root / "source_a").mkdir(parents=True)
            (root / "source_b").mkdir()
            output_root.mkdir()
            (root / "source_a" / "doc_1.pdf").write_bytes(b"%PDF")
            (root / "source_a" / ".hidden.pdf").write_bytes(b"%PDF")
            (root / "source_a" / "notes.txt").write_text("ignore", encoding="utf-8")
            (root / "source_b" / "doc_2.pdf").write_bytes(b"%PDF")
            (output_root / "full.pdf").write_bytes(b"%PDF")

            documents = discover_documents(
                root,
                output_root,
                model_name=model_name,
                source_filter="source_a",
            )

        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].source, "source_a")
        self.assertEqual(documents[0].document_id, "doc_1")
        self.assertEqual(
            documents[0].output_dir,
            output_root / "source_a" / "org_model" / "doc_1",
        )

    def test_discovers_a_selected_list_of_document_ids(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "cache"
            output_root = root / "stg_01_ocr"
            source_dir = root / "source_a"
            source_dir.mkdir(parents=True)
            for document_id in ("doc_1", "doc_2", "doc_3"):
                (source_dir / f"{document_id}.pdf").write_bytes(b"%PDF")

            documents = discover_documents(
                root,
                output_root,
                model_name="org/model",
                source_filter="source_a",
                document_id_filter=["doc_3", "doc_1"],
            )

        self.assertEqual(
            [document.document_id for document in documents],
            ["doc_1", "doc_3"],
        )

    def test_stage_output_defaults_to_cache_dir_and_stage_name(self) -> None:
        with patch.dict("os.environ", {"CACHE_DIR": "/tmp/cba-cache"}):
            self.assertEqual(default_stage_output_root(), Path("/tmp/cba-cache/stg_01_ocr"))

    def test_relative_cache_dir_resolves_from_project_root(self) -> None:
        with patch.dict("os.environ", {"CACHE_DIR": "relative-cache"}):
            self.assertEqual(default_input_root(), PROJECT_ROOT / "relative-cache")
            self.assertEqual(
                default_stage_output_root(),
                PROJECT_ROOT / "relative-cache" / "stg_01_ocr",
            )

    def test_resolve_project_path_keeps_absolute_paths(self) -> None:
        self.assertEqual(resolve_project_path("/tmp/cba-cache"), Path("/tmp/cba-cache"))

    def test_model_name_is_safe_for_single_path_segment(self) -> None:
        self.assertEqual(path_safe_model_name("AIDC-AI/Ovis2.6-30B-A3B"), "AIDC-AI_Ovis2.6-30B-A3B")

    def test_builds_page_jobs_for_all_pages_before_processing(self) -> None:
        document = DocumentJob(
            source="source",
            document_id="doc",
            pdf_path=Path("cache/source/doc.pdf"),
            output_dir=Path("cache/stg_01_ocr/source/model/doc"),
        )

        jobs, states = build_page_jobs([document], page_counter=lambda _: 3)

        self.assertEqual([job.page_number for job in jobs], [1, 2, 3])
        self.assertEqual(states[("source", "doc")].total_pages, 3)
        self.assertEqual(
            states[("source", "doc")].page_paths[2],
            Path("cache/stg_01_ocr/source/model/doc/page_2.txt"),
        )

    def test_isolated_renderer_reports_native_crash_as_python_error(self) -> None:
        completed = __import__("subprocess").CompletedProcess(
            args=["python"],
            returncode=-6,
            stdout="",
            stderr="*** stack smashing detected ***: terminated\n",
        )

        with patch("pipeline.stg_01_ocr.render.subprocess.run", return_value=completed):
            with self.assertRaisesRegex(RuntimeError, "stack smashing"):
                render_page_isolated(Path("doc.pdf"), 1, 200)


class OcrQueueTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_client_is_closed_on_the_operation_event_loop(self) -> None:
        events = []

        class FakeClient:
            async def close(self) -> None:
                events.append(("close", asyncio.get_running_loop()))

        async def operation() -> str:
            events.append(("operation", asyncio.get_running_loop()))
            return "done"

        result = await common._run_with_client_cleanup(operation(), FakeClient())

        self.assertEqual(result, "done")
        self.assertEqual([event[0] for event in events], ["operation", "close"])
        self.assertIs(events[0][1], events[1][1])

    async def test_async_client_is_closed_when_the_operation_fails(self) -> None:
        closed = False

        class FakeClient:
            async def close(self) -> None:
                nonlocal closed
                closed = True

        async def operation() -> None:
            raise RuntimeError("queue failed")

        with self.assertRaisesRegex(RuntimeError, "queue failed"):
            await common._run_with_client_cleanup(operation(), FakeClient())

        self.assertTrue(closed)

    async def test_process_page_job_skips_existing_output_without_client(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "page_1.txt"
            output_path.write_text("already done\n", encoding="utf-8")
            output_path.with_suffix(".md").write_text("already done\n", encoding="utf-8")
            job = PageJob(
                source="source",
                document_id="doc",
                pdf_path=Path("doc.pdf"),
                page_number=1,
                output_path=output_path,
            )

            result = await process_page_job(
                job=job,
                spec=general.SPEC,
                client=None,
                model_name=general.DEFAULT_MODEL_NAME,
                args=(args := general.parse_args([])),
                request_limiter=common.RequestLimiter(args.max_inflight_requests),
                repetition_policy=general.SPEC.repetition_policy(args),
            )

        self.assertEqual(result.status, "skipped")

    async def test_queue_records_success_skips_and_failures(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            document = DocumentJob(
                source="source",
                document_id="doc",
                pdf_path=Path(tmp_dir) / "doc.pdf",
                output_dir=Path(tmp_dir) / "out",
            )
            jobs, states = build_page_jobs([document], page_counter=lambda _: 3)
            jobs[1].output_path.parent.mkdir(parents=True)
            jobs[1].output_path.write_text("existing\n", encoding="utf-8")

            async def processor(job: PageJob) -> PageResult:
                await asyncio.sleep(0)
                if job.page_number == 1:
                    job.output_path.parent.mkdir(parents=True, exist_ok=True)
                    job.output_path.write_text("page one\n", encoding="utf-8")
                    return PageResult(
                        job=job,
                        status="completed",
                        expanded_merged_cells=True,
                        repetition_trimmed=True,
                    )
                if job.page_number == 2:
                    return PageResult(job=job, status="skipped")
                raise RuntimeError("bad page")

            await run_page_queue(jobs, states, concurrency=2, processor=processor)
            state = states[("source", "doc")]

        self.assertEqual(state.completed_pages, {1, 2})
        self.assertEqual(state.skipped_pages, {2})
        self.assertEqual(state.expanded_merged_cell_pages, {1})
        self.assertEqual(state.repetition_trimmed_pages, {1})
        self.assertEqual(set(state.failed_pages), {3})
        self.assertFalse(state.is_complete)

    def test_raw_writer_preserves_legacy_trailing_whitespace_contract(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "page_1.txt"
            common.write_page_text(path, "model text  \n\n")
            self.assertEqual(path.read_bytes(), b"model text\n")

    async def test_queue_then_combines_complete_document_pages(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            document = DocumentJob(
                source="source",
                document_id="doc",
                pdf_path=Path(tmp_dir) / "doc.pdf",
                output_dir=Path(tmp_dir) / "out",
            )
            jobs, states = build_page_jobs([document], page_counter=lambda _: 2)

            async def processor(job: PageJob) -> PageResult:
                job.output_path.parent.mkdir(parents=True, exist_ok=True)
                if job.page_number == 1:
                    text = "text for page 1\n<think>internal note</think>\nvisible ending\n"
                else:
                    text = "text for page 2\n<THINK>\nmultiline\nnote\n</THINK>\nvisible page 2\n"
                job.output_path.write_text(text, encoding="utf-8")
                write_page_markdown(job.markdown_path, text, raw_output=False)
                return PageResult(job=job, status="completed")

            await run_page_queue(jobs, states, concurrency=2, processor=processor)
            full_path = combine_document_pages(states[("source", "doc")])
            full_text = full_path.read_text(encoding="utf-8")
            page_one_text = states[("source", "doc")].page_paths[1].read_text(encoding="utf-8")

        self.assertIn("<think>internal note</think>", page_one_text)
        self.assertIn("--- Page 1 ---\n\ntext for page 1\n\nvisible ending", full_text)
        self.assertIn("--- Page 2 ---\n\ntext for page 2\n\nvisible page 2", full_text)
        self.assertNotIn("internal note", full_text)
        self.assertNotIn("multiline\nnote", full_text)
        self.assertNotIn("<think>", full_text.lower())

    async def test_queue_updates_progress_callback_for_each_page(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            document = DocumentJob(
                source="source",
                document_id="doc",
                pdf_path=Path(tmp_dir) / "doc.pdf",
                output_dir=Path(tmp_dir) / "out",
            )
            jobs, states = build_page_jobs([document], page_counter=lambda _: 2)
            seen_statuses: list[str] = []
            closed = False

            def progress_callback(result: PageResult) -> None:
                seen_statuses.append(result.status)

            def close_progress() -> None:
                nonlocal closed
                closed = True

            async def processor(job: PageJob) -> PageResult:
                return PageResult(job=job, status="completed")

            with patch(
                "pipeline.stg_01_ocr.common.make_progress_callback",
                return_value=ProgressReporter(
                    callback=progress_callback,
                    close=close_progress,
                ),
            ):
                await run_page_queue(
                    jobs,
                    states,
                    concurrency=2,
                    processor=processor,
                    show_progress=True,
                )

        self.assertEqual(seen_statuses, ["completed", "completed"])
        self.assertTrue(closed)


if __name__ == "__main__":
    unittest.main()
