from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pipeline.stg_01_ocr import common
from pipeline.stg_01_ocr.general import runner as general
from pipeline.stg_01_ocr.render import RenderedPage


class BackfillTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.transcribe_calls = 0

        async def transcribe(context: common.PageContext) -> str:
            self.transcribe_calls += 1
            return "new model output"

        self.spec = common.RunnerSpec(
            name="fake",
            description="fake",
            default_model_name="fake/model",
            transcribe=transcribe,
        )

    def _job(self, root: Path) -> common.PageJob:
        return common.PageJob(
            source="source",
            document_id="doc",
            pdf_path=root / "doc.pdf",
            page_number=1,
            output_path=root / "out" / "page_1.txt",
        )

    async def _process(self, job: common.PageJob, args, *, client=None):
        return await common.process_page_job(
            job=job,
            spec=self.spec,
            client=client,
            model_name=args.model_name,
            args=args,
            request_limiter=common.RequestLimiter(args.max_inflight_requests),
            repetition_policy=self.spec.repetition_policy(args),
        )

    async def test_raw_page_without_markdown_is_backfilled_without_model(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            job = self._job(Path(tmp_dir))
            job.output_path.parent.mkdir(parents=True)
            job.output_path.write_text(
                "before\n<think>private</think>\nafter\n",
                encoding="utf-8",
            )
            args = general.parse_args([])
            result = await self._process(job, args, client=None)
            markdown = job.markdown_path.read_text(encoding="utf-8")

        self.assertEqual(result.status, "backfilled")
        self.assertEqual(self.transcribe_calls, 0)
        self.assertEqual(markdown, "before\n\nafter\n")

    async def test_backfill_applies_comparison_postprocessor_without_rewriting_raw(
        self,
    ) -> None:
        seen: list[str] = []

        def postprocess(text: str, args) -> str:
            seen.append(text)
            return common.Transcription(
                text.replace("[remove]", "cleaned"),
                raw_text="postprocessor audit text must be ignored",
                repetition_trimmed=True,
            )

        spec = common.RunnerSpec(
            name="postprocessed",
            description="postprocessed",
            default_model_name="fake/model",
            transcribe=self.spec.transcribe,
            postprocess_comparison_text=postprocess,
        )
        with TemporaryDirectory() as tmp_dir:
            job = self._job(Path(tmp_dir))
            job.output_path.parent.mkdir(parents=True)
            raw_text = "verbatim [remove] raw response\n"
            job.output_path.write_text(raw_text, encoding="utf-8")
            args = common.parse_args(spec, [])

            result = await common.process_page_job(
                job=job,
                spec=spec,
                client=None,
                model_name=args.model_name,
                args=args,
                request_limiter=common.RequestLimiter(args.max_inflight_requests),
                repetition_policy=spec.repetition_policy(args),
            )

            self.assertEqual(job.output_path.read_text(encoding="utf-8"), raw_text)
            self.assertEqual(
                job.markdown_path.read_text(encoding="utf-8"),
                "verbatim cleaned raw response\n",
            )

        self.assertEqual(seen, [raw_text])
        self.assertEqual(result.status, "backfilled")
        self.assertTrue(result.repetition_trimmed)

    async def test_page_with_both_artifacts_is_skipped(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            job = self._job(Path(tmp_dir))
            job.output_path.parent.mkdir(parents=True)
            job.output_path.write_text("raw\n", encoding="utf-8")
            job.markdown_path.write_text("markdown\n", encoding="utf-8")
            result = await self._process(job, general.parse_args([]), client=None)
        self.assertEqual(result.status, "skipped")
        self.assertEqual(self.transcribe_calls, 0)

    async def test_explicit_backfill_renormalizes_existing_markdown(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            job = self._job(Path(tmp_dir))
            job.output_path.parent.mkdir(parents=True)
            job.output_path.write_text("raw source\n", encoding="utf-8")
            job.markdown_path.write_text("obsolete normalization\n", encoding="utf-8")
            args = general.parse_args(["--backfill-markdown"])

            result = await self._process(job, args, client=None)

            self.assertEqual(result.status, "backfilled")
            self.assertEqual(
                job.markdown_path.read_text(encoding="utf-8"),
                "raw source\n",
            )
            self.assertEqual(self.transcribe_calls, 0)

    async def test_page_with_neither_artifact_is_ocrd(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            job = self._job(Path(tmp_dir))
            args = general.parse_args([])
            with patch(
                "pipeline.stg_01_ocr.common.render_page_isolated",
                return_value=RenderedPage("unused"),
            ):
                result = await self._process(job, args, client=object())
            raw = job.output_path.read_text(encoding="utf-8")
            markdown = job.markdown_path.read_text(encoding="utf-8")

        self.assertEqual(result.status, "completed")
        self.assertEqual(self.transcribe_calls, 1)
        self.assertEqual(raw, "new model output\n")
        self.assertEqual(markdown, "new model output\n")

    async def test_force_regenerates_raw_and_markdown(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            job = self._job(Path(tmp_dir))
            job.output_path.parent.mkdir(parents=True)
            job.output_path.write_text("old raw\n", encoding="utf-8")
            job.markdown_path.write_text("old markdown\n", encoding="utf-8")
            args = general.parse_args(["--force"])
            with patch(
                "pipeline.stg_01_ocr.common.render_page_isolated",
                return_value=RenderedPage("unused"),
            ):
                result = await self._process(job, args, client=object())

            self.assertEqual(result.status, "completed")
            self.assertEqual(job.output_path.read_text(encoding="utf-8"), "new model output\n")
            self.assertEqual(job.markdown_path.read_text(encoding="utf-8"), "new model output\n")

    async def test_force_failure_leaves_retry_marker_and_next_run_retries(self) -> None:
        async def fail(context: common.PageContext) -> str:
            raise RuntimeError("model failed")

        failing_spec = common.RunnerSpec(
            name="failing",
            description="failing",
            default_model_name="fake/model",
            transcribe=fail,
        )
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            job = self._job(root)
            job.output_path.parent.mkdir(parents=True)
            job.output_path.write_text("old raw\n", encoding="utf-8")
            job.markdown_path.write_text("old markdown\n", encoding="utf-8")
            full_path = job.output_path.parent / "full.txt"
            full_path.write_text("old full\n", encoding="utf-8")
            state = common.DocumentState("source", "doc", job.output_path.parent, 1)
            force_args = general.parse_args(["--force"])

            backups = common.invalidate_stale_full_texts(
                [job],
                {job.document_key: state},
                force_args,
            )
            with patch(
                "pipeline.stg_01_ocr.common.render_page_isolated",
                return_value=RenderedPage("unused"),
            ):
                with self.assertRaisesRegex(RuntimeError, "model failed"):
                    await common.process_page_job(
                        job=job,
                        spec=failing_spec,
                        client=object(),
                        model_name=force_args.model_name,
                        args=force_args,
                        request_limiter=common.RequestLimiter(
                            force_args.max_inflight_requests
                        ),
                        repetition_policy=failing_spec.repetition_policy(force_args),
                    )

            self.assertFalse(full_path.exists())
            self.assertEqual(backups[job.document_key].read_text(encoding="utf-8"), "old full\n")
            self.assertTrue(job.retry_marker_path.exists())
            normal_args = general.parse_args([])
            self.assertTrue(common.page_jobs_need_model([job], normal_args))

            with patch(
                "pipeline.stg_01_ocr.common.render_page_isolated",
                return_value=RenderedPage("unused"),
            ):
                result = await self._process(job, normal_args, client=object())

            self.assertEqual(result.status, "completed")
            self.assertFalse(job.retry_marker_path.exists())
            self.assertEqual(job.output_path.read_text(encoding="utf-8"), "new model output\n")
            self.assertFalse(common.page_jobs_need_model([job], normal_args))

    async def test_trimmed_repetition_keeps_verbatim_raw_audit_text(self) -> None:
        async def transcribe(context: common.PageContext) -> str:
            return common.Transcription(
                "accepted prefix + one cycle",
                raw_text="accepted prefix + cyclecyclecyclecycle",
                repetition_trimmed=True,
            )

        spec = common.RunnerSpec(
            name="trimmed",
            description="trimmed",
            default_model_name="fake/model",
            transcribe=transcribe,
        )
        with TemporaryDirectory() as tmp_dir:
            job = self._job(Path(tmp_dir))
            args = general.parse_args([])
            with patch(
                "pipeline.stg_01_ocr.common.render_page_isolated",
                return_value=RenderedPage("unused"),
            ):
                result = await common.process_page_job(
                    job=job,
                    spec=spec,
                    client=object(),
                    model_name=args.model_name,
                    args=args,
                    request_limiter=common.RequestLimiter(args.max_inflight_requests),
                    repetition_policy=spec.repetition_policy(args),
                )

            self.assertEqual(
                job.output_path.read_text(encoding="utf-8"),
                "accepted prefix + cyclecyclecyclecycle\n",
            )
            self.assertEqual(
                job.markdown_path.read_text(encoding="utf-8"),
                "accepted prefix + one cycle\n",
            )
            self.assertTrue(result.repetition_trimmed)

    async def test_fresh_ocr_postprocessor_preserves_transcriber_audit_text(
        self,
    ) -> None:
        seen: list[str] = []

        async def transcribe(context: common.PageContext) -> str:
            return common.Transcription(
                "comparison [remove] text",
                raw_text="verbatim model response",
            )

        def postprocess(text: str, args) -> str:
            seen.append(text)
            return common.Transcription(
                text.replace("[remove] ", ""),
                raw_text="postprocessor audit text must be ignored",
                repetition_trimmed=True,
            )

        spec = common.RunnerSpec(
            name="postprocessed",
            description="postprocessed",
            default_model_name="fake/model",
            transcribe=transcribe,
            postprocess_comparison_text=postprocess,
        )
        with TemporaryDirectory() as tmp_dir:
            job = self._job(Path(tmp_dir))
            args = common.parse_args(spec, [])
            with patch(
                "pipeline.stg_01_ocr.common.render_page_isolated",
                return_value=RenderedPage("unused"),
            ):
                result = await common.process_page_job(
                    job=job,
                    spec=spec,
                    client=object(),
                    model_name=args.model_name,
                    args=args,
                    request_limiter=common.RequestLimiter(
                        args.max_inflight_requests
                    ),
                    repetition_policy=spec.repetition_policy(args),
                )

            self.assertEqual(
                job.output_path.read_text(encoding="utf-8"),
                "verbatim model response\n",
            )
            self.assertEqual(
                job.markdown_path.read_text(encoding="utf-8"),
                "comparison text\n",
            )

        self.assertEqual(seen, ["comparison [remove] text"])
        self.assertTrue(result.repetition_trimmed)

    async def test_ovisocr2_fresh_and_backfill_share_official_cleanup(self) -> None:
        raw = (
            "Heading\n\n"
            '<img src="images/bbox_1_2_3_4.jpg" />\n\n'
            + "a" * 7_900
            + "wxyz" * 25
        )

        class Completions:
            def __init__(self) -> None:
                self.calls = 0

            async def create(self, **kwargs):
                del kwargs
                self.calls += 1
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(content=raw),
                            finish_reason="stop",
                        )
                    ]
                )

        completions = Completions()
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=completions)
        )
        args = general.parse_args(
            ["--model-name", general.OVISOCR2_MODEL_NAME]
        )
        with TemporaryDirectory() as tmp_dir:
            job = self._job(Path(tmp_dir))
            with patch(
                "pipeline.stg_01_ocr.common.render_page_isolated",
                return_value=RenderedPage("unused"),
            ):
                fresh = await common.process_page_job(
                    job=job,
                    spec=general.SPEC,
                    client=client,
                    model_name=args.model_name,
                    args=args,
                    request_limiter=common.RequestLimiter(
                        args.max_inflight_requests
                    ),
                    repetition_policy=general.SPEC.repetition_policy(args),
                )

            verbatim = job.output_path.read_text(encoding="utf-8")
            fresh_markdown = job.markdown_path.read_text(encoding="utf-8")
            job.markdown_path.unlink()
            backfilled = await common.process_page_job(
                job=job,
                spec=general.SPEC,
                client=None,
                model_name=args.model_name,
                args=args,
                request_limiter=common.RequestLimiter(
                    args.max_inflight_requests
                ),
                repetition_policy=general.SPEC.repetition_policy(args),
            )
            backfilled_markdown = job.markdown_path.read_text(encoding="utf-8")

        # The official cleaner accepts a smaller repeated tail than the shared
        # retry detector's stop-finish gate, so no sampling retry is needed.
        self.assertEqual(completions.calls, 1)
        self.assertEqual(verbatim, raw + "\n")
        self.assertNotIn("images/bbox_", fresh_markdown)
        self.assertTrue(fresh_markdown.endswith("wxyz\n"))
        self.assertEqual(backfilled_markdown, fresh_markdown)
        self.assertTrue(fresh.repetition_trimmed)
        self.assertTrue(backfilled.repetition_trimmed)

    async def test_ovisocr2_long_repeat_unit_backfills_identically(self) -> None:
        unit = "".join(chr(0x400 + index) for index in range(250))
        raw = "P" * 5_000 + unit * 12

        class Completions:
            def __init__(self) -> None:
                self.calls = 0

            async def create(self, **kwargs):
                del kwargs
                self.calls += 1
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(content=raw),
                            finish_reason="stop",
                        )
                    ]
                )

        completions = Completions()
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=completions)
        )
        args = general.parse_args(
            ["--model-name", general.OVISOCR2_MODEL_NAME]
        )
        with TemporaryDirectory() as tmp_dir:
            job = self._job(Path(tmp_dir))
            with patch(
                "pipeline.stg_01_ocr.common.render_page_isolated",
                return_value=RenderedPage("unused"),
            ):
                fresh = await common.process_page_job(
                    job=job,
                    spec=general.SPEC,
                    client=client,
                    model_name=args.model_name,
                    args=args,
                    request_limiter=common.RequestLimiter(
                        args.max_inflight_requests
                    ),
                    repetition_policy=general.SPEC.repetition_policy(args),
                )

            fresh_markdown = job.markdown_path.read_text(encoding="utf-8")
            job.markdown_path.unlink()
            backfilled = await common.process_page_job(
                job=job,
                spec=general.SPEC,
                client=None,
                model_name=args.model_name,
                args=args,
                request_limiter=common.RequestLimiter(
                    args.max_inflight_requests
                ),
                repetition_policy=general.SPEC.repetition_policy(args),
            )
            backfilled_markdown = job.markdown_path.read_text(encoding="utf-8")

        self.assertEqual(completions.calls, 3)
        self.assertEqual(fresh_markdown, raw + "\n")
        self.assertEqual(backfilled_markdown, fresh_markdown)
        self.assertFalse(fresh.repetition_trimmed)
        self.assertFalse(backfilled.repetition_trimmed)

    async def test_backfill_only_missing_raw_fails_without_model(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            job = self._job(Path(tmp_dir))
            args = general.parse_args(["--backfill-markdown"])
            result = await self._process(job, args, client=None)
        self.assertEqual(result.status, "failed")
        self.assertIn("raw page text is missing", result.error)
        self.assertEqual(self.transcribe_calls, 0)

    def test_needs_model_ignores_markdown_only_work(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            job = self._job(Path(tmp_dir))
            job.output_path.parent.mkdir(parents=True)
            job.output_path.write_text("raw\n", encoding="utf-8")
            self.assertFalse(common.page_jobs_need_model([job], general.parse_args([])))
            job.output_path.unlink()
            self.assertTrue(common.page_jobs_need_model([job], general.parse_args([])))
            self.assertFalse(
                common.page_jobs_need_model(
                    [job],
                    general.parse_args(["--backfill-markdown"]),
                )
            )

    def test_argument_resolver_runs_after_common_defaults(self) -> None:
        seen: list[tuple[str, int, int]] = []

        def resolve(args) -> None:
            seen.append(
                (
                    args.output_variant,
                    args.max_inflight_requests,
                    args.max_num_seqs,
                )
            )
            if args.max_tokens is None:
                args.max_tokens = 16384

        spec = common.RunnerSpec(
            name="resolved",
            description="resolved",
            default_model_name="fake/model",
            transcribe=self.spec.transcribe,
            default_max_tokens=None,
            resolve_arguments=resolve,
        )

        args = common.parse_args(spec, ["--concurrency", "3"])
        common.validate_args(spec, args)

        self.assertEqual(seen, [("", 3, 3)])
        self.assertEqual(args.max_tokens, 16384)

    def test_full_text_is_assembled_from_page_markdown(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            document = common.DocumentJob("source", "doc", root / "doc.pdf", root / "out")
            jobs, states = common.build_page_jobs([document], page_counter=lambda _: 1)
            jobs[0].output_path.parent.mkdir(parents=True)
            jobs[0].output_path.write_text("raw must not appear\n", encoding="utf-8")
            jobs[0].markdown_path.write_text("canonical markdown\n", encoding="utf-8")
            states[("source", "doc")].completed_pages.add(1)
            full_path = common.combine_document_pages(states[("source", "doc")])
            full_text = full_path.read_text(encoding="utf-8")

        self.assertEqual(full_text, "--- Page 1 ---\n\ncanonical markdown\n")


if __name__ == "__main__":
    unittest.main()
