import base64
from io import BytesIO, StringIO
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from PIL import Image

from pipeline.stg_01_ocr.render import (
    PROJECT_ROOT,
    PageRenderError,
    RenderedPage,
    main,
    render_page_isolated,
)


def make_png_base64(*, size: tuple[int, int] = (3, 2)) -> str:
    buffer = BytesIO()
    Image.new("RGB", size, (12, 34, 56)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class RenderPageTests(unittest.TestCase):
    def test_isolated_renderer_uses_dedicated_module_from_project_root(self) -> None:
        encoded = make_png_base64()
        completed = subprocess.CompletedProcess(
            args=["python"],
            returncode=0,
            stdout=f"{encoded}\n",
            stderr="",
        )
        pdf_path = Path("relative/path/doc.pdf")

        with patch(
            "pipeline.stg_01_ocr.render.subprocess.run",
            return_value=completed,
        ) as run:
            rendered = render_page_isolated(pdf_path, 4, 200)

        self.assertEqual(rendered.png_base64, encoded)
        run.assert_called_once_with(
            [
                sys.executable,
                "-m",
                "pipeline.stg_01_ocr.render",
                str(pdf_path.expanduser().resolve()),
                "4",
                "200",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )

    def test_isolated_renderer_reports_nonzero_exit(self) -> None:
        completed = subprocess.CompletedProcess(
            args=["python"],
            returncode=-6,
            stdout="",
            stderr="*** stack smashing detected ***: terminated\n",
        )

        with patch(
            "pipeline.stg_01_ocr.render.subprocess.run",
            return_value=completed,
        ):
            with self.assertRaisesRegex(PageRenderError, "stack smashing"):
                render_page_isolated(Path("doc.pdf"), 1, 200)

    def test_isolated_renderer_rejects_non_png_stdout(self) -> None:
        encoded_text = base64.b64encode(b"plain text, not a PNG").decode("ascii")
        completed = subprocess.CompletedProcess(
            args=["python"],
            returncode=0,
            stdout=encoded_text,
            stderr="",
        )

        with patch(
            "pipeline.stg_01_ocr.render.subprocess.run",
            return_value=completed,
        ):
            with self.assertRaisesRegex(PageRenderError, "invalid PNG"):
                render_page_isolated(Path("doc.pdf"), 1, 200)

    def test_rendered_page_exposes_data_url_without_reencoding(self) -> None:
        encoded = make_png_base64()

        rendered = RenderedPage(encoded)

        self.assertEqual(rendered.data_url, f"data:image/png;base64,{encoded}")

    def test_rendered_page_to_image_round_trips_png(self) -> None:
        encoded = make_png_base64(size=(7, 5))

        image = RenderedPage(encoded).to_image()

        self.assertEqual(image.size, (7, 5))
        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.getpixel((0, 0)), (12, 34, 56))

    def test_cli_keeps_native_library_messages_off_stdout_wire(self) -> None:
        encoded = make_png_base64()

        def noisy_render(*args, **kwargs):
            print("native library warning")
            return RenderedPage(encoded)

        stdout = StringIO()
        stderr = StringIO()
        with (
            patch("pipeline.stg_01_ocr.render.render_page", side_effect=noisy_render),
            patch("pipeline.stg_01_ocr.render.sys.stdout", stdout),
            patch("pipeline.stg_01_ocr.render.sys.stderr", stderr),
        ):
            exit_code = main(["doc.pdf", "1", "200"])

        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout.getvalue(), encoded)
        self.assertIn("native library warning", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
