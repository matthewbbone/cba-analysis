import unittest

from pipeline.stg_01_ocr.markdown import (
    MarkdownBlock,
    assemble_markdown_blocks,
    assemble_raw_blocks,
    normalize_html_tables,
    normalize_page_markdown,
    remove_think_blocks,
)


class MarkdownNormalizationTests(unittest.TestCase):
    def test_html_table_becomes_pipe_table(self) -> None:
        source = (
            "Intro\n"
            "<table><thead><tr><th>Grade</th><th>Rate</th></tr></thead>"
            "<tbody><tr><td>A</td><td>$20 &amp; up</td></tr></tbody></table>"
            "\nOutro"
        )

        result = normalize_html_tables(source)

        self.assertEqual(
            result.text,
            "Intro\n"
            "| Grade | Rate |\n"
            "| --- | --- |\n"
            "| A | $20 & up |\n"
            "Outro",
        )
        self.assertFalse(result.expanded_spans)

    def test_colspan_and_rowspan_repeat_merged_cell_values(self) -> None:
        source = (
            '<table><tr><th rowspan="2">Band</th><th colspan="3">Hourly</th></tr>'
            "<tr><td>Start</td><td>After 1 year</td><td>After 2 years</td></tr>"
            "</table>"
        )

        result = normalize_html_tables(source)

        self.assertEqual(
            result.text,
            "| Band | Hourly | Hourly | Hourly |\n"
            "| --- | --- | --- | --- |\n"
            "| Band | Start | After 1 year | After 2 years |",
        )
        self.assertTrue(result.expanded_spans)

    def test_non_html_page_passes_through_byte_identically(self) -> None:
        source = "# Wages\n\n| Grade | Rate |\n| --- | --- |\n| A | $20.00 |\n"

        result = normalize_page_markdown(source)

        self.assertEqual(result.text, source)
        self.assertFalse(result.expanded_spans)

    def test_raw_output_bypasses_html_conversion_but_still_strips_think(self) -> None:
        source = "<think>private</think><table><tr><td>A</td></tr></table>"

        result = normalize_page_markdown(source, raw_output=True)

        self.assertEqual(result.text, "<table><tr><td>A</td></tr></table>")
        self.assertFalse(result.expanded_spans)

    def test_think_blocks_use_the_shared_case_insensitive_rule(self) -> None:
        source = (
            "before<think>one</think>middle"
            "<THINK data-model='x'>\ntwo\n</THINK>after"
        )

        self.assertEqual(remove_think_blocks(source), "beforemiddleafter")
        self.assertEqual(normalize_page_markdown(source).text, "beforemiddleafter")

    def test_raw_block_assembly_does_not_normalize_model_responses(self) -> None:
        blocks = [
            MarkdownBlock("  first  ", label="text"),
            " ",
            "<think>audit</think><table><tr><td>A</td></tr></table>\n",
        ]

        raw = assemble_raw_blocks(blocks)
        normalized = assemble_markdown_blocks(blocks)

        self.assertEqual(
            raw,
            "first\n\n<think>audit</think><table><tr><td>A</td></tr></table>",
        )
        self.assertEqual(normalized.text, "first\n\n| A |\n| --- |")


if __name__ == "__main__":
    unittest.main()
