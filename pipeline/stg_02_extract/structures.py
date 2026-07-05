from __future__ import annotations

from dataclasses import dataclass


WAGE_TABLE_EXTRACTION_CLASS = "wage_table"
WAGE_TABLE_DIMENSIONS = ("occupation", "experience", "education")

WAGE_TABLE_PROMPT = """
Extract every distinct wage schedule table from the contract text.

A wage schedule table is a table-like block whose purpose is to state wages,
rates of pay, salary steps, hourly rates, annual salaries, or compensation
schedules. Extract the full verbatim table text, including headings, row labels,
column labels, notes immediately attached to the table, and all numeric values.

Do not extract narrative wage clauses unless they contain a table-like wage
schedule. Do not summarize, normalize, transpose, or calculate values.

For each wage table, set extraction_class to wage_table. Set attributes to a
single key named dimensions. The dimensions value must be a list containing only
the stratification dimensions present in the table, chosen from:
occupation, experience, education.
""".strip()


@dataclass(frozen=True)
class ExtractionTask:
    prompt: str
    extraction_class: str
    attributes: dict[str, list[str]]

    def as_json_object(self) -> dict[str, object]:
        return {
            "prompt": self.prompt,
            "extraction_class": self.extraction_class,
            "attributes": self.attributes,
        }


WAGE_TABLE_TASK = ExtractionTask(
    prompt=WAGE_TABLE_PROMPT,
    extraction_class=WAGE_TABLE_EXTRACTION_CLASS,
    attributes={"dimensions": list(WAGE_TABLE_DIMENSIONS)},
)


def synthetic_wage_table_examples():
    import langextract as lx

    table_text = (
        "Wage Schedule\n"
        "| Occupation | Start | After 1 Year |\n"
        "| Laborer | $15.00 | $16.25 |\n"
        "| Clerk | $14.50 | $15.75 |"
    )
    return [
        lx.data.ExampleData(
            text=table_text,
            extractions=[
                lx.data.Extraction(
                    extraction_class=WAGE_TABLE_EXTRACTION_CLASS,
                    extraction_text=table_text,
                    attributes={"dimensions": ["occupation", "experience"]},
                )
            ],
        )
    ]
