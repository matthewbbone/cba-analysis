from __future__ import annotations

from dataclasses import dataclass


WAGE_TABLE_EXTRACTION_CLASS = "wage_table"
WAGE_TABLE_DIMENSIONS = ("occupation", "experience", "education")

WAGE_TABLE_PROMPT = """
Extract every distinct wage schedule table from the contract text.

A wage schedule table is a table-like block whose purpose is to state the pay
rate for work performed: wages, rates of pay, salary steps, hourly rates,
annual salaries, or compensation schedules. Extract the full verbatim table
text, including headings, row labels, column labels, notes immediately attached
to the table, and all numeric values.

Only extract a table if its values are amounts of pay (e.g. dollar amounts per
hour, week, month, or year) earned for working. Do NOT extract other
table-like blocks even if they contain numbers, dates, hours, or dollar
amounts. Specifically exclude:
- lists of holidays or observed days off
- vacation, annual leave, sick leave, or other leave accrual schedules
  (values are hours or days of leave, not pay rates)
- leave buy-out or leave conversion tables
- insurance, health care, pension, or other benefit contribution rates
- lists of job titles, class codes, or pay-range assignments that do not
  themselves state a wage amount
- dues, fees, allowance, or reimbursement schedules

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

    wage_table_text = (
        "Wage Schedule\n"
        "| Occupation | Start | After 1 Year |\n"
        "| Laborer | $15.00 | $16.25 |\n"
        "| Clerk | $14.50 | $15.75 |"
    )
    # Distractor blocks (holiday list, leave accrual schedule) are included in
    # the example text but deliberately NOT extracted, to teach the model that
    # table-like blocks without pay rates are not wage tables.
    document_text = (
        "Holidays\n"
        "| Holiday |\n"
        "| New Year's Day |\n"
        "| Independence Day |\n"
        "| Thanksgiving Day |\n"
        "\n"
        + wage_table_text
        + "\n\n"
        "Annual Leave\n"
        "| Years of Service | Leave Days Per Year |\n"
        "| Up to 5 | 10 |\n"
        "| Over 5 | 15 |"
    )
    return [
        lx.data.ExampleData(
            text=document_text,
            extractions=[
                lx.data.Extraction(
                    extraction_class=WAGE_TABLE_EXTRACTION_CLASS,
                    extraction_text=wage_table_text,
                    attributes={"dimensions": ["occupation", "experience"]},
                )
            ],
        )
    ]
