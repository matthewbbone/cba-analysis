from __future__ import annotations

from dataclasses import dataclass


WAGE_TABLE_EXTRACTION_CLASS = "wage_table"
WAGE_TABLE_DIMENSIONS = ("occupation", "experience", "education")

WAGE_TABLE_PROMPT = """
Extract every distinct BASE WAGE schedule table from the contract text.

A base wage schedule table is a table-like block whose purpose is to state the
base rate of pay for a job classification, salary step or lane, or pay grade:
wages, rates of pay, salary steps, hourly rates, annual salaries, or
compensation schedules. Every data row must carry an explicit monetary pay
amount — either a currency figure such as $27.44, or a salary/hourly/annual
figure presented in a rate grid (bare numbers like 18105 are fine when they are
laid out as pay-rate cells in a table). Extract the full verbatim table text,
including headings, row labels, column labels, notes immediately attached to the
table, and all numeric values.

Each extraction must be the COMPLETE table: begin at the table's title or
heading (or its first header row if it has no title) and end at its final row.
Never emit a fragment of a table, a single row, or a table that is missing its
header rows. Never emit a header row or a list of job titles on its own with no
pay amounts. If the same table continues after an interruption such as a page
break, include the continuation as part of the same extraction. One table =
one extraction; do not split a table into multiple extractions.

Only extract a table whose cells are absolute amounts of pay (e.g. dollar
amounts per hour, week, month, or year) earned for working. Do NOT extract other
table-like blocks even if they contain numbers, dates, hours, ratios, percentages,
or dollar amounts. Specifically exclude:
- lists of holidays or observed days off
- vacation, annual leave, sick leave, or other leave accrual schedules
  (values are hours or days of leave, not pay rates)
- leave buy-out or leave conversion tables
- insurance, health care, pension, or other benefit contribution rates
- lists of job titles, class codes, or pay-range assignments that do not
  themselves state a wage amount
- dues, fees, allowance, or reimbursement schedules
- percentage-of-pay rules, such as disability or injury pay stated as a percent
  of the regular rate, or shift/rank differentials stated as "X% above" another
  rank (these state a rule, not a base rate)
- supplemental, extra-duty, or stipend schedules: coaching, athletic and
  activity "ratio" or stipend tables, per-session, per-game, or per-hour
  stipends, club or activity advisor pay, chaperone pay
- longevity or career-increment clauses, whether stated as prose or as single
  amounts

Do not extract narrative wage clauses unless they contain a table-like base wage
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

    # Example 0: occupation x experience wage table, embedded in a document
    # that also contains distractor blocks (a vacation schedule and a sick
    # leave accrual schedule). The distractors are deliberately NOT extracted,
    # to teach the model that table-like blocks without pay rates are not wage
    # tables.
    wage_table_text_0 = (
        "Payment Rates\n"
        "| Occupation | Start | After 1 Year |\n"
        "| Laborer | $15.00 | $16.25 |\n"
        "| Clerk | $14.50 | $15.75 |"
    )
    document_text_0 = (
        "...\n"
        "Section 3. Vacations shall be scheduled and granted for periods of time requested by the employee subject to management's responsibility to maintain efficient operations. If the nature of the work makes it necessary to limit the number of employees on vacation at the same time, the employee with the greatest seniority as it relates to total years of service with the Employer shall be given preference in the event of any conflict in selection. Where reasonable opportunities are available for selection of vacation on a seniority basis, approved requests shall not be revoked if a conflict in selection develops after the selection period. The selection periods shall be as follows, unless there are existing or subsequent agreements on the selection period at appropriate local levels: \n"
        "\n\nVacations\n"
        "| Selection Period       | Response to Requests | Vacation Period         |\n"
        "|------------------------|----------------------|-------------------------|\n"
        "| September 1-30         | October 10           | January 1-June 30       |\n"
        "| March 1-31             | April 10             | July 1-December 31      |\n"
        "\n"
        "The scheduling of weekends off in conjunction with pre-selected vacations may be the subject of a local level meet and discuss.\n"
        "...\n"
        + wage_table_text_0 +
        "...\n"
        "\n"
        "Section 5. Where a family member’s serious health condition requires the employee’s absence from work beyond 20 days (150/160 hours as applicable) in a calendar year, permanent employees with at least one year of service may use accrued sick leave, in addition to that provided by Section 4 above.\n\n"
        "a. Employees who meet the eligibility criteria in b. through e. below may use accrued sick leave in accordance with the following schedule:\n\n"
        "Leave Service Credit | Sick Family Allowance\n"
        "Over 1 year to 3 years | Up to 52.5/56 additional hours (7 days)\n"
        "Over 3 years to 15 years | Up to 112.5/120 additional hours (15 days)\n"
        "Over 15 years to 25 years | Up to 150/160 additional hours (20 days)\n"
        "Over 25 years | Up to 195/208 additional hours (26 days)\n"
        "b. During the initial 20 days (150/160 hours) of absence, paid annual and personal leave and/or unpaid leave shall be used and may include leave provided under Section 4 above. The additional sick family leave allowance must be used prospectively, and may not be retroactively charged for any of the initial 20 days (150/160 hours). A separate 20 day (150/160 hour) requirement must be met for each different serious health condition and/or family member and for each calendar year, even if not all of the additional days were used during the previous calendar year.\n"
        "...\n"
    )

    # Example 1: occupation x education wage table (rows are occupations,
    # columns are educational attainment levels; cell values are pay rates).
    wage_table_text_1 = (
        "Wage Schedule\n"
        "| Occupation | Apprenticeship | Bachelor's Degree |\n"
        "| Laborer | $18.00 | $20.50 |\n"
        "| Machinist | $22.00 | $24.75 |\n"
        "| Clerk | $17.25 | $19.50 |\n"
        "| Engineer | $28.50 | $32.00 |"
    )
    document_text_1 = (
        "...\n"
        "Section 8. The hourly rates of pay set forth below reflect the "
        "employee's occupational classification together with the level of "
        "educational attainment achieved.\n"
        "...\n"
        + wage_table_text_1 +
        "\n...\n"
    )

    # Example 2: occupation-only wage table (columns are effective dates, not a
    # stratification dimension, so the only dimension present is occupation).
    wage_table_text_2 = (
        "Wage Schedule\n"
        "| Occupation | 1 April, 1984 | 1 April, 1985 |\n"
        "| Laborer | $15.00-$18.00 | 2% increase |\n"
        "| Clerk | $14.50-$17.50 | 2% increase |"
    )
    document_text_2 = (
        "...\n"
        "Section 12. Rates of pay for the term of this Agreement shall be as "
        "set forth in the schedule below, with the adjustments indicated "
        "effective on the dates shown.\n"
        "...\n"
        + wage_table_text_2 +
        "\n...\n"
    )

    # Example 3: a genuine base wage table embedded among three distractor
    # blocks that must NOT be extracted, mirroring the real false positives:
    # (a) an injury-pay percentage-of-rate block, (b) a coaching/activity
    # stipend "ratio" schedule, and (c) a header-only classification list with
    # no pay amounts. Only the base wage table is extracted.
    wage_table_text_3 = (
        "Wage Schedule\n"
        "| Occupation | Start | After 1 Year |\n"
        "| Operator | $24.18 | $24.78 |\n"
        "| Technician | $28.61 | $29.20 |"
    )
    injury_pay_distractor_3 = (
        "For Non-Work Related Injuries: 80% of Regular Pay Rate\n"
        "For Work Related Injuries, 0-19 Years of Service: a pay rate that is "
        "not less than 85% of Regular Pay Rate\n"
        "For Work Related Injuries, 20 Years and Over: 90% of Regular Pay Rate"
    )
    stipend_distractor_3 = (
        "Extra-Curricular Assignments\n"
        "| Activity | Ratio | Stipend |\n"
        "| Head Basketball Coach | .06 | $2,040 |\n"
        "| Debate Coach | .067 | $2,280 |\n"
        "| Band Director | .056 | $1,905 |"
    )
    classification_distractor_3 = (
        "MAIL SERVICES\n"
        "| NO. | POSITION | STEP 1 | STEP 2 | STEP 3 |\n"
        "| J01100 | Automated Mail Processor |\n"
        "| J01101 | Courier Driver |"
    )
    document_text_3 = (
        "...\n"
        "Section 7. An employee absent due to injury shall be compensated "
        "according to the following schedule:\n"
        + injury_pay_distractor_3 +
        "\n...\n"
        "Section 9. Coaches and activity advisors shall receive supplemental "
        "compensation as set forth below.\n"
        + stipend_distractor_3 +
        "\n...\n"
        "Section 10. The hourly rates of pay for the classifications listed "
        "below are set forth in the wage schedule.\n"
        + wage_table_text_3 +
        "\n...\n"
        "The following classifications are assigned to the Mail Services unit:\n"
        + classification_distractor_3 +
        "\n...\n"
    )

    return [
        lx.data.ExampleData(
            text=document_text_0,
            extractions=[
                lx.data.Extraction(
                    extraction_class=WAGE_TABLE_EXTRACTION_CLASS,
                    extraction_text=wage_table_text_0,
                    attributes={"dimensions": ["occupation", "experience"]},
                ),
            ],
        ),
        lx.data.ExampleData(
            text=document_text_1,
            extractions=[
                lx.data.Extraction(
                    extraction_class=WAGE_TABLE_EXTRACTION_CLASS,
                    extraction_text=wage_table_text_1,
                    attributes={"dimensions": ["occupation", "education"]},
                ),
            ],
        ),
        lx.data.ExampleData(
            text=document_text_2,
            extractions=[
                lx.data.Extraction(
                    extraction_class=WAGE_TABLE_EXTRACTION_CLASS,
                    extraction_text=wage_table_text_2,
                    attributes={"dimensions": ["occupation"]},
                ),
            ],
        ),
        lx.data.ExampleData(
            text=document_text_3,
            extractions=[
                lx.data.Extraction(
                    extraction_class=WAGE_TABLE_EXTRACTION_CLASS,
                    extraction_text=wage_table_text_3,
                    attributes={"dimensions": ["occupation", "experience"]},
                ),
            ],
        ),
    ]
