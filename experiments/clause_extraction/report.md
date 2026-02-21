# Clause Extraction Experiment Report

## Overall Pairwise Agreement

| Method A | Method B | Avg Char IoU | Avg Token Jaccard | Label Agreement | N |
|----------|----------|-------------|-------------------|----------------|---|
| llm_segmentation | summary_segmentation | 0.409 | 0.569 | 93/144 (65%) | 144 |

## Per-Clause Agreement (across all method pairs)

| Clause | Avg Char IoU | Avg Token Jaccard | Occurrences |
|--------|-------------|-------------------|-------------|
| Overtime Clause | 0.191 | 0.698 | 17 |
| Work Hours Clause | 0.331 | 0.283 | 14 |
| Holidays Clause | 0.256 | 0.651 | 12 |
| Grievance Procedure Clause | 0.318 | 0.618 | 11 |
| Wages Clause | 0.331 | 0.406 | 9 |
| Vacation Clause | 0.638 | 0.887 | 9 |
| Parties to Agreement and Preamble | 0.674 | 0.753 | 8 |
| Jury Duty Clause | 0.790 | 0.691 | 7 |
| Bereavement Leave Clause | 0.772 | 0.788 | 6 |
| Arbitration Clause | 0.637 | 0.782 | 5 |
| Bargaining Unit Clause | 0.499 | 0.250 | 4 |
| Discipline and Discharge Clause | 0.457 | 0.340 | 4 |
| Dues Checkoff Clause | 0.512 | 0.623 | 3 |
| Management Rights Clause | 0.091 | 0.105 | 3 |
| Non-Discrimination Clause | 0.656 | 0.842 | 3 |
| Probationary Period Clause | 0.333 | 0.667 | 3 |
| Sick Leave Clause | 0.247 | 0.968 | 3 |
| Wage Differentials Clause | 0.307 | 0.258 | 3 |
| Bulletin Board Clause | 0.054 | 0.652 | 2 |
| Call-Back Pay Clause | 0.500 | 0.000 | 2 |
| No-Lockout Clause | 0.000 | 0.000 | 2 |
| No-Strike Clause | 0.840 | 0.827 | 2 |
| Union Business Leave Clause | 0.500 | 0.000 | 2 |
| Company Benefits and Policies Clause | 0.161 | 0.813 | 2 |
| Duration Clause | 0.038 | 0.241 | 2 |
| Recognition Clause | 0.993 | 0.942 | 1 |
| Reporting-Pay Clause | 0.990 | 0.985 | 1 |
| Notices Clause | 0.000 | 0.000 | 1 |
| Subcontracting Clause | 0.540 | 0.595 | 1 |
| Savings Clause | 0.000 | 0.000 | 1 |
| Reopener Clause | 0.000 | 0.000 | 1 |

**Total comparison rows:** 144