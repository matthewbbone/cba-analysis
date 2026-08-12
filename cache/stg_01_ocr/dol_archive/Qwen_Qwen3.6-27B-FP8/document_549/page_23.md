The user wants the text from the image transcribed verbatim.

1.  **Identify the structure:** The page is split into two columns. I need to transcribe the left column first, then the right column, or follow the visual flow. The text seems to be continuous across columns in some places, but mostly distinct sections. I will transcribe top-to-bottom, left-to-right as best as possible, maintaining the layout.

2.  **Transcribe Left Column:**
    *   "Assistant Customer Service $620.00 $15.50"
    *   "Assistant Bakery $620.00 $15.50"
    *   "Assistant Deli $620.00 $15.50"
    *   "Assistant Grocery $620.00 $15.50"
    *   "Assistant Produce $620.00 $15.50"
    *   "*For Employees hired or appointed on or after 2/6/2000."
    *   "The following Wage Scale applies to Full-time and Part-time Apprentice Meat Cutters"
    *   Table: "FT & PT Apprentice Meat Cutter"
        *   Headers: "Step", "Months"
        *   Rows: Start 1 $9.50, 3 months 2 3 $10.00, 6 months 3 3 $11.00, 9 months 4 3 $12.00, 12 months 5 3 $13.00, 18 months 6 6 $14.00, 24 months 7 6 cutter rate
    *   "The following Wage Scale applies to full-time overnight Grocery Clerk, Pharmacy Technician, Inventory Control (SRO), Etc., and part-time overnight Grocery Clerk, Pharmacy Technician, Inventory Control (SRO), Etc., hired before March 7, 2004:"
    *   "FT & PT Overnight Grocery Clerk, Pharmacy Tech, Inventory Control (SRO), etc."
    *   Table:
        *   Headers: "Step", "Months"
        *   Rows: Start 1 $9.00, 3 months 2 3 $9.50, 6 months 3 3 $10.00
    *   "-40-"

3.  **Transcribe Right Column:**
    *   Top part (continuation of previous table presumably):
        *   "9 months 4 3 $11.00"
        *   "12 months 5 3 $12.00"
        *   "18 months 6 6 $13.00"
        *   "24 months 7 6 $14.00"
    *   "The following Wage Scale applies to full-time Clerks."
    *   Table: "FT Clerk"
        *   Headers: "Step", "Months"
        *   Rows: Start 1 $8.00, 3 months 2 3 $8.50, 6 months 3 3 $9.00, 9 months 4 3 $9.50, 12 months 5 3 $10.00, 18 months 6 6 $10.50, 24 months 7 6 $11.00, 30 months 8 6 $12.00
    *   "The following Wage Scale applies to part-time Clerks hired prior to March 7, 2004"
    *   Table: "PT Clerk"
        *   Headers: "Step", "Months"
        *   Rows: Start 1 $7.25, 3 months 2 3 $7.50, 6 months 3 3 $7.75, 9 months 4 3 $8.00, 12 months 5 3 $8.25, 18 months 6 6 $8.50, 24 months 7 6 $9.00, 30 months 8 6 $9.50
    *   "The following Wage Scale applies to full-time and part-time Packer/Bagger hired prior to March 7, 2004:"
    *   "-41-"

4.  **Refine formatting:** Ensure tables look like tables or aligned text. The prompt asks for Markdown tables for visible tables. I will convert the structured lists into Markdown tables where appropriate.

    *   Left column top list:
        Assistant Customer Service $620.00 $15.50
        Assistant Bakery $620.00 $15.50
        ...
    *   Left column table 1:
        | Step | Months | |
        |---|---|---|
        | Start | 1 | $9.50 |
        ...
    *   Right column top list:
        9 months 4 3 $11.00
        ...
    *   Right column table 1:
        | Step | Months | |
        |---|---|---|
        | Start | 1 | $8.00 |
        ...

    Wait, looking closely at the top right, it's a continuation of the table on the top left (which isn't fully visible, but the rows align).
    Actually, the top left block is:
    Assistant Customer Service $620.00 $15.50
    ...
    Assistant Produce $620.00 $15.50

    The top right block is:
    9 months 4 3 $11.00
    12 months 5 3 $12.00
    18 months 6 6 $13.00
    24 months 7 6 $14.00

    This looks like a continuation of a table that started on a previous page or is just floating. I will transcribe it exactly as it appears.

    Let's assemble the text.
</think>

Assistant Customer Service $620.00 $15.50
Assistant Bakery $620.00 $15.50
Assistant Deli $620.00 $15.50
Assistant Grocery $620.00 $15.50
Assistant Produce $620.00 $15.50

*For Employees hired or appointed on or after 2/6/2000.

The following Wage Scale applies to Full-time and Part-time Apprentice Meat Cutters

FT & PT Apprentice Meat Cutter
| Step | Months | |
|---|---|---|
| Start | 1 | $9.50 |
| 3 months | 2 | 3 | $10.00 |
| 6 months | 3 | 3 | $11.00 |
| 9 months | 4 | 3 | $12.00 |
| 12 months | 5 | 3 | $13.00 |
| 18 months | 6 | 6 | $14.00 |
| 24 months | 7 | 6 | cutter rate |

The following Wage Scale applies to full-time overnight Grocery Clerk, Pharmacy Technician, Inventory Control (SRO), Etc., and part-time overnight Grocery Clerk, Pharmacy Technician, Inventory Control (SRO), Etc., hired before March 7, 2004:

FT & PT Overnight Grocery Clerk, Pharmacy Tech, Inventory Control (SRO), etc.
| Step | Months | |
|---|---|---|
| Start | 1 | $9.00 |
| 3 months | 2 | 3 | $9.50 |
| 6 months | 3 | 3 | $10.00 |

-40-

9 months 4 3 $11.00
12 months 5 3 $12.00
18 months 6 6 $13.00
24 months 7 6 $14.00

The following Wage Scale applies to full-time Clerks.
FT Clerk
| Step | Months | |
|---|---|---|
| Start | 1 | $8.00 |
| 3 months | 2 | 3 | $8.50 |
| 6 months | 3 | 3 | $9.00 |
| 9 months | 4 | 3 | $9.50 |
| 12 months | 5 | 3 | $10.00 |
| 18 months | 6 | 6 | $10.50 |
| 24 months | 7 | 6 | $11.00 |
| 30 months | 8 | 6 | $12.00 |

The following Wage Scale applies to part-time Clerks hired prior to March 7, 2004
PT Clerk
| Step | Months | |
|---|---|---|
| Start | 1 | $7.25 |
| 3 months | 2 | 3 | $7.50 |
| 6 months | 3 | 3 | $7.75 |
| 9 months | 4 | 3 | $8.00 |
| 12 months | 5 | 3 | $8.25 |
| 18 months | 6 | 6 | $8.50 |
| 24 months | 7 | 6 | $9.00 |
| 30 months | 8 | 6 | $9.50 |

The following Wage Scale applies to full-time and part-time Packer/Bagger hired prior to March 7, 2004:

-41-
