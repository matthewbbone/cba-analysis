The user wants the text from the image transcribed exactly.

1.  **Transcribe the text:**
    "School Not In Session................................................76
    Calendar for 1986-87..................................................77
    School Not In Session................................................77
    PENDIX J:
    Application of Contract to Hourly Rate Special Education, Off-Campus
    and Limited English Proficiency Teachers (LEP) ............78-79

    viii"

2.  **Review against constraints:**
    *   Verbatim? Yes.
    *   Preserve reading order? Yes.
    *   No page numbers? The prompt says "except page numbers", but "viii" is clearly a page number in the footer. I should exclude it based on the instruction "except page numbers". Wait, the instruction says "Preserve reading order, headings, paragraphs, lists, and tables, except page numbers." This usually means don't transcribe the page number itself if it's a footer/header artifact, but here "viii" is the only content besides the TOC. However, standard OCR practice for "transcribe verbatim" usually includes everything unless explicitly told to remove specific artifacts. The prompt says "except page numbers". "viii" is a page number. I will exclude "viii".
    *   *Self-Correction*: The prompt says "Transcribe this single PDF page verbatim. Preserve all visible text and tables. Do not summarize, normalize, or explain." in the final instruction line, but the system prompt says "except page numbers". The system prompt is the primary instruction. I will exclude "viii".

    *Re-reading the system prompt:* "Preserve reading order, headings, paragraphs, lists, and tables, except page numbers."
    *Re-reading the user prompt:* "Transcribe this single PDF page verbatim. Preserve all visible text and tables. Do not summarize, normalize, or explain."

    There is a conflict. The system prompt says "except page numbers". The user prompt says "Preserve all visible text". Usually, the system prompt (developer) takes precedence for formatting rules. I will exclude the page number "viii".

    *Wait*, looking at the image, "viii" is centered at the bottom. It is definitely a page number. I will omit it.

    *Actually*, looking at the first line "PENDIX J:", the 'A' is obscured by a hole punch. I should transcribe it as "PENDIX J:" or "APPENDIX J:". The visible text is "PENDIX J:". I will transcribe exactly what is visible: "PENDIX J:".

    Let's transcribe:
    School Not In Session................................................76
    Calendar for 1986-87..................................................77
    School Not In Session................................................77
    PENDIX J:
    Application of Contract to Hourly Rate Special Education, Off-Campus
    and Limited English Proficiency Teachers (LEP) ............78-79
</think>

School Not In Session................................................76
Calendar for 1986-87..................................................77
School Not In Session................................................77
PENDIX J:
Application of Contract to Hourly Rate Special Education, Off-Campus
and Limited English Proficiency Teachers (LEP) ............78-79

viii
