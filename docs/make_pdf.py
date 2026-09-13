"""Generate docs/MARA_MASTER_GUIDE.pdf from docs/MARA_MASTER_GUIDE.md.

One-off utility (fpdf2, pure Python). Usage:
    python docs/make_pdf.py
Re-run it whenever the markdown guide changes.
"""
from __future__ import annotations

import re
from pathlib import Path

from fpdf import FPDF

ROOT = Path(__file__).resolve().parent.parent
MD_PATH = ROOT / "docs" / "MARA_MASTER_GUIDE.md"
PDF_PATH = ROOT / "docs" / "MARA_MASTER_GUIDE.pdf"


class GuidePDF(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("helvetica", "I", 8)
            self.set_text_color(120, 120, 120)
            self.cell(0, 6, "MARA - Master Build & Career Guide", align="C")
            self.ln(10)

    def footer(self):
        self.set_y(-14)
        self.set_font("helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, f"Page {self.page_no()}/{{nb}}", align="C")


# fpdf's built-in fonts are latin-1 only; map common Unicode in the guide
# to safe ASCII equivalents so the PDF generates on any machine.
_ASCII_MAP = {
    "→": "->", "✓": "v", "✅": "[OK]", "⬜": "[ ]", "⚠️": "[!]", "⚠": "[!]",
    "—": " - ", "–": "-", "·": "*", "•": "*", "≥": ">=", "≤": "<=",
    "\u00a0": " ", "\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'",
    "…": "...", "×": "x",
}


def clean(text: str) -> str:
    """Strip markdown emphasis and transliterate Unicode to latin-1-safe ASCII."""
    for src, dst in _ASCII_MAP.items():
        text = text.replace(src, dst)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"(?<!\w)\*(.+?)\*(?!\w)", r"\1", text)
    text = text.replace("`", "")
    return text.encode("latin-1", "replace").decode("latin-1")


def add_table(pdf: GuidePDF, rows: list[list[str]]):
    """Render a markdown table as a simple bordered grid."""
    pdf.set_font("helvetica", "", 8.5)
    col_count = max(len(r) for r in rows)
    usable = pdf.w - pdf.l_margin - pdf.r_margin
    # Give the first column of each table extra width (labels tend to be long).
    widths = [usable * 0.38] + [usable * (0.62 / (col_count - 1))] * (col_count - 1) if col_count > 1 else [usable]
    for i, row in enumerate(rows):
        if pdf.get_y() + 12 > pdf.page_break_trigger:
            pdf.add_page()
        h = 7
        x = pdf.l_margin
        for j, cell in enumerate(row[:col_count]):
            pdf.set_xy(x, pdf.get_y())
            pdf.set_draw_color(190, 190, 190)
            if i == 0:
                pdf.set_font("helvetica", "B", 8.5)
                pdf.set_fill_color(235, 235, 235)
            else:
                pdf.set_font("helvetica", "", 8.5)
                pdf.set_fill_color(255, 255, 255)
            pdf.cell(widths[j], h, " " + cell, border=1, fill=True)
            x += widths[j]
        pdf.ln(h)
    pdf.ln(3)


def render(pdf: GuidePDF, md: str):
    lines = md.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]

        # Tables
        if line.strip().startswith("|") and i + 1 < len(lines) and set(lines[i + 1].replace("|", "").replace("-", "").strip()) <= {":"}:
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                rows.append([clean(c) for c in cells])
                i += 1
            add_table(pdf, rows[1:])  # drop the separator row
            continue

        stripped = line.strip()
        if stripped.startswith("# "):
            pdf.set_font("helvetica", "B", 19)
            pdf.set_text_color(20, 20, 60)
            pdf.multi_cell(0, 9, "\n" + clean(stripped[2:]))
            pdf.ln(2)
        elif stripped.startswith("## "):
            if pdf.get_y() + 30 > pdf.page_break_trigger:
                pdf.add_page()
            pdf.ln(3)
            pdf.set_font("helvetica", "B", 13.5)
            pdf.set_text_color(20, 20, 60)
            pdf.multi_cell(0, 7.5, clean(stripped[3:]))
            pdf.set_draw_color(20, 20, 60)
            pdf.set_line_width(0.4)
            pdf.line(pdf.l_margin, pdf.get_y() + 0.5, pdf.w - pdf.r_margin, pdf.get_y() + 0.5)
            pdf.ln(4)
        elif stripped.startswith("### "):
            pdf.ln(2)
            pdf.set_font("helvetica", "B", 11)
            pdf.set_text_color(40, 40, 90)
            pdf.multi_cell(0, 6.5, clean(stripped[4:]))
            pdf.ln(1.5)
        elif stripped.startswith("> "):
            pdf.set_font("helvetica", "I", 9.5)
            pdf.set_text_color(90, 90, 90)
            pdf.set_x(pdf.l_margin + 4)
            pdf.multi_cell(0, 5.2, clean(stripped[2:]))
            pdf.ln(1.5)
        elif stripped in ("---", "***", "___"):
            pdf.ln(2)
            pdf.set_draw_color(200, 200, 200)
            pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
            pdf.ln(4)
        elif re.match(r"^\d+\.\s", stripped):
            pdf.set_font("helvetica", "", 9.8)
            pdf.set_text_color(30, 30, 30)
            pdf.set_x(pdf.l_margin + 4)
            pdf.multi_cell(0, 5.2, clean(stripped))
            pdf.ln(0.8)
        elif stripped.startswith(("- ", "* ")):
            pdf.set_font("helvetica", "", 9.8)
            pdf.set_text_color(30, 30, 30)
            x = pdf.l_margin + 5
            pdf.set_x(x)
            pdf.cell(4, 5.2, "*")
            pdf.set_x(x + 4)
            pdf.multi_cell(0, 5.2, clean(stripped[2:]))
            pdf.ln(0.6)
        elif stripped:
            pdf.set_font("helvetica", "", 9.8)
            pdf.set_text_color(30, 30, 30)
            pdf.multi_cell(0, 5.2, clean(stripped))
            pdf.ln(1.2)
        else:
            pdf.ln(1.4)
        i += 1


def main():
    pdf = GuidePDF(orientation="P", unit="mm", format="A4")
    pdf.alias_nb_pages()
    pdf.set_margins(18, 16, 18)
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()
    render(pdf, MD_PATH.read_text(encoding="utf-8"))
    pdf.output(PDF_PATH)
    print(f"PDF written: {PDF_PATH} ({PDF_PATH.stat().st_size / 1024:.0f} KB, {pdf.page_no()} pages)")


if __name__ == "__main__":
    main()
