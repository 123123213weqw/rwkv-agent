#!/usr/bin/env python3
"""Build the English Track 2 Project Specification PDF from Markdown."""
from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Flowable,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

BG = colors.HexColor("#090B0E")
PANEL = colors.HexColor("#141820")
INK = colors.HexColor("#161A22")
SOFT = colors.HexColor("#5E6672")
LINE = colors.HexColor("#D9DDE4")
VIOLET = colors.HexColor("#7659D6")
CYAN = colors.HexColor("#148A9A")
GREEN = colors.HexColor("#148A58")
LIGHT = colors.HexColor("#F4F5F8")


def register_fonts() -> tuple[str, str, str]:
    candidates = [
        (
            Path("/Library/Fonts/Arial Unicode.ttf"),
            Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
            Path("/System/Library/Fonts/Supplemental/Arial Italic.ttf"),
        ),
        (
            Path("/System/Library/Fonts/Supplemental/Verdana.ttf"),
            Path("/System/Library/Fonts/Supplemental/Verdana Bold.ttf"),
            Path("/System/Library/Fonts/Supplemental/Verdana Italic.ttf"),
        ),
    ]
    for regular, bold, italic in candidates:
        if regular.exists() and bold.exists() and italic.exists():
            pdfmetrics.registerFont(TTFont("RWKV-Regular", regular))
            pdfmetrics.registerFont(TTFont("RWKV-Bold", bold))
            pdfmetrics.registerFont(TTFont("RWKV-Italic", italic))
            return "RWKV-Regular", "RWKV-Bold", "RWKV-Italic"
    return "Helvetica", "Helvetica-Bold", "Helvetica-Oblique"


FONT, FONT_BOLD, FONT_ITALIC = register_fonts()


def inline_markup(text: str) -> str:
    escaped = html.escape(text.strip())
    escaped = re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", rf'<font name="{FONT_BOLD}">\1</font>', escaped)
    escaped = re.sub(r"\[([^]]+)\]\(([^)]+)\)", r'<link href="\2" color="#148A9A">\1</link>', escaped)
    return escaped


class Cover(Flowable):
    def __init__(self) -> None:
        super().__init__()
        self.width, self.height = A4

    def wrap(self, avail_width, avail_height):
        return avail_width, avail_height

    def draw(self):
        c = self.canv
        width, height = A4
        c.saveState()
        c.setFillColor(BG)
        c.rect(-22 * mm, -22 * mm, width + 44 * mm, height + 44 * mm, fill=1, stroke=0)
        c.setFillColor(colors.HexColor("#181323"))
        c.circle(width - 20 * mm, height - 25 * mm, 65 * mm, fill=1, stroke=0)
        c.setStrokeColor(colors.HexColor("#33274F"))
        c.setLineWidth(1)
        for radius in (20, 32, 45):
            c.circle(width - 28 * mm, height - 37 * mm, radius * mm, fill=0, stroke=1)
        c.setFillColor(colors.HexColor("#A98AFF"))
        c.setFont(FONT_BOLD, 10)
        c.drawString(0, height - 28 * mm, "AMD AI DEVMASTER HACKATHON 2026  /  TRACK 2")
        c.setFillColor(colors.white)
        c.setFont(FONT_BOLD, 34)
        c.drawString(0, height - 78 * mm, "RWKV")
        c.setFillColor(colors.HexColor("#A98AFF"))
        c.drawString(0, height - 94 * mm, "STATE AGENT")
        c.setFillColor(colors.HexColor("#C9CFD8"))
        c.setFont(FONT, 16)
        c.drawString(0, height - 118 * mm, "Project Specification")
        c.setFillColor(colors.HexColor("#8B95A3"))
        c.setFont(FONT, 10)
        lines = [
            "Local private Agent  /  recurrent memory  /  strict tools",
            "100 independent States  /  AMD Radeon ROCm",
        ]
        for idx, line in enumerate(lines):
            c.drawString(0, height - (142 + idx * 7) * mm, line)
        c.setFillColor(colors.HexColor("#65D8E6"))
        c.rect(0, 36 * mm, 30 * mm, 1.2 * mm, fill=1, stroke=0)
        c.setFillColor(colors.HexColor("#AAB2BE"))
        c.setFont(FONT, 9)
        c.drawString(0, 26 * mm, "RWKV-7 G1I Preview4922 13.3B  /  gfx1100  /  ROCm 7.2.1")
        c.drawString(0, 18 * mm, "Candidate 0.3.0-beta.1  /  5 August 2026")
        c.restoreState()


class ArchitectureDiagram(Flowable):
    def __init__(self, width: float = 165 * mm, height: float = 98 * mm) -> None:
        super().__init__()
        self.width = width
        self.height = height

    def wrap(self, avail_width, avail_height):
        return min(self.width, avail_width), self.height

    def _box(self, c, x, y, w, h, title, subtitle, accent):
        c.setFillColor(colors.white)
        c.setStrokeColor(LINE)
        c.roundRect(x, y, w, h, 6, fill=1, stroke=1)
        c.setFillColor(accent)
        c.rect(x, y, 4, h, fill=1, stroke=0)
        c.setFillColor(INK)
        c.setFont(FONT_BOLD, 8)
        c.drawString(x + 10, y + h - 15, title)
        c.setFillColor(SOFT)
        c.setFont(FONT, 6.7)
        c.drawString(x + 10, y + 8, subtitle)

    def draw(self):
        c = self.canv
        w = self.width
        c.saveState()
        c.setFillColor(LIGHT)
        c.roundRect(0, 0, w, self.height, 8, fill=1, stroke=0)
        bw, bh = 46 * mm, 20 * mm
        x1, x2, x3 = 7 * mm, 59.5 * mm, 112 * mm
        y1, y2, y3 = 70 * mm, 39 * mm, 8 * mm
        self._box(c, x1, y1, bw, bh, "LOCAL CLIENTS", "Web UI / Rust CLI / tasks", VIOLET)
        self._box(c, x2, y1, bw, bh, "RUST CONTROLLER", "sessions / loop / policy / trace", CYAN)
        self._box(c, x3, y1, bw, bh, "SEMANTIC GATE", "prewarmed recurrent root", GREEN)
        self._box(c, x2, y2, bw, bh, "RWKV ROCm SIDECAR", "prefill / continue / true stream", VIOLET)
        self._box(c, x1, y3, bw, bh, "LOCAL TOOLS", "command / knowledge / long text", CYAN)
        self._box(c, x2, y3, bw, bh, "STATE SCHEDULER", "132 capacity / physical batch 32", GREEN)
        self._box(c, x3, y3, bw, bh, "AMD RADEON", "gfx1100 / ROCm 7.2.1", VIOLET)
        c.setStrokeColor(colors.HexColor("#8C95A3"))
        c.setLineWidth(1.2)
        arrows = [
            (x1 + bw, y1 + bh / 2, x2, y1 + bh / 2),
            (x2 + bw, y1 + bh / 2, x3, y1 + bh / 2),
            (x3 + bw / 2, y1, x2 + bw / 2, y2 + bh),
            (x2, y2 + bh / 2, x1 + bw, y3 + bh / 2),
            (x2 + bw / 2, y2, x2 + bw / 2, y3 + bh),
            (x2 + bw, y3 + bh / 2, x3, y3 + bh / 2),
        ]
        for ax, ay, bx, by in arrows:
            c.line(ax, ay, bx, by)
            c.setFillColor(colors.HexColor("#8C95A3"))
            c.circle(bx, by, 1.7, fill=1, stroke=0)
        c.restoreState()


def page_chrome(canvas, doc):
    if doc.page == 1:
        return
    canvas.saveState()
    width, height = A4
    canvas.setStrokeColor(LINE)
    canvas.setLineWidth(0.6)
    canvas.line(20 * mm, height - 15 * mm, width - 20 * mm, height - 15 * mm)
    canvas.setFont(FONT_BOLD, 7)
    canvas.setFillColor(VIOLET)
    canvas.drawString(20 * mm, height - 11.5 * mm, "RWKV STATE AGENT")
    canvas.setFont(FONT, 7)
    canvas.setFillColor(SOFT)
    canvas.drawRightString(width - 20 * mm, height - 11.5 * mm, "PROJECT SPECIFICATION  /  TRACK 2")
    canvas.line(20 * mm, 14 * mm, width - 20 * mm, 14 * mm)
    canvas.drawString(20 * mm, 9.5 * mm, "AMD Radeon gfx1100  /  ROCm 7.2.1  /  Local private Agent")
    canvas.drawRightString(width - 20 * mm, 9.5 * mm, f"{doc.page:02d}")
    canvas.restoreState()


def styles():
    base = getSampleStyleSheet()
    return {
        "h1": ParagraphStyle("H1", parent=base["Heading1"], fontName=FONT_BOLD, fontSize=23, leading=27, textColor=INK, spaceBefore=4 * mm, spaceAfter=4 * mm),
        "h2": ParagraphStyle("H2", parent=base["Heading2"], fontName=FONT_BOLD, fontSize=16, leading=20, textColor=VIOLET, spaceBefore=5 * mm, spaceAfter=3 * mm, keepWithNext=True),
        "h3": ParagraphStyle("H3", parent=base["Heading3"], fontName=FONT_BOLD, fontSize=11, leading=14, textColor=CYAN, spaceBefore=4 * mm, spaceAfter=2 * mm, keepWithNext=True),
        "body": ParagraphStyle("Body", parent=base["BodyText"], fontName=FONT, fontSize=8.7, leading=13.2, textColor=INK, spaceAfter=2.3 * mm),
        "bullet": ParagraphStyle("Bullet", parent=base["BodyText"], fontName=FONT, fontSize=8.4, leading=12.5, textColor=INK, leftIndent=4 * mm),
        "code": ParagraphStyle("Code", parent=base["Code"], fontName="Courier", fontSize=6.7, leading=9.2, textColor=colors.HexColor("#E3E7EB"), backColor=PANEL, borderPadding=8, spaceBefore=2 * mm, spaceAfter=3 * mm),
        "quote": ParagraphStyle("Quote", parent=base["BodyText"], fontName=FONT_ITALIC, fontSize=9, leading=13, textColor=SOFT, leftIndent=6 * mm, borderColor=VIOLET, borderWidth=1.4, borderPadding=6, spaceAfter=3 * mm),
        "table": ParagraphStyle("Table", parent=base["BodyText"], fontName=FONT, fontSize=6.8, leading=9, textColor=INK),
        "table_head": ParagraphStyle("TableHead", parent=base["BodyText"], fontName=FONT_BOLD, fontSize=6.8, leading=9, textColor=colors.white),
    }


def table_flowable(rows: list[list[str]], st) -> Table:
    width = 170 * mm
    cols = max(len(row) for row in rows)
    normalized = [row + [""] * (cols - len(row)) for row in rows]
    data = []
    for ridx, row in enumerate(normalized):
        style = st["table_head"] if ridx == 0 else st["table"]
        data.append([Paragraph(inline_markup(cell), style) for cell in row])
    longest = [max(len(row[idx]) for row in normalized) for idx in range(cols)]
    total = sum(max(8, value) for value in longest)
    widths = [width * max(8, value) / total for value in longest]
    table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), VIOLET),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 1), (-1, -1), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.45, LINE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
    ]))
    return table


def parse_markdown(path: Path):
    st = styles()
    lines = path.read_text(encoding="utf-8").splitlines()
    story = [Cover(), PageBreak()]
    paragraph: list[str] = []
    bullets: list[str] = []
    code: list[str] = []
    table: list[list[str]] = []
    in_code = False
    code_kind = ""
    seen_main = False
    major_page_sections = {"4.", "7.", "8.", "9.", "10.", "11.", "14."}

    def flush_paragraph():
        nonlocal paragraph
        if paragraph:
            text = " ".join(item.strip() for item in paragraph)
            story.append(Paragraph(inline_markup(text), st["body"]))
            paragraph = []

    def flush_bullets():
        nonlocal bullets
        if bullets:
            items = [ListItem(Paragraph(inline_markup(item), st["bullet"]), leftIndent=3 * mm) for item in bullets]
            story.append(ListFlowable(items, bulletType="bullet", start="circle", leftIndent=7 * mm, bulletFontName=FONT, bulletFontSize=6, spaceAfter=2 * mm))
            bullets = []

    def flush_table():
        nonlocal table
        if table:
            cleaned = [row for row in table if not all(re.fullmatch(r":?-{3,}:?", cell.strip()) for cell in row)]
            if len(cleaned) >= 2:
                story.extend([table_flowable(cleaned, st), Spacer(1, 3 * mm)])
            table = []

    for raw in lines:
        line = raw.rstrip()
        if line.startswith("```"):
            flush_paragraph()
            flush_bullets()
            flush_table()
            if in_code:
                if code_kind == "mermaid":
                    story.extend([ArchitectureDiagram(), Spacer(1, 3 * mm)])
                elif code:
                    story.append(Preformatted("\n".join(code), st["code"], maxLineLength=105))
                code, code_kind, in_code = [], "", False
            else:
                in_code = True
                code_kind = line[3:].strip().lower()
            continue
        if in_code:
            code.append(line)
            continue
        if line.startswith("# "):
            continue
        if line.startswith("**AMD AI") or line.startswith("**Track 2") or line.startswith("**Candidate") or line.startswith("**Date"):
            continue
        if line.startswith("## "):
            flush_paragraph()
            flush_bullets()
            flush_table()
            title = line[3:].strip()
            prefix = title.split(maxsplit=1)[0]
            if seen_main and prefix in major_page_sections:
                story.append(PageBreak())
            story.append(Paragraph(inline_markup(title), st["h1"]))
            seen_main = True
            continue
        if line.startswith("### "):
            flush_paragraph()
            flush_bullets()
            flush_table()
            story.append(Paragraph(inline_markup(line[4:]), st["h2"]))
            continue
        if line.startswith("#### "):
            flush_paragraph()
            flush_bullets()
            flush_table()
            story.append(Paragraph(inline_markup(line[5:]), st["h3"]))
            continue
        if line.startswith("> "):
            flush_paragraph()
            flush_bullets()
            flush_table()
            story.append(Paragraph(inline_markup(line[2:]), st["quote"]))
            continue
        if line.startswith("|") and line.endswith("|"):
            flush_paragraph()
            flush_bullets()
            table.append([cell.strip() for cell in line.strip("|").split("|")])
            continue
        if table:
            flush_table()
        bullet = re.match(r"^[-*]\s+(.+)$", line)
        numbered = re.match(r"^\d+\.\s+(.+)$", line)
        if bullet or numbered:
            flush_paragraph()
            bullets.append((bullet or numbered).group(1))
            continue
        if bullets:
            flush_bullets()
        if not line.strip() or line.strip() == "---":
            flush_paragraph()
            continue
        paragraph.append(line)

    flush_paragraph()
    flush_bullets()
    flush_table()
    return story


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="docs/PROJECT_SPECIFICATION.md")
    parser.add_argument("--output", default="submission/RWKV_State_Agent_Project_Specification.pdf")
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=21 * mm,
        bottomMargin=19 * mm,
        title="RWKV State Agent - Project Specification",
        author="RWKV State Agent",
        subject="AMD AI DevMaster Hackathon 2026 - Track 2",
    )
    doc.build(parse_markdown(Path(args.input)), onFirstPage=page_chrome, onLaterPages=page_chrome)
    print(output.resolve())


if __name__ == "__main__":
    main()
