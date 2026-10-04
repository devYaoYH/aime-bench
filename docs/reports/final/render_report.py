"""Render the three-page report or illustrated evidence packet with the bundled runtime."""
from pathlib import Path
import argparse
import html
import re
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from pypdf import PdfReader, PdfWriter
from PIL import Image as PILImage

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "report.md"
OUTPUT = ROOT / "output/pdf/callosum-speedrun-report.pdf"
FONT_DIR = Path("/System/Library/Fonts/Supplemental")
for name, filename in [
    ("ReportSerif", "Georgia.ttf"), ("ReportSerif-Bold", "Georgia Bold.ttf"),
    ("ReportSerif-Italic", "Georgia Italic.ttf"),
    ("ReportSerif-BoldItalic", "Georgia Bold Italic.ttf"),
    ("ReportSans", "Arial.ttf"), ("ReportSans-Bold", "Arial Bold.ttf"),
    ("ReportMono", "Courier New.ttf"),
]:
    pdfmetrics.registerFont(TTFont(name, str(FONT_DIR / filename)))
pdfmetrics.registerFontFamily("ReportSerif", normal="ReportSerif", bold="ReportSerif-Bold",
                              italic="ReportSerif-Italic", boldItalic="ReportSerif-BoldItalic")
pdfmetrics.registerFontFamily("ReportSans", normal="ReportSans", bold="ReportSans-Bold")

INK = colors.HexColor("#182536")
ACCENT = colors.HexColor("#245f73")
MUTED = colors.HexColor("#5b6875")
BODY = ParagraphStyle("body", fontName="ReportSerif", fontSize=10.1, leading=13.2,
                      textColor=INK, spaceAfter=6.8, alignment=TA_LEFT)
TITLE = ParagraphStyle("title", fontName="ReportSans-Bold", fontSize=23, leading=27,
                       textColor=INK, spaceAfter=8)
SUBTITLE = ParagraphStyle("subtitle", fontName="ReportSans", fontSize=9.2, leading=12,
                          textColor=MUTED, spaceAfter=13)
H2 = ParagraphStyle("h2", fontName="ReportSans-Bold", fontSize=13.5, leading=17,
                    textColor=ACCENT, spaceBefore=3, spaceAfter=9, keepWithNext=True)
H3 = ParagraphStyle("h3", fontName="ReportSans-Bold", fontSize=10.4, leading=13,
                    textColor=ACCENT, spaceBefore=5, spaceAfter=5, keepWithNext=True)
CELL = ParagraphStyle("cell", fontName="ReportSans", fontSize=8.45, leading=10.8,
                      textColor=INK)
HEAD_CELL = ParagraphStyle("headcell", parent=CELL, fontName="ReportSans-Bold",
                           textColor=colors.white)
REF = ParagraphStyle("reference", fontName="ReportSans", fontSize=7.45, leading=9.2,
                     textColor=MUTED, spaceAfter=2.5, leftIndent=0)
NOTE = ParagraphStyle("note", parent=BODY, fontSize=8.1, leading=10.2, spaceAfter=5)
CAPTION = ParagraphStyle("caption", parent=NOTE, fontName="ReportSans", fontSize=8,
                         leading=10, spaceBefore=4, spaceAfter=8)
DOCKET_BODY = ParagraphStyle("docketbody", parent=BODY, fontSize=9.6, leading=12.6,
                            spaceAfter=4)
DOCKET_H3 = ParagraphStyle("docketh3", parent=H3, fontSize=10.4, leading=13,
                          spaceBefore=5, spaceAfter=4)


def inline(text):
    """Translate the Markdown inline subset used by the report, keeping evidence links."""
    pattern = re.compile(r"\[([^\]]+)\]\(([^)]+)\)|\*\*(.+?)\*\*|`([^`]+)`")
    result, pos = [], 0
    for match in pattern.finditer(text):
        result.append(html.escape(text[pos:match.start()]))
        if match.group(1) is not None:
            dest = match.group(2)
            if not re.match(r"https?://|#", dest):
                filename, separator, fragment = dest.partition("#")
                path = (ROOT / filename).resolve()
                if not path.exists():
                    raise FileNotFoundError(path)
                dest = path.as_uri() + (separator + fragment if separator else "")
            result.append(f'<link href="{html.escape(dest, quote=True)}" color="#245f73">'
                          f'{html.escape(match.group(1))}</link>')
        elif match.group(3) is not None:
            result.append(f"<b>{html.escape(match.group(3))}</b>")
        else:
            result.append(f'<font name="ReportMono" size="9.2">{html.escape(match.group(4))}</font>')
        pos = match.end()
    result.append(html.escape(text[pos:]))
    return "".join(result)


def story_from_markdown(source):
    lines = source.splitlines()
    story = []
    i = 0
    references = False
    docket = False
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if re.fullmatch(r'<a id="(?:e|a)\d+"></a>', line):
            # The following section heading supplies the equivalent PDF destination.
            i += 1
            continue
        if line == "<!-- pagebreak -->":
            story.append(PageBreak())
            i += 1
            continue
        if line.startswith("# "):
            story.append(Paragraph(inline(line[2:]), TITLE))
        elif line.startswith("## "):
            heading = line[3:]
            docket = heading.startswith("E6.")
            references = False
            key_match = re.match(r"(E\d+|A\d+)\.", heading)
            key = key_match.group(1).lower() if key_match else None
            label = (f'<a name="{key}"/>' if key else '') + inline(heading)
            p = Paragraph(label, H2)
            p.section_key = key
            p.section_label = heading
            story.append(p)
        elif line.startswith("### "):
            references = line[4:] == "Evidence index"
            story.append(Paragraph(inline(line[4:]), DOCKET_H3 if docket else H3))
        elif line.startswith("!["):
            match = re.fullmatch(r"!\[([^\]]*)\]\(([^)]+)\)", line)
            if not match:
                raise ValueError(f"Invalid image line: {line}")
            image_path = ROOT / match.group(2)
            with PILImage.open(image_path) as im:
                w, h = im.size
            scale = min(512 / w, 400 / h)
            story.append(Image(str(image_path), width=w * scale, height=h * scale,
                               hAlign="LEFT"))
            story.append(Paragraph(inline(match.group(1)), CAPTION))
        elif line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [part.strip() for part in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", cell) for cell in cells):
                    style = HEAD_CELL if not rows else CELL
                    rows.append([Paragraph(inline(cell), style) for cell in cells])
                i += 1
            table_head = rows[0][0].getPlainText()
            col_widths = {
                "Interview question": [199, 270, 43],
                "Improvement target": [116, 193, 203],
                "Initial sweep setting": [260, 126, 126],
                "Matched NVFP4 30 x 1 / 16K comparison": [260, 126, 126],
                "Completed comparison": [237, 188, 87],
                "Metric": [153, 67, 67, 225],
                "Five-trial policy": [230, 210, 72],
            }.get(table_head)
            if col_widths is None or len(col_widths) != len(rows[0]):
                col_widths = [217, 90, 205] if len(rows[0]) == 3 else [512 / len(rows[0])] * len(rows[0])
            if table_head == "Seed":
                col_widths = [65] + [(512-65)/(len(rows[0])-1)] * (len(rows[0])-1)
            table = Table(rows, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                 [colors.HexColor("#eff4f6"), colors.HexColor("#f8fafb")]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LINEBELOW", (0, -1), (-1, -1), 0.5, colors.HexColor("#cad6dd")),
            ]))
            story.extend([Spacer(1, 2), table, Spacer(1, 9)])
            continue
        elif line.startswith("- "):
            story.append(Paragraph(inline(line[2:]), REF if references else BODY))
        else:
            paragraph = [line]
            while i + 1 < len(lines) and lines[i + 1].strip():
                if lines[i + 1].strip().startswith(("#", "|", "- ", "<!--")):
                    break
                i += 1
                paragraph.append(lines[i].strip())
            style = SUBTITLE if line.startswith("**Callosum") else NOTE if references else DOCKET_BODY if docket else BODY
            story.append(Paragraph(inline(" ".join(paragraph)), style))
        i += 1
    return story


def footer(c, doc):
    c.saveState()
    if doc.page > 1:
        c.setFont("ReportSans", 7.4)
        c.setFillColor(MUTED)
        c.drawString(44, 765, "CALLOSUM  /  VIBETHINKER EVIDENCE" if getattr(doc, 'evidence', False)
                     else "CALLOSUM  /  VERIFIED MATH SPEEDRUN")
    c.setStrokeColor(colors.HexColor("#d5dfe5"))
    c.setLineWidth(0.5)
    c.line(44, 33, 568, 33)
    c.setFont("ReportSans", 7.4)
    c.setFillColor(MUTED)
    c.drawString(44, 21, "4 October 2026  |  Measured runs and separately labeled replays")
    c.drawRightString(568, 21, str(doc.page))
    c.restoreState()


class ReportDocument(SimpleDocTemplate):
    def afterFlowable(self, flowable):
        key = getattr(flowable, 'section_key', None)
        if key:
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(flowable.section_label, key, level=0, closed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', action='store_true', help='Render the illustrated evidence packet')
    args = parser.parse_args()
    source = ROOT / 'evidence.md' if args.evidence else SOURCE
    output = ROOT / 'output/pdf/callosum-evidence-packet.pdf' if args.evidence else OUTPUT
    output.parent.mkdir(parents=True, exist_ok=True)
    doc = ReportDocument(str(output), pagesize=letter, rightMargin=44, leftMargin=44,
                         topMargin=43, bottomMargin=43,
                         title="VibeThinker evidence packet" if args.evidence else "Faster verified math answers",
                         author="Callosum speedrun experiments")
    doc.evidence = args.evidence
    doc.build(story_from_markdown(source.read_text()), onFirstPage=footer, onLaterPages=footer)
    reader = PdfReader(output)
    if args.evidence:
        # ReportLab resolves internal bookmarks but does not export a PDF name
        # tree. External report links use #nameddest, so publish those names.
        writer = PdfWriter(clone_from=reader)
        for entry in reader.outline:
            key = re.match(r"(E\d+|A\d+)\.", entry.title)
            if key:
                writer.add_named_destination(key.group(1).lower(), reader.get_destination_page_number(entry))
        temporary = output.with_suffix('.tmp.pdf')
        writer.write(temporary)
        temporary.replace(output)
        reader = PdfReader(output)
    print(f"Rendered {output}: {len(reader.pages)} pages")
    for num, page in enumerate(reader.pages, 1):
        text = page.extract_text()
        print(f"Page {num}: {len(text.split())} words; {len(page.get('/Annots', []))} links")
    if not args.evidence and len(reader.pages) > 3:
        raise SystemExit("The report exceeds the three-page limit; revise before delivery.")


if __name__ == '__main__':
    main()
