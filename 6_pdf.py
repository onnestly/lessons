
import logging
from datetime import datetime
import re
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    ListFlowable,
    ListItem,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

logger = logging.getLogger(__name__)

FONT_FAMILY = "ConspectFont"

COLOR_ACCENT = colors.HexColor("#1F4E79")
COLOR_TEXT = colors.HexColor("#222222")
COLOR_MUTED = colors.HexColor("#6B6B6B")
COLOR_DEFINITION_BG = colors.HexColor("#EAF2FB")
COLOR_DEFINITION_LINE = colors.HexColor("#2E75B6")
COLOR_NOTICE_BG = colors.HexColor("#FFF4E0")
COLOR_NOTICE_LINE = colors.HexColor("#E0A030")
COLOR_KEY_BG = colors.HexColor("#F2F7EE")
COLOR_KEY_LINE = colors.HexColor("#5A9A3A")

FONT_CANDIDATES = [
    ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf",
     "C:/Windows/Fonts/ariali.ttf", "C:/Windows/Fonts/arialbi.ttf"),
    ("C:/Windows/Fonts/calibri.ttf", "C:/Windows/Fonts/calibrib.ttf",
     "C:/Windows/Fonts/calibrii.ttf", "C:/Windows/Fonts/calibriz.ttf"),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
     "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf",
     "/usr/share/fonts/truetype/dejavu/DejaVuSans-BoldOblique.ttf"),
    ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
     "/System/Library/Fonts/Supplemental/Arial Italic.ttf",
     "/System/Library/Fonts/Supplemental/Arial Bold Italic.ttf"),
]


class PdfError(Exception):
    def __init__(self, user_message: str, technical_details: str = ""):
        super().__init__(technical_details or user_message)
        self.user_message = user_message


def clean_text(text) -> str:
    text = str(text or "").replace("**", "").replace("__", "").strip()
    text = re.sub(r"^#+\s*", "", text)
    text = escape(text)
    return text.replace("\n", "<br/>")


class PdfRenderer:
    def __init__(self, font_path: str = ""):
        self.font_path = font_path
        self._fonts_ready = False
        self._regular_path = ""
        self.styles = {}


    def check_fonts(self) -> str:
        if self._fonts_ready:
            return self._regular_path

        candidates = []
        if self.font_path:
            candidates.append((self.font_path, "", "", ""))
        candidates += FONT_CANDIDATES

        for regular, bold, italic, bold_italic in candidates:
            if not Path(regular).is_file():
                continue
            try:
                pdfmetrics.registerFont(TTFont(FONT_FAMILY, regular))
                variants = {"-Bold": bold, "-Italic": italic, "-BoldItalic": bold_italic}
                for suffix, path in variants.items():
                    font_file = path if path and Path(path).is_file() else regular
                    pdfmetrics.registerFont(TTFont(FONT_FAMILY + suffix, font_file))
                pdfmetrics.registerFontFamily(
                    FONT_FAMILY,
                    normal=FONT_FAMILY,
                    bold=FONT_FAMILY + "-Bold",
                    italic=FONT_FAMILY + "-Italic",
                    boldItalic=FONT_FAMILY + "-BoldItalic",
                )
            except Exception as error:
                logger.warning("Не удалось подключить шрифт %s: %s", regular, error)
                continue

            self._regular_path = regular
            self._fonts_ready = True
            self._build_styles()
            logger.info("Шрифт для PDF: %s", regular)
            return regular

        raise PdfError(
            "Не найден шрифт с русскими буквами для PDF. Укажите путь к TTF-шрифту в .env (FONT_PATH).",
            "Не найден ни один шрифт из списка FONT_CANDIDATES",
        )

    def _build_styles(self) -> None:
        def style(name, **settings):
            settings.setdefault("fontName", FONT_FAMILY)
            settings.setdefault("textColor", COLOR_TEXT)
            return ParagraphStyle(name, **settings)

        self.styles = {
            "title": style("title", fontSize=22, leading=27, alignment=TA_CENTER,
                           textColor=COLOR_ACCENT, spaceAfter=4),
            "subtitle": style("subtitle", fontSize=11, leading=14, alignment=TA_CENTER,
                              textColor=COLOR_MUTED, spaceAfter=12),
            "meta": style("meta", fontSize=9.5, leading=13),
            "h1": style("h1", fontSize=15, leading=19, textColor=COLOR_ACCENT,
                        spaceBefore=14, spaceAfter=6, keepWithNext=1),
            "h2": style("h2", fontSize=12, leading=15, textColor=COLOR_ACCENT,
                        spaceBefore=8, spaceAfter=4, keepWithNext=1),
            "label": style("label", fontSize=10.5, leading=14, spaceBefore=6, spaceAfter=2, keepWithNext=1),
            "body": style("body", fontSize=11, leading=15.5, alignment=TA_JUSTIFY, spaceAfter=6),
            "bullet": style("bullet", fontSize=11, leading=15),
            "example": style("example", fontSize=10.5, leading=14.5),
            "box": style("box", fontSize=10.5, leading=14.5),
            "toc": style("toc", fontSize=10.5, leading=15, leftIndent=10),
            "small": style("small", fontSize=9, leading=12, textColor=COLOR_MUTED),
        }


    def render(self, conspect, output_path: Path) -> Path:
        output_path = Path(output_path)
        self.check_fonts()

        try:
            story = self._build_story(conspect)
            document = SimpleDocTemplate(
                str(output_path),
                pagesize=A4,
                leftMargin=2.2 * cm,
                rightMargin=2.2 * cm,
                topMargin=2 * cm,
                bottomMargin=2.2 * cm,
                title=conspect.title,
                author="Система формирования конспектов",
                subject=f"Конспект лекции ({conspect.mode_name})",
            )
            document.build(story)
        except Exception as error:
            raise PdfError("Не удалось создать PDF-файл.", repr(error)) from error

        if not output_path.exists() or output_path.stat().st_size == 0:
            raise PdfError("Не удалось создать PDF-файл.", "Файл не появился на диске")
        logger.info("PDF создан: %s (%d КБ)", output_path.name, output_path.stat().st_size // 1024)
        return output_path


    def _build_story(self, conspect) -> list:
        s = self.styles
        story = [
            Paragraph(clean_text(conspect.title), s["title"]),
            Paragraph(f"Дата составления: {datetime.now():%d.%m.%Y}", s["subtitle"]),
        ]
        story.append(HRFlowable(width="100%", thickness=1, color=COLOR_ACCENT, spaceBefore=4, spaceAfter=8))

        if conspect.notice:
            story.append(self._box(clean_text(conspect.notice), COLOR_NOTICE_BG, COLOR_NOTICE_LINE))
            story.append(Spacer(1, 8))

        sections = list(conspect.sections)

        if len(sections) >= 3:
            story.append(Paragraph("Содержание", s["h1"]))
            for number, section in enumerate(sections, start=1):
                story.append(Paragraph(f"{number}. {clean_text(section.title)}", s["toc"]))
            if conspect.conclusions:
                story.append(Paragraph(f"{len(sections) + 1}. Выводы", s["toc"]))
            story.append(Spacer(1, 6))

        for number, section in enumerate(sections, start=1):
            story += self._section(section, number)

        if conspect.conclusions:
            story.append(Paragraph(f"{len(sections) + 1}. Выводы", s["h1"]))
            items = [Paragraph(clean_text(text), s["bullet"]) for text in conspect.conclusions]
            story.append(self._list(items, numbered=True))

        return story

    def _section(self, section, number: int) -> list:
        s = self.styles
        parts = [Paragraph(f"{number}. {clean_text(section.title)}", s["h1"])]

        kinds = sum(bool(x) for x in (
            section.explanation, section.subsections, section.definitions,
            section.examples, section.facts, section.key_points,
        ))
        show_labels = kinds > 1

        for paragraph in re.split(r"\n\s*\n", section.explanation or ""):
            if paragraph.strip():
                parts.append(Paragraph(clean_text(paragraph), s["body"]))

        for sub_number, subsection in enumerate(section.subsections, start=1):
            if subsection.title:
                parts.append(Paragraph(f"{number}.{sub_number}. {clean_text(subsection.title)}", s["h2"]))
            if subsection.text:
                parts.append(Paragraph(clean_text(subsection.text), s["body"]))
            if subsection.points:
                parts.append(self._list([Paragraph(clean_text(p), s["bullet"]) for p in subsection.points]))

        if section.definitions:
            if show_labels:
                parts.append(Paragraph("<b>Определения</b>", s["label"]))
            for definition in section.definitions:
                text = f"<b>{clean_text(definition.term)}</b> — {clean_text(definition.meaning)}"
                parts.append(self._box(text, COLOR_DEFINITION_BG, COLOR_DEFINITION_LINE))
                parts.append(Spacer(1, 4))

        if section.examples:
            if show_labels:
                parts.append(Paragraph("<b>Примеры</b>", s["label"]))
            items = [Paragraph(f"<i>{clean_text(e)}</i>", s["example"]) for e in section.examples]
            parts.append(self._list(items))

        if section.facts:
            if show_labels:
                parts.append(Paragraph("<b>Даты, числа и факты</b>", s["label"]))
            parts.append(self._list([Paragraph(clean_text(f), s["bullet"]) for f in section.facts]))

        if section.key_points:
            items = [Paragraph(clean_text(p), s["bullet"]) for p in section.key_points]
            if show_labels:
                content = [Paragraph("<b>Главное</b>", s["box"]), self._list(items)]
                parts.append(Spacer(1, 4))
                parts.append(self._box(content, COLOR_KEY_BG, COLOR_KEY_LINE))
            else:
                parts.append(self._list(items))

        if len(parts) > 1:
            return [KeepTogether(parts[:2])] + parts[2:]
        return parts

    def _list(self, items: list, numbered: bool = False) -> ListFlowable:
        return ListFlowable(
            [ListItem(item, leftIndent=16) for item in items],
            bulletType="1" if numbered else "bullet",
            start=None if numbered else "•",
            bulletFontName=FONT_FAMILY,
            bulletFontSize=10,
            leftIndent=16,
            spaceBefore=2,
            spaceAfter=6,
        )

    def _box(self, content, background, line_color) -> Table:
        if isinstance(content, str):
            content = Paragraph(content, self.styles["box"])
        table = Table([[content]], colWidths=["100%"])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), background),
            ("LINEBEFORE", (0, 0), (0, -1), 3, line_color),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        return table
