
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

MODE_NAMES = {
    "detailed": "Подробный",
    "normal": "Обычный",
    "short": "Краткий",
    "topic": "По теме",
}

MODE_EXPLANATIONS = {
    "detailed": "практически вся информация лекции, структурированная по разделам",
    "normal": "основные мысли, определения, важные факты и примеры",
    "short": "только ключевые тезисы и то, что необходимо запомнить",
    "topic": "только информация, связанная с выбранной темой",
}


class ConspectError(Exception):
    def __init__(self, user_message: str, technical_details: str = ""):
        super().__init__(technical_details or user_message)
        self.user_message = user_message


@dataclass
class Definition:
    term: str
    meaning: str


@dataclass
class Subsection:
    title: str
    text: str = ""
    points: list = field(default_factory=list)


@dataclass
class Section:
    title: str
    explanation: str = ""
    subsections: list = field(default_factory=list)
    definitions: list = field(default_factory=list)
    examples: list = field(default_factory=list)
    facts: list = field(default_factory=list)
    key_points: list = field(default_factory=list)

    def is_empty(self) -> bool:
        return not (
            self.explanation or self.subsections or self.definitions
            or self.examples or self.facts or self.key_points
        )


@dataclass
class Conspect:
    title: str
    mode: str
    mode_name: str
    mode_explanation: str
    subject: str = ""
    focus_topic: str = ""
    sections: list = field(default_factory=list)
    conclusions: list = field(default_factory=list)
    notice: str = ""
    stats: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


STRING_LIST = {"type": "array", "items": {"type": "string"}}

SECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "explanation": {"type": "string"},
        "subsections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "text": {"type": "string"},
                    "points": STRING_LIST,
                },
                "required": ["title", "text", "points"],
            },
        },
        "definitions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "term": {"type": "string"},
                    "meaning": {"type": "string"},
                },
                "required": ["term", "meaning"],
            },
        },
        "examples": STRING_LIST,
        "facts": STRING_LIST,
        "key_points": STRING_LIST,
    },
    "required": ["title", "explanation", "subsections", "definitions", "examples", "facts", "key_points"],
}

SECTIONS_SCHEMA = {
    "type": "object",
    "properties": {"sections": {"type": "array", "items": SECTION_SCHEMA}},
    "required": ["sections"],
}

SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "conclusions": STRING_LIST,
    },
    "required": ["title", "conclusions"],
}

SHORT_SCHEMA = {
    "type": "object",
    "properties": {
        "theses": STRING_LIST,
        "remember": STRING_LIST,
    },
    "required": ["theses", "remember"],
}


COMMON_RULES = (
    "Правила:\n"
    "- используй ТОЛЬКО информацию из текста, ничего не выдумывай и не добавляй от себя;\n"
    "- пиши грамотным литературным русским языком, без разговорных слов и слов-паразитов;\n"
    "- исправляй очевидные ошибки распознавания речи, если смысл понятен из контекста;\n"
    "- сохраняй все числа, даты, имена, формулы и термины точно;\n"
    "- если какого-то поля в тексте нет, оставь пустую строку или пустой список;\n"
    "- отвечай только JSON."
)

EXTRACT_SYSTEM_PROMPT = (
    "Ты — опытный учитель. По тексту лекции ты составляешь ПОДРОБНЫЙ структурированный конспект, "
    "в котором сохраняется практически вся полезная информация.\n\n"
    "Разбей текст на логические разделы (обычно 1–4 раздела на часть текста). Для каждого раздела заполни:\n"
    "- title: короткий заголовок раздела;\n"
    "- explanation: связное объяснение материала раздела своими словами (3–8 предложений);\n"
    "- subsections: подразделы, если в разделе есть отдельные подтемы (title, text — объяснение, "
    "points — пункты списка);\n"
    "- definitions: определения терминов (term — термин, meaning — что он означает);\n"
    "- examples: примеры, которые приводил преподаватель;\n"
    "- facts: даты, числа, формулы, имена и конкретные факты;\n"
    "- key_points: 1–5 главных мыслей раздела, каждая — одно предложение.\n\n"
    + COMMON_RULES
)

SUMMARY_SYSTEM_PROMPT = (
    "Ты — опытный учитель. По плану конспекта лекции придумай точное название лекции "
    "(title, до 10 слов) и сформулируй 3–6 выводов (conclusions) — главное, что ученик должен "
    "понять после урока. Каждый вывод — одно-два предложения.\n\n" + COMMON_RULES
)

SHORT_SYSTEM_PROMPT = (
    "Ты — опытный учитель. Сделай КРАТКИЙ конспект лекции.\n"
    "- theses: не больше 12 ключевых тезисов, каждый — одно короткое предложение, "
    "без подробностей и длинных примеров;\n"
    "- remember: не больше 8 пунктов, которые нужно обязательно запомнить "
    "(ключевые определения в виде «термин — суть», важные даты, формулы).\n\n" + COMMON_RULES
)

TOPIC_SYSTEM_PROMPT = (
    "Ты — опытный учитель. Ученик хочет найти в конспекте лекции информацию на определённую тему. "
    "Выбери из конспекта ТОЛЬКО то, что связано с этой темой прямо или помогает её понять, "
    "и оформи это в виде разделов (та же структура: title, explanation, subsections, definitions, "
    "examples, facts, key_points). Всё, что не связано с темой, не включай. "
    "Если в тексте нет ничего по теме, верни пустой список sections.\n\n" + COMMON_RULES
)


def _text(value, limit: int = 3000) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        value = " ".join(str(item) for item in value)
    text = str(value).replace("**", "").strip()
    text = re.sub(r"[ \t]+", " ", text)
    return text[:limit]


def _text_list(value, limit_items: int = 40) -> list:
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        value = [value]
    result, seen = [], set()
    for item in value:
        if isinstance(item, dict):
            item = " — ".join(_text(v) for v in item.values() if _text(v))
        text = _text(item, 1000).lstrip("-•* ").strip()
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            result.append(text)
    return result[:limit_items]


def _first_sentences(text: str, count: int) -> str:
    sentences = re.split(r"(?<=[.!?…])\s+", text.strip())
    return " ".join(sentences[:count]).strip()


def _split_text(text: str, max_chars: int) -> list:
    pieces = []
    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= max_chars:
            pieces.append(paragraph)
        else:
            pieces.extend(s for s in re.split(r"(?<=[.!?…])\s+", paragraph) if s)

    parts, current = [], ""
    for piece in pieces:
        if current and len(current) + len(piece) + 2 > max_chars:
            parts.append(current)
            current = piece
        else:
            current = f"{current}\n\n{piece}" if current else piece
    if current:
        parts.append(current)
    return parts


def parse_section(data) -> Section:
    if not isinstance(data, dict):
        return Section(title="")

    subsections = []
    for item in data.get("subsections") or []:
        if not isinstance(item, dict):
            continue
        subsection = Subsection(
            title=_text(item.get("title"), 200),
            text=_text(item.get("text")),
            points=_text_list(item.get("points")),
        )
        if subsection.title or subsection.text or subsection.points:
            subsections.append(subsection)

    definitions = []
    for item in data.get("definitions") or []:
        if isinstance(item, dict):
            term, meaning = _text(item.get("term"), 200), _text(item.get("meaning"), 1500)
        else:
            term, _, meaning = _text(item).partition(" — ")
        if term and meaning:
            definitions.append(Definition(term=term, meaning=meaning))

    return Section(
        title=_text(data.get("title"), 200),
        explanation=_text(data.get("explanation")),
        subsections=subsections,
        definitions=definitions,
        examples=_text_list(data.get("examples")),
        facts=_text_list(data.get("facts")),
        key_points=_text_list(data.get("key_points")),
    )


def section_to_text(section: Section, index: int) -> str:
    lines = [f"{index}. {section.title}"]
    if section.explanation:
        lines.append(section.explanation)
    for subsection in section.subsections:
        lines.append(f"  Подраздел: {subsection.title}. {subsection.text}")
        lines.extend(f"  - {point}" for point in subsection.points)
    lines.extend(f"Определение: {d.term} — {d.meaning}" for d in section.definitions)
    lines.extend(f"Пример: {example}" for example in section.examples)
    lines.extend(f"Факт: {fact}" for fact in section.facts)
    lines.extend(f"Главное: {point}" for point in section.key_points)
    return "\n".join(lines)


class ConspectBuilder:
    def __init__(self, llm, part_chars: int = 6000):
        self.llm = llm
        self.part_chars = part_chars

    def build(self, cleaned, mode: str, focus_topic: str = "", on_progress=None) -> Conspect:
        if mode not in MODE_NAMES:
            raise ConspectError("Неизвестный режим конспекта.", f"mode={mode!r}")
        if mode == "topic" and not focus_topic.strip():
            raise ConspectError("Для режима «По теме» нужно указать тему.")
        if not cleaned.text.strip():
            raise ConspectError("После очистки не осталось текста лекции.")

        sections = self._extract_full_structure(cleaned, on_progress)
        title, conclusions = self._summarize(sections, cleaned.topic)

        conspect = Conspect(
            title=title,
            mode=mode,
            mode_name=MODE_NAMES[mode],
            mode_explanation=MODE_EXPLANATIONS[mode],
            subject=cleaned.subject,
            focus_topic=focus_topic.strip(),
        )

        if on_progress is not None and mode in ("short", "topic"):
            on_progress(f"готовлю вариант «{MODE_NAMES[mode]}»")

        if mode == "detailed":
            conspect.sections = sections
            conspect.conclusions = conclusions
        elif mode == "normal":
            conspect.sections = self._make_normal(sections)
            conspect.conclusions = conclusions
        elif mode == "short":
            conspect.sections = self._make_short(sections)
        else:
            conspect.sections = self._make_topic(sections, conspect.focus_topic, title)
            if not conspect.sections:
                conspect.notice = (
                    f"В лекции не найдено информации по теме «{conspect.focus_topic}». "
                    "Попробуйте сформулировать тему иначе или выберите другой режим."
                )

        conspect.stats = {
            "Предложений в расшифровке": str(cleaned.total_fragments),
            "Убрано шума распознавания и повторов": str(cleaned.removed_by_rules),
            "Убрано фраз не по теме лекции": str(cleaned.removed_as_offtopic),
        }
        if getattr(cleaned, "failed_blocks", 0):
            conspect.stats["Частей текста без смысловой фильтрации"] = str(cleaned.failed_blocks)
        return conspect


    def _extract_full_structure(self, cleaned, on_progress) -> list:
        parts = _split_text(cleaned.text, self.part_chars)
        subject_line = f"Предмет: {cleaned.subject}\n" if cleaned.subject else ""
        sections = []

        for index, part in enumerate(parts, start=1):
            if on_progress is not None and len(parts) > 1:
                on_progress(f"часть {index} из {len(parts)}")
            user_prompt = (
                f"{subject_line}Общая тема лекции: {cleaned.topic}\n"
                f"Это часть {index} из {len(parts)}.\n\n"
                f"Текст лекции:\n{part}"
            )
            try:
                answer = self.llm.chat_json(EXTRACT_SYSTEM_PROMPT, user_prompt, SECTIONS_SCHEMA, max_tokens=4096)
                new_sections = [parse_section(item) for item in answer.get("sections") or []]
                new_sections = [s for s in new_sections if not s.is_empty()]
            except Exception as error:
                if getattr(error, "fatal", False):
                    raise
                logger.warning("Не удалось структурировать часть %d: %s", index, error)
                new_sections = []

            if not new_sections:
                logger.warning("Часть %d добавлена в конспект без структурирования", index)
                new_sections = [Section(title=f"Материал лекции (часть {index})", explanation=part)]

            for section in new_sections:
                if not section.title:
                    section.title = f"Раздел {len(sections) + 1}"
                self._add_or_merge(sections, section)

        return sections

    @staticmethod
    def _add_or_merge(sections: list, section: Section) -> None:
        if sections and sections[-1].title.lower() == section.title.lower():
            previous = sections[-1]
            previous.explanation = f"{previous.explanation}\n\n{section.explanation}".strip()
            previous.subsections += section.subsections
            previous.definitions += section.definitions
            previous.examples = _text_list(previous.examples + section.examples, 100)
            previous.facts = _text_list(previous.facts + section.facts, 100)
            previous.key_points = _text_list(previous.key_points + section.key_points, 100)
        else:
            sections.append(section)

    def _summarize(self, sections: list, topic: str):
        plan = self._compact_plan(sections, max_chars=self.part_chars)
        try:
            answer = self.llm.chat_json(
                SUMMARY_SYSTEM_PROMPT,
                f"Предполагаемая тема: {topic}\n\nПлан конспекта:\n{plan}",
                SUMMARY_SCHEMA,
                max_tokens=800,
            )
            title = _text(answer.get("title"), 150).strip("«»\"")
            conclusions = _text_list(answer.get("conclusions"), 8)
        except Exception as error:
            if getattr(error, "fatal", False):
                raise
            logger.warning("Не удалось сформулировать выводы: %s", error)
            title, conclusions = "", []
        return title or topic or "Конспект лекции", conclusions

    @staticmethod
    def _compact_plan(sections: list, max_chars: int) -> str:
        lines = []
        for index, section in enumerate(sections, start=1):
            lines.append(f"{index}. {section.title}")
            lines.extend(f"   - {point}" for point in section.key_points)
            lines.extend(f"   - {d.term} — {d.meaning}" for d in section.definitions)
            lines.extend(f"   - {fact}" for fact in section.facts)
            if not section.key_points and section.explanation:
                lines.append(f"   - {_first_sentences(section.explanation, 2)}")
        return "\n".join(lines)[:max_chars]


    @staticmethod
    def _make_normal(sections: list) -> list:
        result = []
        for section in sections:
            key_points = list(section.key_points)
            if not key_points:
                for subsection in section.subsections:
                    if subsection.points:
                        key_points.append(subsection.points[0])
                    elif subsection.text:
                        key_points.append(_first_sentences(subsection.text, 1))
            result.append(
                Section(
                    title=section.title,
                    explanation=_first_sentences(section.explanation, 2),
                    definitions=list(section.definitions),
                    examples=section.examples[:2],
                    facts=list(section.facts),
                    key_points=key_points,
                )
            )
        return result

    def _make_short(self, sections: list) -> list:
        plan = self._compact_plan(sections, max_chars=self.part_chars * 2)
        theses, remember = [], []
        for part in _split_text(plan, self.part_chars):
            try:
                answer = self.llm.chat_json(
                    SHORT_SYSTEM_PROMPT, f"План конспекта лекции:\n{part}", SHORT_SCHEMA, max_tokens=1200
                )
                theses += _text_list(answer.get("theses"), 12)
                remember += _text_list(answer.get("remember"), 8)
            except Exception as error:
                if getattr(error, "fatal", False):
                    raise
                logger.warning("Краткий конспект: нейросеть не справилась, использую запасной вариант: %s", error)

        if not theses:
            for section in sections:
                if section.key_points:
                    theses.append(section.key_points[0])
                elif section.explanation:
                    theses.append(_first_sentences(section.explanation, 1))
            remember = [f"{d.term} — {d.meaning}" for s in sections for d in s.definitions]

        result = [Section(title="Ключевые тезисы", key_points=_text_list(theses, 12))]
        if remember:
            result.append(Section(title="Нужно запомнить", key_points=_text_list(remember, 8)))
        return result

    def _make_topic(self, sections: list, focus_topic: str, title: str) -> list:
        full_text = "\n\n".join(section_to_text(section, i) for i, section in enumerate(sections, start=1))
        result = []
        parts = _split_text(full_text, self.part_chars)
        failed_parts = 0
        for part in parts:
            user_prompt = (
                f"Название лекции: {title}\n"
                f"ТЕМА, КОТОРАЯ ИНТЕРЕСУЕТ УЧЕНИКА: {focus_topic}\n\n"
                f"Конспект лекции:\n{part}"
            )
            try:
                answer = self.llm.chat_json(TOPIC_SYSTEM_PROMPT, user_prompt, SECTIONS_SCHEMA, max_tokens=3000)
            except Exception as error:
                if getattr(error, "fatal", False):
                    raise
                logger.warning("Режим «По теме»: не удалось обработать часть: %s", error)
                failed_parts += 1
                continue
            for item in answer.get("sections") or []:
                section = parse_section(item)
                if not section.is_empty():
                    section.title = section.title or focus_topic
                    self._add_or_merge(result, section)
        if parts and failed_parts == len(parts):
            raise ConspectError("Локальная нейросеть не смогла выбрать информацию по теме. Попробуйте ещё раз.")
        return result
