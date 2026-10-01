
import json
import logging
import re
from dataclasses import dataclass, field

import requests

logger = logging.getLogger(__name__)


class LLMError(Exception):
    def __init__(self, user_message: str, technical_details: str = "", fatal: bool = True):
        super().__init__(technical_details or user_message)
        self.user_message = user_message
        self.fatal = fatal


class OllamaClient:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen2.5:7b",
        timeout_seconds: int = 600,
        num_ctx: int = 8192,
        temperature: float = 0.2,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.num_ctx = num_ctx
        self.temperature = temperature

    def check_available(self) -> None:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            models = response.json().get("models", [])
        except requests.RequestException as error:
            raise LLMError(
                "Локальная нейросеть недоступна: программа Ollama не запущена. "
                "Запустите Ollama и попробуйте снова.",
                f"{self.base_url}: {error!r}",
            ) from error

        installed = set()
        for item in models:
            for key in ("name", "model"):
                if item.get(key):
                    installed.add(item[key])

        wanted = {self.model}
        if ":" not in self.model:
            wanted.add(f"{self.model}:latest")
        if not wanted & installed:
            raise LLMError(
                f"Локальная модель «{self.model}» не скачана. "
                f"На компьютере с ботом выполните: ollama pull {self.model}",
                f"Установленные модели: {sorted(installed)}",
            )

    def chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: dict,
        max_tokens: int = 1024,
        retries: int = 1,
    ) -> dict:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "format": schema,
            "keep_alive": "15m",
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
                "num_predict": max_tokens,
            },
        }

        last_error = ""
        for attempt in range(1, retries + 2):
            try:
                response = requests.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    timeout=self.timeout_seconds,
                )
            except requests.ConnectionError as error:
                raise LLMError(
                    "Локальная нейросеть недоступна: программа Ollama не запущена.",
                    repr(error),
                ) from error
            except requests.Timeout:
                last_error = f"таймаут {self.timeout_seconds} с"
                logger.warning("Ollama не ответил вовремя (попытка %d)", attempt)
                continue

            if response.status_code != 200:
                error_text = self._error_text(response)
                lowered = error_text.lower()
                if response.status_code == 404 or "not found" in lowered:
                    raise LLMError(
                        f"Локальная модель «{self.model}» не найдена. Выполните: ollama pull {self.model}",
                        error_text,
                    )
                if "memory" in lowered:
                    raise LLMError(
                        "Компьютеру не хватает памяти для этой нейросети. "
                        "Выберите модель поменьше в файле .env (например, qwen2.5:3b).",
                        error_text,
                    )
                last_error = f"HTTP {response.status_code}: {error_text}"
                logger.warning("Ollama вернул ошибку (попытка %d): %s", attempt, last_error)
                continue

            try:
                content = response.json().get("message", {}).get("content", "")
            except ValueError:
                content = ""
            data = parse_json_answer(content)
            if isinstance(data, dict):
                return data
            last_error = f"не удалось разобрать JSON: {content[:300]!r}"
            logger.warning("Нейросеть вернула некорректный JSON (попытка %d)", attempt)

        raise LLMError(
            "Локальная нейросеть не смогла обработать текст.",
            last_error,
            fatal=False,
        )

    @staticmethod
    def _error_text(response) -> str:
        try:
            return str(response.json().get("error", response.text))
        except ValueError:
            return response.text[:500]


def parse_json_answer(content: str):
    if not content:
        return None
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", content, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


@dataclass
class Fragment:
    id: int
    start: float
    text: str


@dataclass
class CleanedLecture:
    topic: str
    subject: str
    text: str
    kept_fragments: list = field(default_factory=list)
    total_fragments: int = 0
    removed_by_rules: int = 0
    removed_as_offtopic: int = 0
    failed_blocks: int = 0


class CleaningError(Exception):
    def __init__(self, user_message: str, technical_details: str = ""):
        super().__init__(technical_details or user_message)
        self.user_message = user_message


HALLUCINATION_PATTERNS = [
    r"субтитр\w*\s+(сделал|создавал|подготовил|делал|предоставил)",
    r"редактор\s+субтитров",
    r"корректор\s+\w\.",
    r"продолжение\s+следует",
    r"спасибо\s+за\s+просмотр",
    r"подписывайтесь\s+на\s+(канал|наш)",
    r"ставьте\s+лайк",
    r"dimatorzok",
    r"amara\.org",
]
HALLUCINATION_RE = re.compile("|".join(HALLUCINATION_PATTERNS), re.IGNORECASE)

FILLER_RE = re.compile(r"(?<!\w)(?:э+(?:-э+)*|эм+|м{2,}|хм+|угу)(?!\w)[,.]?\s*", re.IGNORECASE)
LEADING_FILLER_RE = re.compile(r"^(?:ну|короче|так вот|вот)\s*,\s*", re.IGNORECASE)
REPEATED_WORD_RE = re.compile(r"\b([^\W\d_]+)(?:[\s,]+\1\b)+", re.IGNORECASE)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+")


def capitalize_first_letter(text: str) -> str:
    return text[0].upper() + text[1:] if text else text


def clean_sentence(text: str) -> str:
    if HALLUCINATION_RE.search(text):
        return ""
    text = FILLER_RE.sub("", text)
    text = LEADING_FILLER_RE.sub("", text.strip())
    text = REPEATED_WORD_RE.sub(r"\1", text)
    text = re.sub(r"\s+([,.!?:;])", r"\1", text)
    text = re.sub(r"([,.!?])\1+", r"\1", text)
    text = re.sub(r"\s{2,}", " ", text).strip(" -").lstrip(", ")
    if len(re.findall(r"[^\W\d_]", text)) < 2:
        return ""
    return text


def split_long_text(text: str, max_length: int = 350) -> list:
    if len(text) <= max_length:
        return [text]
    parts, current = [], ""
    for word in text.split():
        if current and len(current) + len(word) + 1 > max_length:
            parts.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        parts.append(current)
    return parts


class LectureCleaner:
    TOPIC_SCHEMA = {
        "type": "object",
        "properties": {
            "subject": {"type": "string"},
            "topic": {"type": "string"},
        },
        "required": ["subject", "topic"],
    }

    REMOVE_SCHEMA = {
        "type": "object",
        "properties": {
            "remove": {"type": "array", "items": {"type": "integer"}},
        },
        "required": ["remove"],
    }

    KEEP_SCHEMA = {
        "type": "object",
        "properties": {
            "keep": {"type": "array", "items": {"type": "integer"}},
        },
        "required": ["keep"],
    }

    TOPIC_SYSTEM_PROMPT = (
        "Ты помогаешь ученику разобрать запись урока. Тебе дают отрывки автоматической "
        "расшифровки (в ней бывают ошибки распознавания и посторонние разговоры). "
        "Определи школьный предмет и главную тему урока. Тему напиши коротко, "
        "как заголовок, на русском языке (например: «Фотосинтез» или «Причины Первой мировой войны»). "
        "Если тему понять невозможно, напиши «Лекция». Ответь только JSON."
    )

    REMOVE_SYSTEM_PROMPT = (
        "Ты помогаешь ученику сделать конспект урока. Тебе дают автоматическую расшифровку "
        "аудиозаписи, разбитую на пронумерованные фрагменты. В расшифровке бывают ошибки распознавания.\n\n"
        "Задача: найти фрагменты, которые НЕ относятся к учебному материалу, чтобы их удалить.\n\n"
        "УДАЛЯТЬ:\n"
        "- разговоры учеников между собой;\n"
        "- бытовые разговоры (еда, перемена, телефоны, игры, планы на вечер);\n"
        "- случайные реплики, шутки и комментарии не по теме урока;\n"
        "- организационные мелочи, не связанные с материалом («закройте окно», «кто дежурный», «тише»);\n"
        "- бессмысленные обрывки фраз;\n"
        "- дословные повторы того, что уже было сказано.\n\n"
        "НЕ УДАЛЯТЬ:\n"
        "- объяснения преподавателя;\n"
        "- определения, правила, формулы;\n"
        "- примеры;\n"
        "- даты, числа, имена, факты;\n"
        "- вопросы преподавателя по теме и ответы учеников по теме;\n"
        "- уточнения, выводы, домашнее задание и важные объявления по предмету.\n\n"
        "Если сомневаешься — НЕ удаляй фрагмент.\n"
        "Ответь только JSON: {\"remove\": [номера фрагментов для удаления]}. "
        "Если удалять нечего, верни {\"remove\": []}."
    )

    def __init__(self, llm: OllamaClient, block_chars: int = 3000):
        self.llm = llm
        self.block_chars = block_chars


    def clean(self, transcript, on_progress=None) -> CleanedLecture:
        fragments, total_sentences = self._make_fragments(transcript.segments)
        removed_by_rules = total_sentences - len(fragments)
        if not fragments:
            raise CleaningError(
                "В записи не найдено осмысленной речи. Проверьте, что на записи слышна лекция.",
                "После очистки правилами не осталось фрагментов",
            )

        subject, topic = self._detect_topic(fragments)
        logger.info("Предмет: %s | тема: %s", subject, topic)

        blocks = self._make_blocks(fragments)
        removed_ids = set()
        failed_blocks = 0

        for index, block in enumerate(blocks):
            try:
                removed_ids |= self._find_offtopic(block, subject, topic)
            except LLMError as error:
                if error.fatal:
                    raise
                failed_blocks += 1
                logger.warning("Блок %d оставлен без фильтрации: %s", index + 1, error)
            if on_progress is not None:
                on_progress(index + 1, len(blocks))

        kept = [fragment for fragment in fragments if fragment.id not in removed_ids]
        logger.info(
            "Фрагментов: %d, убрано правилами: %d, убрано как «не по теме»: %d",
            total_sentences, removed_by_rules, len(removed_ids),
        )
        for fragment in fragments:
            if fragment.id in removed_ids:
                logger.debug("Удалено как «не по теме»: %s", fragment.text)

        if not kept:
            raise CleaningError(
                "В записи не найдено учебного материала: похоже, там только посторонние разговоры."
            )

        return CleanedLecture(
            topic=topic,
            subject=subject,
            text=self._join_paragraphs(fragments, removed_ids),
            kept_fragments=kept,
            total_fragments=total_sentences,
            removed_by_rules=removed_by_rules,
            removed_as_offtopic=len(removed_ids),
            failed_blocks=failed_blocks,
        )


    @staticmethod
    def _make_fragments(segments):
        fragments = []
        total = 0
        previous_text = ""
        for segment in segments:
            for sentence in SENTENCE_SPLIT_RE.split(segment.text.strip()):
                for piece in split_long_text(sentence.strip()):
                    if not piece:
                        continue
                    total += 1
                    cleaned = clean_sentence(piece)
                    if not cleaned or cleaned.lower() == previous_text.lower():
                        continue
                    if not previous_text or previous_text[-1] in ".!?…":
                        cleaned = capitalize_first_letter(cleaned)
                    previous_text = cleaned
                    fragments.append(Fragment(id=len(fragments) + 1, start=segment.start, text=cleaned))
        return fragments, total


    def _detect_topic(self, fragments):
        full_text = " ".join(fragment.text for fragment in fragments)
        if len(full_text) > 4000:
            middle = len(full_text) // 2
            sample = f"{full_text[:2500]}\n...\n{full_text[middle:middle + 1500]}"
        else:
            sample = full_text

        try:
            answer = self.llm.chat_json(
                self.TOPIC_SYSTEM_PROMPT,
                f"Отрывки расшифровки урока:\n\n{sample}",
                self.TOPIC_SCHEMA,
                max_tokens=150,
            )
        except LLMError as error:
            if error.fatal:
                raise
            return "", "Лекция"

        subject = str(answer.get("subject", "")).strip()[:100]
        topic = str(answer.get("topic", "")).strip().strip("«»\"")[:150] or "Лекция"
        return subject, topic


    def _make_blocks(self, fragments):
        blocks, current, length = [], [], 0
        for fragment in fragments:
            if current and length + len(fragment.text) > self.block_chars:
                blocks.append(current)
                current, length = [], 0
            current.append(fragment)
            length += len(fragment.text) + 8
        if current:
            blocks.append(current)
        return blocks

    def _find_offtopic(self, block, subject: str, topic: str) -> set:
        numbered = "\n".join(f"[{fragment.id}] {fragment.text}" for fragment in block)
        subject_line = f"Предмет: {subject}\n" if subject else ""
        user_prompt = (
            f"{subject_line}Тема урока: {topic}\n\n"
            f"Фрагменты расшифровки:\n{numbered}\n\n"
            "Верни номера фрагментов, которые не относятся к учебному материалу."
        )
        answer = self.llm.chat_json(self.REMOVE_SYSTEM_PROMPT, user_prompt, self.REMOVE_SCHEMA, max_tokens=400)

        valid_ids = {fragment.id for fragment in block}
        result = set()
        for value in answer.get("remove", []):
            try:
                number = int(value)
            except (TypeError, ValueError):
                continue
            if number in valid_ids:
                result.add(number)

        if len(result) >= 2 and len(result) > len(valid_ids) / 2:
            removed_fragments = [fragment for fragment in block if fragment.id in result]
            restored = self._restore_lesson_fragments(removed_fragments, subject_line, topic)
            if restored:
                logger.info("Повторная проверка вернула в текст фрагментов: %d", len(restored))
            result -= restored
        return result

    def _restore_lesson_fragments(self, fragments, subject_line: str, topic: str) -> set:
        numbered = "\n".join(f"[{fragment.id}] {fragment.text}" for fragment in fragments)
        user_prompt = (
            f"{subject_line}Тема урока: {topic}\n\n"
            f"Фрагменты расшифровки:\n{numbered}\n\n"
            "Верни номера фрагментов, в которых есть объяснение, определение, пример, факт, дата, "
            "формула, вопрос или вывод, относящиеся к теме урока или к предмету. "
            "Разговоры не по теме не включай."
        )
        valid_ids = {fragment.id for fragment in fragments}
        try:
            answer = self.llm.chat_json(
                "Ты внимательно проверяешь расшифровку урока. Ответь только JSON: {\"keep\": [номера]}.",
                user_prompt,
                self.KEEP_SCHEMA,
                max_tokens=400,
            )
        except LLMError as error:
            if error.fatal:
                raise
            return valid_ids

        restored = set()
        for value in answer.get("keep", []):
            try:
                number = int(value)
            except (TypeError, ValueError):
                continue
            if number in valid_ids:
                restored.add(number)
        return restored


    @staticmethod
    def _join_paragraphs(fragments, removed_ids, paragraph_chars: int = 900) -> str:
        paragraphs, current = [], []
        for fragment in fragments:
            if fragment.id in removed_ids:
                if current:
                    paragraphs.append(" ".join(current))
                    current = []
                continue
            current.append(fragment.text)
            if sum(len(text) for text in current) > paragraph_chars:
                paragraphs.append(" ".join(current))
                current = []
        if current:
            paragraphs.append(" ".join(current))

        cleaned_paragraphs = []
        for paragraph in paragraphs:
            paragraph = REPEATED_WORD_RE.sub(r"\1", paragraph)
            cleaned_paragraphs.append(capitalize_first_letter(paragraph))
        return "\n\n".join(cleaned_paragraphs)
