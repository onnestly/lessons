import asyncio
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from telegram import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
    ReplyKeyboardMarkup,
    Update,
)
from telegram.constants import ChatAction
from telegram.error import BadRequest, TelegramError
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

logger = logging.getLogger(__name__)

BUTTON_HELP = "Порядок работы"
BUTTON_MODES = "Режимы конспекта"

MAX_TOPIC_LENGTH = 200

MIME_TO_EXTENSION = {
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/ogg": "ogg",
    "audio/opus": "opus",
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/wave": "wav",
    "audio/mp4": "m4a",
    "audio/x-m4a": "m4a",
    "audio/m4a": "m4a",
    "audio/aac": "aac",
    "audio/flac": "flac",
    "audio/x-flac": "flac",
    "audio/webm": "webm",
    "audio/amr": "amr",
    "audio/x-ms-wma": "wma",
    "video/mp4": "mp4",
    "video/webm": "webm",
}


class BotUserError(Exception):
    def __init__(self, user_message: str, technical_details: str = ""):
        super().__init__(technical_details or user_message)
        self.user_message = user_message


@dataclass
class IncomingAudio:
    file_id: str
    extension: str
    size_bytes: int
    description: str


class ConspectatorBot:
    def __init__(
        self,
        token: str,
        pipeline,
        modes: dict,
        mode_descriptions: dict,
        supported_extensions: tuple,
        max_file_size_mb: int,
        proxy: str = "",
    ):
        self._token = token
        self._proxy = proxy
        self._pipeline = pipeline
        self._modes = modes
        self._mode_descriptions = mode_descriptions
        self._supported_extensions = supported_extensions
        self._max_file_size_mb = max_file_size_mb
        self._processing_lock = asyncio.Lock()

    def build_application(self) -> Application:
        builder = (
            ApplicationBuilder()
            .token(self._token)
            .concurrent_updates(True)
            .connect_timeout(30)
            .read_timeout(60)
            .write_timeout(120)
        )
        if self._proxy:
            logger.info("Подключение к Telegram через прокси")
            builder = builder.proxy(self._proxy).get_updates_proxy(self._proxy)
        application = builder.build()

        application.add_handler(CommandHandler("start", self.start_command))
        application.add_handler(CommandHandler("help", self.help_command))
        application.add_handler(CommandHandler("modes", self.modes_command))
        application.add_handler(CommandHandler("cancel", self.cancel_command))

        audio_filter = filters.VOICE | filters.AUDIO | filters.Document.ALL
        application.add_handler(MessageHandler(audio_filter, self.handle_audio))
        application.add_handler(CallbackQueryHandler(self.handle_mode_choice, pattern=r"^mode:"))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_text))

        application.add_error_handler(self.handle_telegram_error)
        return application

    def run(self) -> None:
        application = self.build_application()
        logger.info("Бот запущен. Для остановки нажмите Ctrl+C.")
        application.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)

    def _main_menu(self) -> ReplyKeyboardMarkup:
        return ReplyKeyboardMarkup(
            [[BUTTON_HELP, BUTTON_MODES]],
            resize_keyboard=True,
        )

    def _mode_keyboard(self) -> InlineKeyboardMarkup:
        keys = list(self._modes.keys())
        rows = []
        for i in range(0, len(keys), 2):
            rows.append(
                [InlineKeyboardButton(self._modes[key], callback_data=f"mode:{key}") for key in keys[i:i + 2]]
            )
        rows.append([InlineKeyboardButton("Отмена", callback_data="mode:cancel")])
        return InlineKeyboardMarkup(rows)

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        context.user_data.clear()
        text = (
            "Система формирования конспектов по аудиозаписям лекций.\n\n"
            "Отправьте голосовое сообщение или аудиофайл, выберите формат конспекта "
            "и получите готовый документ в формате PDF."
        )
        await update.message.reply_text(text, reply_markup=self._main_menu())

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        extensions = ", ".join(ext.upper() for ext in self._supported_extensions)
        text = (
            "Порядок работы\n\n"
            "1. Отправьте голосовое сообщение или аудиофайл с лекцией.\n"
            "2. Выберите формат конспекта.\n"
            "3. Дождитесь завершения обработки и получите документ PDF.\n\n"
            f"Допустимые форматы: {extensions}.\n"
            f"Максимальный размер файла: {self._max_file_size_mb} МБ.\n\n"
            "Команды:\n"
            "/start — начать заново\n"
            "/modes — описание режимов\n"
            "/cancel — отменить выбранную запись"
        )
        await update.message.reply_text(text, reply_markup=self._main_menu())

    async def modes_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        lines = ["Режимы конспекта\n"]
        for key, title in self._modes.items():
            lines.append(f"{title} — {self._mode_descriptions.get(key, '')}")
        await update.message.reply_text("\n".join(lines), reply_markup=self._main_menu())

    async def cancel_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        context.user_data.pop("pending_audio", None)
        context.user_data["awaiting_topic"] = False
        await update.message.reply_text(
            "Операция отменена. Вы можете отправить новую запись.",
            reply_markup=self._main_menu(),
        )

    def _describe_incoming_audio(self, message: Message):
        if message.voice:
            return IncomingAudio(
                file_id=message.voice.file_id,
                extension="ogg",
                size_bytes=message.voice.file_size or 0,
                description="голосовое сообщение",
            )

        if message.audio:
            extension = self._guess_extension(message.audio.file_name, message.audio.mime_type)
            return IncomingAudio(
                file_id=message.audio.file_id,
                extension=extension or "mp3",
                size_bytes=message.audio.file_size or 0,
                description="аудиофайл",
            )

        if message.document:
            document = message.document
            extension = self._guess_extension(document.file_name, document.mime_type)
            mime = (document.mime_type or "").lower()
            is_audio_mime = mime.startswith("audio/")
            if extension in self._supported_extensions or is_audio_mime:
                return IncomingAudio(
                    file_id=document.file_id,
                    extension=extension or "bin",
                    size_bytes=document.file_size or 0,
                    description="аудиофайл",
                )
        return None

    @staticmethod
    def _guess_extension(file_name, mime_type) -> str:
        if file_name:
            suffix = Path(file_name).suffix.lower().lstrip(".")
            if re.fullmatch(r"[a-z0-9]{1,5}", suffix):
                return suffix
        if mime_type:
            return MIME_TO_EXTENSION.get(mime_type.lower(), "")
        return ""

    async def handle_audio(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = update.message
        if message is None:
            return

        audio = self._describe_incoming_audio(message)
        if audio is None:
            extensions = ", ".join(ext.upper() for ext in self._supported_extensions)
            await message.reply_text(
                "Формат файла не поддерживается.\n"
                f"Допустимые форматы: {extensions}."
            )
            return

        max_bytes = self._max_file_size_mb * 1024 * 1024
        if audio.size_bytes > max_bytes:
            size_mb = audio.size_bytes / (1024 * 1024)
            await message.reply_text(
                f"Размер файла превышает допустимый ({size_mb:.1f} МБ). "
                f"Максимальный размер: {self._max_file_size_mb} МБ.\n\n"
                "Рекомендация: сожмите запись в формат OGG/Opus или разделите её на части "
                "(описание приведено в README)."
            )
            return

        context.user_data["pending_audio"] = audio
        context.user_data["awaiting_topic"] = False
        await message.reply_text(
            f"Получено: {audio.description}. Выберите формат конспекта.",
            reply_markup=self._mode_keyboard(),
        )

    async def handle_mode_choice(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        await query.answer()
        choice = query.data.split(":", 1)[1]

        audio = context.user_data.get("pending_audio")
        if audio is None:
            await query.edit_message_text("Сначала отправьте голосовое сообщение или аудиофайл.")
            return

        if choice == "cancel":
            context.user_data.pop("pending_audio", None)
            context.user_data["awaiting_topic"] = False
            await query.edit_message_text("Операция отменена.")
            return

        if choice not in self._modes:
            return

        if choice == "topic":
            context.user_data["awaiting_topic"] = True
            await query.edit_message_text(
                "Режим «По теме».\n\n"
                "Укажите одним сообщением тему, по которой требуется составить конспект.\n"
                "Например: «причины Первой мировой войны»."
            )
            return

        await query.edit_message_text(f"Выбран режим: {self._modes[choice]}")
        await self._process_lecture(query.message, context, audio, choice, "")

    async def handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = update.message
        if message is None or message.text is None:
            return
        text = message.text.strip()

        if text == BUTTON_HELP:
            await self.help_command(update, context)
            return
        if text == BUTTON_MODES:
            await self.modes_command(update, context)
            return

        if context.user_data.get("awaiting_topic"):
            audio = context.user_data.get("pending_audio")
            if audio is None:
                context.user_data["awaiting_topic"] = False
                await message.reply_text("Сначала отправьте голосовое сообщение или аудиофайл.")
                return

            topic = re.sub(r"\s+", " ", text)[:MAX_TOPIC_LENGTH]
            if len(topic) < 2:
                await message.reply_text("Тема указана слишком кратко. Уточните формулировку.")
                return

            context.user_data["awaiting_topic"] = False
            await message.reply_text(f"Тема принята: «{topic}».")
            await self._process_lecture(message, context, audio, "topic", topic)
            return

        await message.reply_text(
            "Отправьте голосовое сообщение или аудиофайл с лекцией.",
            reply_markup=self._main_menu(),
        )

    async def _process_lecture(
        self,
        message: Message,
        context: ContextTypes.DEFAULT_TYPE,
        audio: IncomingAudio,
        mode: str,
        focus_topic: str,
    ) -> None:
        context.user_data.pop("pending_audio", None)
        context.user_data["awaiting_topic"] = False

        if self._processing_lock.locked():
            await message.reply_text(
                "В данный момент обрабатывается другая запись. "
                "Ваша запись поставлена в очередь."
            )

        async with self._processing_lock:
            status_message = await message.reply_text("Получение аудио...")
            loop = asyncio.get_running_loop()

            def on_status(text: str) -> None:
                future = asyncio.run_coroutine_threadsafe(self._set_status(status_message, text), loop)
                try:
                    future.result(timeout=30)
                except Exception:
                    logger.debug("Не удалось обновить статус", exc_info=True)

            try:
                with self._pipeline.create_workspace() as workspace:
                    input_path = workspace.file(f"input.{audio.extension}")
                    await self._download_audio(context, audio, input_path)

                    pdf_path = await asyncio.to_thread(
                        self._pipeline.process,
                        input_path,
                        mode,
                        focus_topic,
                        workspace,
                        on_status,
                    )

                    await context.bot.send_chat_action(message.chat_id, ChatAction.UPLOAD_DOCUMENT)
                    mode_name = self._modes.get(mode, mode)
                    file_name = f"Конспект_{datetime.now():%Y-%m-%d_%H-%M}.pdf"
                    caption = f"Конспект готов.\nРежим: {mode_name}"
                    if focus_topic:
                        caption += f"\nТема: {focus_topic}"
                    with open(pdf_path, "rb") as pdf_file:
                        await message.reply_document(
                            document=pdf_file,
                            filename=file_name,
                            caption=caption,
                            reply_markup=self._main_menu(),
                        )
                await self._set_status(status_message, "Обработка завершена.")

            except Exception as error:
                logger.exception("Ошибка при обработке лекции (режим=%s)", mode)
                user_text = getattr(error, "user_message", None) or (
                    "Произошла ошибка при обработке записи. Повторите попытку позже."
                )
                await self._set_status(status_message, user_text)

    async def _download_audio(
        self,
        context: ContextTypes.DEFAULT_TYPE,
        audio: IncomingAudio,
        destination: Path,
    ) -> None:
        try:
            telegram_file = await context.bot.get_file(audio.file_id)
            await telegram_file.download_to_drive(custom_path=destination)
        except BadRequest as error:
            if "too big" in str(error).lower():
                raise BotUserError(
                    "Размер файла превышает ограничение Telegram (20 МБ). "
                    "Сожмите запись или разделите её на части.",
                    str(error),
                ) from error
            raise BotUserError("Не удалось загрузить файл из Telegram. Повторите отправку.", str(error)) from error
        except TelegramError as error:
            raise BotUserError(
                "Не удалось загрузить файл из Telegram из-за проблемы с сетью. Повторите попытку.",
                str(error),
            ) from error

        if not destination.exists() or destination.stat().st_size == 0:
            raise BotUserError("Файл пуст. Повторите отправку.")

    @staticmethod
    async def _set_status(status_message: Message, text: str) -> None:
        try:
            await status_message.edit_text(text)
        except BadRequest as error:
            if "not modified" not in str(error).lower():
                logger.warning("Не удалось обновить статус: %s", error)
        except TelegramError as error:
            logger.warning("Не удалось обновить статус: %s", error)

    async def handle_telegram_error(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        logger.error("Необработанная ошибка Telegram-бота", exc_info=context.error)
