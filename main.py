
import argparse
import importlib
import logging
import sys
from pathlib import Path

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

try:
    import config

    bot_module = importlib.import_module("1_bot")
    audio_module = importlib.import_module("2_audio")
    transcription_module = importlib.import_module("3_transcription")
    cleaning_module = importlib.import_module("4_cleaning")
    conspect_module = importlib.import_module("5_conspect")
    pdf_module = importlib.import_module("6_pdf")
except ImportError as import_error:
    print(
        "\n[ОШИБКА] Не установлены библиотеки Python: "
        f"{import_error}\n"
        "Активируйте виртуальное окружение и выполните:\n"
        "    pip install -r requirements.txt\n"
    )
    sys.exit(1)

logger = logging.getLogger("konspektator")


class SecretMaskingFormatter(logging.Formatter):
    def __init__(self, secret: str):
        super().__init__("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", "%H:%M:%S")
        self.secret = secret

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        if self.secret:
            text = text.replace(self.secret, "***TOKEN***")
        return text


def setup_logging(level: str = "INFO", secret: str = "") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(SecretMaskingFormatter(secret))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    for noisy in ("httpx", "httpcore", "telegram.ext.Updater", "faster_whisper", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def format_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours} ч {minutes} мин {secs} с"
    if minutes:
        return f"{minutes} мин {secs} с"
    return f"{secs} с"


class LecturePipeline:
    def __init__(self, settings, audio_processor, transcriber, cleaner, conspect_builder, pdf_renderer):
        self.settings = settings
        self.audio_processor = audio_processor
        self.transcriber = transcriber
        self.cleaner = cleaner
        self.conspect_builder = conspect_builder
        self.pdf_renderer = pdf_renderer

    def create_workspace(self):
        return audio_module.TempWorkspace(self.settings.temp_dir)

    def process(self, input_path: Path, mode: str, focus_topic: str, workspace, on_status) -> Path:
        on_status("Очищаю запись от шума...")
        prepared = self.audio_processor.prepare(input_path, workspace.file("prepared.wav"))
        Path(input_path).unlink(missing_ok=True)

        on_status("Распознаю речь...")

        def transcription_progress(done: int, total: int) -> None:
            if total > 1 and done < total:
                on_status(f"Распознаю речь... (готово частей: {done} из {total})")

        transcript = self.transcriber.transcribe(prepared.path, on_progress=transcription_progress)

        on_status("Убираю разговоры и информацию не по теме...")

        def cleaning_progress(done: int, total: int) -> None:
            if total > 1 and done < total:
                on_status(f"Убираю разговоры и информацию не по теме... (блок {done + 1} из {total})")

        cleaned = self.cleaner.clean(transcript, on_progress=cleaning_progress)

        on_status("Формирую конспект...")
        conspect = self.conspect_builder.build(
            cleaned,
            mode,
            focus_topic,
            on_progress=lambda text: on_status(f"Формирую конспект... ({text})"),
        )
        conspect.stats = {
            "Длительность записи": format_duration(prepared.original_duration_seconds),
            **conspect.stats,
        }

        on_status("Создаю PDF...")
        return self.pdf_renderer.render(conspect, workspace.file("conspect.pdf"))


def check_components(settings, audio_processor, pdf_renderer, llm, transcriber) -> bool:
    ok = True
    print("\n=== Проверка компонентов ===")

    try:
        version = audio_processor.check_ffmpeg()
        print(f"[OK]     FFmpeg: {version}")
    except audio_module.AudioError as error:
        print(f"[ОШИБКА] {error.user_message}")
        logger.debug("Подробности: %s", error)
        ok = False

    try:
        font = pdf_renderer.check_fonts()
        print(f"[OK]     Шрифт для PDF: {font}")
    except pdf_module.PdfError as error:
        print(f"[ОШИБКА] {error.user_message}")
        ok = False

    try:
        llm.check_available()
        print(f"[OK]     Ollama запущен, модель «{settings.ollama_model}» найдена")
    except cleaning_module.LLMError as error:
        print(f"[ВНИМАНИЕ] {error.user_message}")
        logger.debug("Подробности: %s", error)

    settings.temp_dir.mkdir(parents=True, exist_ok=True)
    removed = audio_module.cleanup_old_workspaces(settings.temp_dir)
    print(f"[OK]     Папка временных файлов: {settings.temp_dir}")
    if removed:
        print(f"         (удалено старых временных папок: {removed})")

    if ok:
        try:
            print(f"...      Загружаю Whisper «{settings.whisper_model}» (первый раз модель скачивается)")
            transcriber.load_model()
            print(f"[OK]     Whisper «{settings.whisper_model}» загружен ({settings.whisper_device})")
        except transcription_module.TranscriptionError as error:
            print(f"[ОШИБКА] {error.user_message}")
            logger.error("Подробности: %s", error)
            ok = False

    print("============================\n")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Система формирования конспектов по аудиозаписям лекций")
    parser.add_argument("--check", action="store_true", help="только проверить компоненты и выйти")
    args = parser.parse_args()

    try:
        settings = config.load_config()
    except config.ConfigError as error:
        setup_logging()
        print(f"\n[ОШИБКА НАСТРОЕК] {error}\n")
        return 1

    setup_logging(settings.log_level, secret=settings.telegram_bot_token)

    audio_processor = audio_module.AudioProcessor(
        ffmpeg_path=settings.ffmpeg_path,
        ffprobe_path=settings.ffprobe_path,
        max_duration_seconds=settings.max_audio_minutes * 60,
        noise_reduction=settings.noise_reduction,
        remove_long_pauses=settings.remove_long_pauses,
    )
    transcriber = transcription_module.Transcriber(
        model_name=settings.whisper_model,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
        language=settings.whisper_language,
        beam_size=settings.whisper_beam_size,
        chunk_minutes=settings.whisper_chunk_minutes,
        download_dir=settings.whisper_download_dir,
    )
    llm = cleaning_module.OllamaClient(
        base_url=settings.ollama_url,
        model=settings.ollama_model,
        timeout_seconds=settings.ollama_timeout_seconds,
        num_ctx=settings.ollama_num_ctx,
    )
    cleaner = cleaning_module.LectureCleaner(llm)
    conspect_builder = conspect_module.ConspectBuilder(llm)
    pdf_renderer = pdf_module.PdfRenderer(font_path=settings.font_path)

    if not check_components(settings, audio_processor, pdf_renderer, llm, transcriber):
        print("Исправьте ошибки выше (см. раздел «Частые ошибки» в README.md) и запустите снова.")
        return 1

    if args.check:
        print("Проверка завершена. Для запуска бота выполните: python main.py")
        return 0

    pipeline = LecturePipeline(settings, audio_processor, transcriber, cleaner, conspect_builder, pdf_renderer)
    bot = bot_module.ConspectatorBot(
        token=settings.telegram_bot_token,
        pipeline=pipeline,
        modes=config.MODES,
        mode_descriptions=config.MODE_DESCRIPTIONS,
        supported_extensions=config.SUPPORTED_EXTENSIONS,
        max_file_size_mb=settings.max_file_size_mb,
        proxy=settings.telegram_proxy,
    )

    try:
        bot.run()
    except KeyboardInterrupt:
        pass
    except Exception as error:
        text = str(error).lower()
        if "unauthorized" in text or "rejected" in text or "invalid token" in text or "not found" in text:
            print("\n[ОШИБКА] Telegram не принял токен. Проверьте TELEGRAM_BOT_TOKEN в .env.\n")
        elif "network" in text or "connect" in text or "timed out" in text:
            print("\n[ОШИБКА] Нет связи с Telegram. Проверьте интернет.\n")
        logger.exception("Бот остановлен из-за ошибки")
        return 1
    finally:
        removed = audio_module.cleanup_old_workspaces(settings.temp_dir)
        if removed:
            logger.info("Удалено временных папок при выходе: %d", removed)

    print("Бот остановлен.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
