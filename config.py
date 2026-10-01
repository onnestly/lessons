
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent

MODES = {
    "detailed": "Подробный",
    "normal": "Обычный",
    "short": "Краткий",
    "topic": "По теме",
}

MODE_DESCRIPTIONS = {
    "detailed": "практически вся информация лекции, но структурированная.",
    "normal": "основные мысли, определения и примеры.",
    "short": "только ключевые тезисы.",
    "topic": "вы указываете, что именно важно найти в лекции.",
}

SUPPORTED_EXTENSIONS = (
    "mp3", "wav", "m4a", "ogg", "oga", "opus", "flac", "aac",
    "wma", "amr", "webm", "mp4", "3gp",
)


class ConfigError(Exception):
    pass


@dataclass(frozen=True)
class Config:
    telegram_bot_token: str = field(repr=False)
    telegram_proxy: str

    whisper_model: str
    whisper_device: str
    whisper_compute_type: str
    whisper_language: str
    whisper_beam_size: int
    whisper_chunk_minutes: int
    whisper_download_dir: str

    ollama_url: str
    ollama_model: str
    ollama_timeout_seconds: int
    ollama_num_ctx: int

    ffmpeg_path: str
    ffprobe_path: str
    noise_reduction: bool
    remove_long_pauses: bool

    max_file_size_mb: int
    max_audio_minutes: int

    temp_dir: Path
    font_path: str
    log_level: str


def _get_str(name: str, default: str) -> str:
    value = os.getenv(name, "").strip()
    return value if value else default


def _get_int(name: str, default: int, minimum: int = 1, maximum: int = 10**9) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as error:
        raise ConfigError(f"В .env параметр {name} должен быть целым числом, сейчас: {raw!r}") from error
    if not minimum <= value <= maximum:
        raise ConfigError(f"В .env параметр {name} должен быть от {minimum} до {maximum}, сейчас: {value}")
    return value


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    if raw in ("1", "true", "yes", "on", "да"):
        return True
    if raw in ("0", "false", "no", "off", "нет"):
        return False
    raise ConfigError(f"В .env параметр {name} должен быть true или false, сейчас: {raw!r}")


def load_config() -> Config:
    env_file = PROJECT_DIR / ".env"
    if not env_file.exists():
        raise ConfigError(
            f"Не найден файл настроек {env_file}.\n"
            "Скопируйте файл .env.example, переименуйте копию в .env и вставьте туда токен бота."
        )
    load_dotenv(env_file)

    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token or token == "your_token_here":
        raise ConfigError(
            "В файле .env не указан TELEGRAM_BOT_TOKEN.\n"
            "Получите токен у @BotFather в Telegram и вставьте его в .env."
        )
    if ":" not in token:
        raise ConfigError("TELEGRAM_BOT_TOKEN в .env выглядит неправильно (должен быть вида 123456:ABC...).")

    whisper_device = _get_str("WHISPER_DEVICE", "cpu").lower()
    if whisper_device not in ("cpu", "cuda", "auto"):
        raise ConfigError("WHISPER_DEVICE в .env должен быть cpu, cuda или auto.")
    default_compute = "float16" if whisper_device == "cuda" else "int8"

    temp_dir = Path(_get_str("TEMP_DIR", "temp"))
    if not temp_dir.is_absolute():
        temp_dir = PROJECT_DIR / temp_dir

    whisper_download_dir = Path(_get_str("WHISPER_DOWNLOAD_DIR", "models"))
    if not whisper_download_dir.is_absolute():
        whisper_download_dir = PROJECT_DIR / whisper_download_dir

    log_level = _get_str("LOG_LEVEL", "INFO").upper()
    if log_level not in ("DEBUG", "INFO", "WARNING", "ERROR"):
        log_level = "INFO"

    return Config(
        telegram_bot_token=token,
        telegram_proxy=_get_str("TELEGRAM_PROXY", ""),
        whisper_model=_get_str("WHISPER_MODEL", "small"),
        whisper_device=whisper_device,
        whisper_compute_type=_get_str("WHISPER_COMPUTE_TYPE", default_compute),
        whisper_language=_get_str("WHISPER_LANGUAGE", "ru"),
        whisper_beam_size=_get_int("WHISPER_BEAM_SIZE", 5, 1, 10),
        whisper_chunk_minutes=_get_int("WHISPER_CHUNK_MINUTES", 10, 1, 60),
        whisper_download_dir=str(whisper_download_dir),
        ollama_url=_get_str("OLLAMA_URL", "http://localhost:11434"),
        ollama_model=_get_str("OLLAMA_MODEL", "qwen2.5:7b"),
        ollama_timeout_seconds=_get_int("OLLAMA_TIMEOUT_SECONDS", 900, 30),
        ollama_num_ctx=_get_int("OLLAMA_NUM_CTX", 8192, 2048, 131072),
        ffmpeg_path=_get_str("FFMPEG_PATH", "ffmpeg"),
        ffprobe_path=_get_str("FFPROBE_PATH", "ffprobe"),
        noise_reduction=_get_bool("NOISE_REDUCTION", True),
        remove_long_pauses=_get_bool("REMOVE_LONG_PAUSES", True),
        max_file_size_mb=_get_int("MAX_FILE_SIZE_MB", 20, 1, 2000),
        max_audio_minutes=_get_int("MAX_AUDIO_MINUTES", 120, 1, 600),
        temp_dir=temp_dir,
        font_path=_get_str("FONT_PATH", ""),
        log_level=log_level,
    )
