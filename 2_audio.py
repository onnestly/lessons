
import json
import logging
import re
import shutil
import subprocess
import tempfile
import time
import wave
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1

WORKSPACE_PREFIX = "job_"


class AudioError(Exception):
    def __init__(self, user_message: str, technical_details: str = ""):
        super().__init__(technical_details or user_message)
        self.user_message = user_message


@dataclass
class PreparedAudio:
    path: Path
    original_duration_seconds: float
    duration_seconds: float


def safe_filename(name: str, default: str = "file") -> str:
    name = Path(str(name)).name
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    name = name.lstrip(".")
    name = name[-60:]
    return name or default


class TempWorkspace:
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.path = None

    def __enter__(self) -> "TempWorkspace":
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.path = Path(tempfile.mkdtemp(prefix=WORKSPACE_PREFIX, dir=self.base_dir))
        logger.debug("Создана временная папка %s", self.path)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.cleanup()
        return False

    def file(self, name: str) -> Path:
        if self.path is None:
            raise RuntimeError("TempWorkspace нужно использовать через with")
        return self.path / safe_filename(name)

    def cleanup(self) -> None:
        if self.path is None or not self.path.exists():
            return
        for _ in range(3):
            shutil.rmtree(self.path, ignore_errors=True)
            if not self.path.exists():
                logger.debug("Временная папка удалена: %s", self.path)
                return
            time.sleep(0.5)
        logger.warning("Не удалось полностью удалить временную папку %s", self.path)


def cleanup_old_workspaces(base_dir: Path) -> int:
    base_dir = Path(base_dir)
    if not base_dir.exists():
        return 0
    removed = 0
    for item in base_dir.iterdir():
        if item.is_dir() and item.name.startswith(WORKSPACE_PREFIX):
            shutil.rmtree(item, ignore_errors=True)
            removed += 1
    return removed


class AudioProcessor:
    def __init__(
        self,
        ffmpeg_path: str = "ffmpeg",
        ffprobe_path: str = "ffprobe",
        max_duration_seconds: int = 7200,
        noise_reduction: bool = True,
        remove_long_pauses: bool = True,
    ):
        self.ffmpeg_path = ffmpeg_path
        self.ffprobe_path = ffprobe_path
        self.max_duration_seconds = max_duration_seconds
        self.noise_reduction = noise_reduction
        self.remove_long_pauses = remove_long_pauses


    def check_ffmpeg(self) -> str:
        version_line = ""
        for program in (self.ffmpeg_path, self.ffprobe_path):
            try:
                result = subprocess.run(
                    [program, "-version"],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=20,
                )
            except (FileNotFoundError, PermissionError, OSError) as error:
                raise AudioError(
                    "На компьютере не установлен FFmpeg — без него нельзя обработать аудио.",
                    f"Не найдена программа '{program}': {error}",
                ) from error
            if result.returncode != 0:
                raise AudioError(
                    "FFmpeg установлен неправильно.",
                    f"'{program} -version' вернул код {result.returncode}: {result.stderr[-500:]}",
                )
            if program == self.ffmpeg_path:
                version_line = (result.stdout.splitlines() or [""])[0]
        return version_line


    def prepare(self, input_path: Path, output_path: Path) -> PreparedAudio:
        input_path = Path(input_path)
        output_path = Path(output_path)

        if not input_path.exists() or input_path.stat().st_size == 0:
            raise AudioError("Файл пустой или не был получен.", f"Нет файла {input_path}")

        original_duration = self._probe_duration(input_path)
        if original_duration > self.max_duration_seconds:
            raise AudioError(
                f"Запись слишком длинная ({original_duration / 60:.0f} мин). "
                f"Максимум — {self.max_duration_seconds / 60:.0f} мин. Разделите её на части."
            )

        attempts = []
        if self.noise_reduction or self.remove_long_pauses:
            attempts.append(("полная очистка", self._build_filters(self.noise_reduction, self.remove_long_pauses)))
        if self.remove_long_pauses:
            attempts.append(("без удаления пауз", self._build_filters(self.noise_reduction, False)))
        attempts.append(("простая конвертация", self._build_filters(False, False)))

        last_error = ""
        for attempt_name, filters in attempts:
            ok, details = self._run_ffmpeg(input_path, output_path, filters, original_duration)
            if not ok:
                last_error = details
                logger.warning("FFmpeg (%s) завершился с ошибкой, пробую проще. %s", attempt_name, details)
                continue

            duration = self._wav_duration(output_path)
            if duration < 0.5 and original_duration > 1.0:
                last_error = f"После обработки ({attempt_name}) осталось {duration:.2f} с звука"
                logger.warning(last_error)
                continue

            logger.info(
                "Аудио подготовлено (%s): было %.1f с, стало %.1f с",
                attempt_name, original_duration, duration,
            )
            return PreparedAudio(
                path=output_path,
                original_duration_seconds=original_duration or duration,
                duration_seconds=duration,
            )

        raise AudioError(
            "Не удалось обработать аудио: файл повреждён, пустой или в нём нет звука.",
            last_error,
        )


    def _probe_duration(self, input_path: Path) -> float:
        command = [
            self.ffprobe_path,
            "-v", "error",
            "-show_entries", "format=duration:stream=codec_type",
            "-of", "json",
            str(input_path),
        ]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=60,
            )
        except FileNotFoundError as error:
            raise AudioError("На компьютере не установлен FFmpeg (ffprobe).", str(error)) from error
        except subprocess.TimeoutExpired as error:
            raise AudioError("Не удалось прочитать файл: он повреждён или слишком сложный.", str(error)) from error

        if result.returncode != 0:
            raise AudioError(
                "Файл повреждён или его формат не поддерживается.",
                f"ffprobe: {result.stderr[-800:]}",
            )

        try:
            info = json.loads(result.stdout or "{}")
        except json.JSONDecodeError as error:
            raise AudioError("Файл повреждён или его формат не поддерживается.", str(error)) from error

        streams = info.get("streams", [])
        if not any(stream.get("codec_type") == "audio" for stream in streams):
            raise AudioError("В файле нет звуковой дорожки. Отправьте аудиозапись лекции.")

        try:
            return float(info.get("format", {}).get("duration", 0) or 0)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _build_filters(noise_reduction: bool, remove_long_pauses: bool) -> str:
        filters = []
        if noise_reduction:
            filters += [
                "highpass=f=80",
                "lowpass=f=7800",
                "afftdn=nr=12:nf=-40:tn=1",
            ]
        filters.append("loudnorm=I=-16:TP=-1.5:LRA=11")
        if remove_long_pauses:
            filters.append(
                "silenceremove=stop_periods=-1:stop_duration=2:stop_threshold=-45dB:stop_silence=0.7"
            )
        return ",".join(filters)

    def _run_ffmpeg(self, input_path: Path, output_path: Path, filters: str, duration: float):
        command = [
            self.ffmpeg_path,
            "-hide_banner", "-nostdin", "-y",
            "-i", str(input_path),
            "-map", "0:a:0",
            "-vn", "-sn", "-dn",
        ]
        if filters:
            command += ["-af", filters]
        command += [
            "-ac", str(TARGET_CHANNELS),
            "-ar", str(TARGET_SAMPLE_RATE),
            "-c:a", "pcm_s16le",
            str(output_path),
        ]

        timeout = 120 + int(duration * 1.5)
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
            )
        except FileNotFoundError as error:
            raise AudioError("На компьютере не установлен FFmpeg.", str(error)) from error
        except subprocess.TimeoutExpired:
            return False, f"FFmpeg не успел за {timeout} с"

        if result.returncode != 0 or not output_path.exists() or output_path.stat().st_size <= 44:
            return False, f"код {result.returncode}: {result.stderr[-1500:]}"
        return True, ""

    @staticmethod
    def _wav_duration(wav_path: Path) -> float:
        try:
            with wave.open(str(wav_path), "rb") as wav_file:
                return wav_file.getnframes() / float(wav_file.getframerate())
        except (wave.Error, EOFError, OSError) as error:
            logger.warning("Не удалось прочитать WAV %s: %s", wav_path, error)
            return 0.0
