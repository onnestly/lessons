
import logging
import os
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000

INITIAL_PROMPT = "Урок. Преподаватель объясняет новую тему, приводит определения и примеры."


class TranscriptionError(Exception):
    def __init__(self, user_message: str, technical_details: str = ""):
        super().__init__(technical_details or user_message)
        self.user_message = user_message


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class Transcript:
    segments: list = field(default_factory=list)
    language: str = "ru"
    duration_seconds: float = 0.0

    @property
    def text(self) -> str:
        return " ".join(segment.text for segment in self.segments).strip()

    def is_empty(self) -> bool:
        return not self.text


def format_timestamp(seconds: float) -> str:
    seconds = int(seconds)
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


class Transcriber:
    def __init__(
        self,
        model_name: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str = "ru",
        beam_size: int = 5,
        chunk_minutes: int = 10,
        download_dir: str = "",
    ):
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.language = language or None
        self.beam_size = beam_size
        self.chunk_seconds = max(60, chunk_minutes * 60)
        self.download_dir = download_dir or None
        self._model = None


    def load_model(self) -> None:
        if self._model is not None:
            return

        if self.download_dir:
            Path(self.download_dir).mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("HF_HOME", str(Path(self.download_dir) / "huggingface"))
            os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

        try:
            from faster_whisper import WhisperModel
        except ImportError as error:
            raise TranscriptionError(
                "Whisper не установлен. Выполните: pip install -r requirements.txt",
                str(error),
            ) from error

        logger.info(
            "Загружаю модель Whisper '%s' (устройство: %s, точность: %s). "
            "При первом запуске модель скачивается — это может занять несколько минут...",
            self.model_name, self.device, self.compute_type,
        )
        started = time.time()
        try:
            self._model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
                download_root=self.download_dir,
            )
        except Exception as error:
            text = str(error).lower()
            if "cuda" in text or "cublas" in text or "cudnn" in text:
                user_message = (
                    "Whisper не смог использовать видеокарту. "
                    "Укажите WHISPER_DEVICE=cpu в файле .env и перезапустите бота."
                )
            elif "connection" in text or "internet" in text or "resolve" in text or "hub" in text:
                user_message = (
                    "Не удалось скачать модель Whisper. Для первого запуска нужен интернет."
                )
            else:
                user_message = (
                    f"Не удалось загрузить модель Whisper '{self.model_name}'. "
                    "Проверьте название модели в файле .env."
                )
            raise TranscriptionError(user_message, repr(error)) from error

        logger.info("Модель Whisper загружена за %.1f с", time.time() - started)


    def transcribe(self, wav_path: Path, on_progress=None) -> Transcript:
        self.load_model()

        samples = self._read_wav(Path(wav_path))
        duration = len(samples) / SAMPLE_RATE
        cut_points = self._find_cut_points(samples)
        total_parts = len(cut_points) - 1
        logger.info("Распознаю речь: %.1f с аудио, частей: %d", duration, total_parts)

        transcript = Transcript(language=self.language or "auto", duration_seconds=duration)
        started = time.time()

        for index in range(total_parts):
            start_sample = cut_points[index]
            end_sample = cut_points[index + 1]
            chunk = samples[start_sample:end_sample].astype(np.float32) / 32768.0
            offset = start_sample / SAMPLE_RATE

            try:
                segments, info = self._model.transcribe(
                    chunk,
                    language=self.language,
                    beam_size=self.beam_size,
                    vad_filter=True,
                    vad_parameters={"min_silence_duration_ms": 700},
                    condition_on_previous_text=False,
                    initial_prompt=INITIAL_PROMPT,
                )
                for segment in segments:
                    text = segment.text.strip()
                    if text:
                        transcript.segments.append(
                            Segment(start=segment.start + offset, end=segment.end + offset, text=text)
                        )
                if self.language is None and index == 0:
                    transcript.language = info.language
            except MemoryError as error:
                raise TranscriptionError(
                    "Не хватило оперативной памяти для распознавания. "
                    "Выберите модель Whisper поменьше (например, base) в файле .env.",
                    repr(error),
                ) from error
            except Exception as error:
                raise TranscriptionError(
                    "Ошибка при распознавании речи. Попробуйте ещё раз или выберите другую модель Whisper.",
                    repr(error),
                ) from error

            if on_progress is not None:
                on_progress(index + 1, total_parts)

        logger.info(
            "Распознавание завершено за %.1f с, сегментов: %d",
            time.time() - started, len(transcript.segments),
        )

        if transcript.is_empty():
            raise TranscriptionError(
                "В записи не удалось распознать речь. Проверьте, что на записи слышен голос."
            )
        return transcript


    @staticmethod
    def _read_wav(wav_path: Path) -> np.ndarray:
        try:
            with wave.open(str(wav_path), "rb") as wav_file:
                if (
                    wav_file.getnchannels() != 1
                    or wav_file.getsampwidth() != 2
                    or wav_file.getframerate() != SAMPLE_RATE
                ):
                    raise TranscriptionError(
                        "Внутренняя ошибка: аудио подготовлено в неверном формате.",
                        f"Ожидался WAV 16 кГц моно 16 бит: {wav_path}",
                    )
                frames = wav_file.readframes(wav_file.getnframes())
        except (wave.Error, EOFError, OSError) as error:
            raise TranscriptionError("Не удалось прочитать подготовленное аудио.", repr(error)) from error

        samples = np.frombuffer(frames, dtype=np.int16)
        if samples.size == 0:
            raise TranscriptionError("Аудио пустое — в записи нет звука.")
        return samples

    def _find_cut_points(self, samples: np.ndarray, search_seconds: int = 20) -> list:
        total = len(samples)
        chunk = self.chunk_seconds * SAMPLE_RATE
        window = SAMPLE_RATE // 4
        points = [0]
        position = 0

        while total - position > chunk * 1.2:
            target = position + chunk
            search_start = max(position + chunk // 2, target - search_seconds * SAMPLE_RATE)
            region = samples[search_start:target].astype(np.float32)
            frame_count = len(region) // window
            if frame_count > 0:
                frames = region[: frame_count * window].reshape(frame_count, window)
                energy = (frames ** 2).mean(axis=1)
                quietest = int(np.argmin(energy))
                cut = search_start + quietest * window + window // 2
            else:
                cut = target
            points.append(cut)
            position = cut

        points.append(total)
        return points
