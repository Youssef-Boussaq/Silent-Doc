"""
Silent Doctor — Speech-to-Text Module
=======================================
Local speech recognition using faster-whisper (CTranslate2-based Whisper).

Fully offline — no API calls required.

Usage:
    stt = SpeechToText()
    result = stt.transcribe("audio.wav")
    print(result["text"], result["language"])
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np

from config.settings import (
    WHISPER_COMPUTE_TYPE,
    WHISPER_DEVICE,
    WHISPER_MODEL_SIZE,
)
from utils.helpers import setup_logger

logger = setup_logger(__name__)


class SpeechToText:
    """
    Local speech-to-text using faster-whisper.

    Faster-whisper uses CTranslate2 under the hood for efficient
    inference with INT8 quantization — ideal for low-resource devices.
    """

    def __init__(
        self,
        model_size: str = WHISPER_MODEL_SIZE,
        device: str = WHISPER_DEVICE,
        compute_type: str = WHISPER_COMPUTE_TYPE,
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = None

    @property
    def model(self):
        """Lazy-load the Whisper model on first use."""
        if self._model is None:
            logger.info(
                f"Loading Whisper model: {self.model_size} "
                f"(device={self.device}, compute={self.compute_type})"
            )
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
            logger.info("✅ Whisper model loaded.")
        return self._model

    def transcribe(
        self,
        audio_input: Union[str, Path, np.ndarray],
        language: Optional[str] = None,
    ) -> dict:
        """
        Transcribe audio to text.

        Args:
            audio_input: Path to audio file, or numpy float32 array.
            language: Optional language hint (e.g., "ar" for Arabic).
                      If None, Whisper auto-detects the language.

        Returns:
            dict with keys:
                - text (str): Full transcribed text
                - language (str): Detected language code
                - segments (list): Individual segments with timestamps
        """
        # Handle file path vs numpy array
        if isinstance(audio_input, (str, Path)):
            audio_path = str(audio_input)
            if not Path(audio_path).exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            source = audio_path
        else:
            source = audio_input

        logger.info("🎧 Transcribing audio ...")

        segments_iter, info = self.model.transcribe(
            source,
            language=language,
            beam_size=5,
            vad_filter=True,  # Voice activity detection
        )

        # Collect all segments
        segments = []
        full_text_parts = []
        for segment in segments_iter:
            segments.append({
                "start": round(segment.start, 2),
                "end": round(segment.end, 2),
                "text": segment.text.strip(),
            })
            full_text_parts.append(segment.text.strip())

        full_text = " ".join(full_text_parts)
        detected_lang = info.language

        logger.info(
            f"✅ Transcription complete. "
            f"Language: {detected_lang}, Length: {len(full_text)} chars"
        )

        return {
            "text": full_text,
            "language": detected_lang,
            "segments": segments,
        }

    def transcribe_from_mic(self, duration: float = 5.0) -> dict:
        """
        Record from microphone and transcribe.

        Args:
            duration: Recording duration in seconds.

        Returns:
            Same dict as transcribe().
        """
        from utils.helpers import record_audio

        audio = record_audio(duration_seconds=duration)
        return self.transcribe(audio)


# ── Convenience function ────────────────────────────────────────────────

_cached_stt: Optional[SpeechToText] = None


def get_stt(**kwargs) -> SpeechToText:
    """Get or create a cached SpeechToText instance."""
    global _cached_stt
    if _cached_stt is None:
        _cached_stt = SpeechToText(**kwargs)
    return _cached_stt
