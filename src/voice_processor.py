"""
Voice Recognition Module using OpenAI Whisper

This module provides a `VoiceDescriptionProcessor` class that transcribes
multilingual audio witness descriptions into text using the open-source
Whisper models (offline, no API key required).

Supported languages (examples): English (en), Hindi (hi), Punjabi (pa),
Bengali (bn), Tamil (ta). Language can be auto-detected or specified.

Inputs:
  - Audio files: .wav, .mp3, .m4a, .flac, .ogg (single-channel or stereo)

Outputs:
  - Transcript text
  - Detected/selected language
  - Confidence score (derived from segment log probabilities)
  - Timing information

Author: Forensic Reconstruction System
Date: 2025
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


SUPPORTED_AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
SUPPORTED_LANGS = {"en", "hi", "pa", "bn", "ta"}


def _estimate_confidence(segments: Optional[list]) -> float:
    """
    Estimate a confidence score from Whisper segments.
    Whisper (openai-whisper) exposes per-segment avg_logprob and no_speech_prob.
    We compute a heuristic confidence in [0,1].
    """
    if not segments:
        return 0.0
    logprobs = []
    penalties = []
    for seg in segments:
        if "avg_logprob" in seg and seg["avg_logprob"] is not None:
            logprobs.append(seg["avg_logprob"])
        if "no_speech_prob" in seg and seg["no_speech_prob"] is not None:
            penalties.append(seg["no_speech_prob"])
    if not logprobs:
        # Fallback
        return float(max(0.0, 1.0 - np.mean(penalties)) if penalties else 0.5)
    avg_lp = float(np.mean(logprobs))  # typically in [-5, 0]
    # Map avg logprob to [0,1] with a soft function
    conf = 1.0 / (1.0 + np.exp(-2.0 * (avg_lp + 1.5)))
    if penalties:
        conf *= float(max(0.0, 1.0 - np.mean(penalties)))
    return float(np.clip(conf, 0.0, 1.0))


@dataclass
class TranscriptionResult:
    text: str
    language: str
    confidence: float
    processing_time: float
    segments: Optional[list]


class VoiceDescriptionProcessor:
    """
    Speech-to-text processor using local Whisper models.
    - Works offline (no API key)
    - CPU and GPU supported
    - Multilingual
    """

    def __init__(
        self,
        model_size: str = "base",
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
        vad_filter: bool = False,
    ):
        """
        Initialize Whisper model.

        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
                        'base' is a good trade-off; 'small' or 'medium' for higher quality.
            device: 'cuda', 'cpu', or None for auto
            compute_type: Reserved for alternative backends; ignored for openai-whisper
            vad_filter: If True, enable VAD-like behavior via higher no_speech_threshold
        """
        try:
            import whisper  # type: ignore
        except ImportError as e:
            raise ImportError(
                "openai-whisper not installed. Install with: pip install -U openai-whisper"
            ) from e

        self.whisper = whisper
        self.model_size = model_size
        self.vad_filter = vad_filter

        if device is None:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info("Loading Whisper model: %s on %s", model_size, self.device)
        t0 = time.time()
        self.model = whisper.load_model(model_size, device=self.device)
        logger.info("✓ Whisper model loaded in %.2fs", time.time() - t0)

    def transcribe_file(
        self,
        audio_path: Union[str, Path],
        language: Optional[str] = None,
        task: str = "transcribe",
        temperature: float = 0.0,
    ) -> TranscriptionResult:
        """
        Transcribe an audio file to text.

        Args:
            audio_path: Path to audio file (.wav, .mp3, .m4a, .flac, .ogg)
            language: Language code (en, hi, pa, bn, ta) or None for auto
            task: 'transcribe' or 'translate' (keep 'transcribe' for original language)
            temperature: Decoding temperature; higher can help with noise (0.0 - 1.0)

        Returns:
            TranscriptionResult with text, language, confidence, timing, segments
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        if audio_path.suffix.lower() not in SUPPORTED_AUDIO_EXTS:
            raise ValueError(f"Unsupported audio format: {audio_path.suffix}. Supported: {sorted(SUPPORTED_AUDIO_EXTS)}")

        # Validate language input
        lang_param: Optional[str]
        if language is None:
            lang_param = None  # auto-detect
        else:
            lang = language.lower()
            if lang not in SUPPORTED_LANGS:
                logger.warning("Language '%s' not in supported set %s. Proceeding anyway.", lang, sorted(SUPPORTED_LANGS))
            lang_param = lang

        # Whisper options
        options = dict(
            language=lang_param,
            task=task,
            temperature=temperature,
            best_of=5 if temperature > 0.0 else 1,
            beam_size=5 if temperature == 0.0 else None,
            condition_on_previous_text=True,
            fp16=(self.device == "cuda"),
            no_speech_threshold=0.5 if self.vad_filter else 0.3,
            logprob_threshold=-1.2,
            compression_ratio_threshold=2.4,
        )

        try:
            t0 = time.time()
            logger.info("Transcribing: %s", audio_path.name)
            result = self.model.transcribe(str(audio_path), **options)
            elapsed = time.time() - t0
        except Exception as e:
            logger.error("Transcription failed: %s", e)
            raise RuntimeError(f"Transcription failed: {e}") from e

        text = (result.get("text") or "").strip()
        detected_language = result.get("language") or (language or "unknown")
        segments = result.get("segments")
        confidence = _estimate_confidence(segments)

        return TranscriptionResult(
            text=text,
            language=detected_language,
            confidence=confidence,
            processing_time=elapsed,
            segments=segments,
        )


