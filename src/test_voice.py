#!/usr/bin/env python3
"""
Test script for voice recognition (Whisper).

Supports:
  - Transcribing an audio file
  - Optional live microphone input (if sounddevice is installed)
  - Text input fallback (for environments without audio)

Outputs:
  - Transcript text
  - Detected language
  - Confidence score
  - Timing
  - Saves transcript to a text file (if --output provided)

Usage:
  # Transcribe an audio file
  python -m src.test_voice --audio_file path/to/audio.wav --language hi --output output/transcripts

  # Auto-detect language
  python -m src.test_voice --audio_file path/to/audio.wav

  # Live mic (if supported) for 5 seconds
  python -m src.test_voice --mic --duration 5 --output output/transcripts

  # Text fallback (no audio): just writes the provided text
  python -m src.test_voice --text "Adult male, thick mustache, Indian, neutral expression" --output output/transcripts
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from voice_processor import VoiceDescriptionProcessor  # noqa: E402


def _record_microphone(duration: int = 5, samplerate: int = 16000, channels: int = 1) -> Optional[Path]:
    """
    Record audio from microphone to a temporary .wav file (if supported).
    Requires: sounddevice, soundfile.
    """
    try:
        import sounddevice as sd
        import soundfile as sf
        import numpy as np
    except Exception:
        print("✗ Microphone recording requires: pip install sounddevice soundfile")
        return None

    print(f"Recording from microphone for {duration}s at {samplerate} Hz...")
    data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=channels, dtype="float32")
    sd.wait()
    tmp_path = Path("mic_recording.wav")
    sf.write(str(tmp_path), data, samplerate)
    print(f"✓ Saved temporary recording: {tmp_path}")
    return tmp_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Test Whisper-based voice transcription",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # File transcription (auto language)
  python -m src.test_voice --audio_file data/sample_hi.wav

  # File transcription (force Hindi)
  python -m src.test_voice --audio_file data/sample_hi.wav --language hi

  # Microphone (5s)
  python -m src.test_voice --mic --duration 5

  # Text fallback
  python -m src.test_voice --text "Adult male, 45, thick mustache"
        """,
    )
    parser.add_argument("--audio_file", type=str, help="Path to audio file (.wav, .mp3, .m4a, .flac, .ogg)")
    parser.add_argument("--language", type=str, help="Language code (en, hi, pa, bn, ta). Default: auto-detect")
    parser.add_argument("--output", type=str, default="output/transcripts", help="Directory to save transcript")
    parser.add_argument("--mic", action="store_true", help="Record from microphone")
    parser.add_argument("--duration", type=int, default=5, help="Microphone recording duration in seconds")
    parser.add_argument("--model_size", type=str, default="base", help="Whisper model size (tiny, base, small, medium, large)")
    parser.add_argument("--text", type=str, help="Text input fallback if no audio is provided")

    args = parser.parse_args()

    # Determine input mode
    audio_path: Optional[Path] = None
    if args.audio_file:
        audio_path = Path(args.audio_file)
    elif args.mic:
        audio_path = _record_microphone(duration=args.duration)
        if audio_path is None:
            return 1
    elif args.text:
        # Text fallback
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "transcript_text_fallback.txt"
        out_file.write_text(args.text.strip() + "\n", encoding="utf-8")
        print(f"✓ Wrote text fallback transcript: {out_file}")
        return 0
    else:
        parser.print_help()
        print("\n✗ Provide --audio_file, --mic, or --text")
        return 1

    # Initialize processor
    try:
        processor = VoiceDescriptionProcessor(model_size=args.model_size)
    except Exception as e:
        print(f"\n✗ Failed to load Whisper: {e}")
        print("  Install with: pip install -U openai-whisper")
        return 1

    # Transcribe
    try:
        result = processor.transcribe_file(audio_path, language=args.language)
    except Exception as e:
        print(f"\n✗ Transcription error: {e}")
        return 1

    # Print results
    print("\n===== Transcription Result =====")
    print(f"  File: {audio_path.name}")
    print(f"  Language: {result.language}")
    print(f"  Confidence: {result.confidence:.2f}")
    print(f"  Time: {result.processing_time:.2f}s")
    print(f"\nTranscript:\n{result.text}\n")

    # Save transcript
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{audio_path.stem}_transcript.txt"
    content = f"Language: {result.language}\nConfidence: {result.confidence:.2f}\nTime: {result.processing_time:.2f}s\n\nTranscript:\n{result.text}\n"
    out_file.write_text(content, encoding="utf-8")
    print(f"✓ Saved transcript: {out_file}")

    # Integration tip
    print("\nTip: Use this transcript as the description for text-to-face generation:")
    print("  python -m src.test_text_to_face --description \"<paste transcript here>\" --num_images 2\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())


