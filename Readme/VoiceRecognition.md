### Voice Recognition (OpenAI Whisper) - Witness Descriptions

This module transcribes multilingual audio (English, Hindi, Punjabi, Bengali, Tamil) into text using the local, offline Whisper models (no API key).

Files:
- `src/voice_processor.py`: VoiceDescriptionProcessor class (API)
- `src/test_voice.py`: CLI test for audio files, microphone input, or text fallback
- `data/sample_descriptions_audio.md`: Examples and testing instructions

#### Quick start
```bash
# Transcribe an audio file (auto detect language)
python -m src.test_voice --audio_file data/sample_hi.wav

# Force Hindi
python -m src.test_voice --audio_file data/sample_hi.wav --language hi

# Live mic for 5s (requires sounddevice & soundfile)
python -m src.test_voice --mic --duration 5
```

#### Installation
```bash
pip install -U openai-whisper
# Optional for mic recording:
pip install sounddevice soundfile
```

#### Parameters
- `--language`: Specify language code (en, hi, pa, bn, ta) or omit for auto
- `--model_size`: tiny/base/small/medium/large (bigger = better but slower)
- `--output`: Directory to save transcripts

#### Output
- Prints transcript, detected language, confidence, and processing time
- Saves a `.txt` transcript to the output directory

#### Integration with Text-to-Face
Use the transcribed text as the input description:
```bash
python -m src.test_text_to_face --description "PASTE TRANSCRIPT HERE" --num_images 2
```

#### Tips
- For noisy backgrounds, try a larger model (`--model_size small` or `medium`) and speak clearly
- CPU works but is slower; GPU recommended if available
- File formats: .wav, .mp3, .m4a, .flac, .ogg

#### Troubleshooting
- If import fails: `pip install -U openai-whisper`
- If mic recording fails: `pip install sounddevice soundfile` and ensure microphone permissions
- Low confidence: record closer to the mic, reduce background noise, or use larger model


