### Sample Audio Descriptions - Whisper Testing

This document helps you test the Whisper-based voice transcription module.

Supported formats: .wav, .mp3, .m4a, .flac, .ogg

Recommended settings:
- Sample rate: 16 kHz or 44.1 kHz
- Mono preferred (stereo also supported)
- Bit depth: 16-bit PCM for .wav

#### Example descriptions (English)
- "Adult male, 45 years old, thick mustache, large ears, dark complexion."
- "Middle-aged Indian female, bindi on forehead, smiling, natural lighting."

#### Example descriptions (Hindi)
- "वयस्क पुरुष, लगभग 45 वर्ष, घनी मूंछ, बड़े कान, सांवला रंग।"
- "मध्यम आयु की भारतीय महिला, माथे पर बिंदी, मुस्कुराती हुई।"

#### File sizes
- 5 seconds of mono 16 kHz WAV ~ 160 KB
- MP3/M4A are smaller but may add compression artifacts

#### Generating test audio
1) Use your phone's voice recorder and export to .m4a or .wav
2) Or on Linux with microphone:
```bash
pip install sounddevice soundfile
python - << 'PY'
import sounddevice as sd, soundfile as sf
sr = 16000
print("Recording 5s ...")
data = sd.rec(int(5*sr), samplerate=sr, channels=1, dtype='float32')
sd.wait()
sf.write('sample_hi.wav', data, sr)
print("Saved sample_hi.wav")
PY
```

#### Running the test
```bash
# Auto language
python -m src.test_voice --audio_file data/sample_hi.wav

# Force language to Hindi
python -m src.test_voice --audio_file data/sample_hi.wav --language hi
```

#### Expected output
- Transcript text
- Detected language (e.g., "hi")
- Confidence score (0.0–1.0)
- Processing time

#### Integration with Text-to-Face
Use the transcript as the description:
```bash
python -m src.test_text_to_face --description "PASTE TRANSCRIPT HERE" --num_images 2
```

#### Troubleshooting
- Install Whisper: `pip install -U openai-whisper`
- CPU is slower; use GPU if available (CUDA)
- If recognition is poor, try:
  - A quieter environment
  - Speaking clearly and slowly
  - Larger model size: `--model_size small` or `--model_size medium`


