# NeuroscribeAI — Intelligent Meeting Assistant

Prosody-aware meeting-intelligence system. Records Google Meet / Zoom calls,
transcribes them, adds speaker diarization, intent detection and prosody
(pitch / energy / emotion / urgency) analysis, produces an LLM summary with
action items, caches derived features in an encrypted local vault, and exposes
a chatbot over the results — all driven from a Tkinter desktop UI.

## Architecture

```
Meeting URL ──Selenium──▶ browser joins call
                 │
   screen (OpenCV) + system audio (sounddevice) ──MoviePy──▶ recordings/<platform>_<ts>_final.mp4
                 │
     ┌───────────┼─────────────────────────────┐
     ▼           ▼                             ▼
 Groq Whisper    Deepgram nova-3         prosody_extraction_2.py
 (large-v3)      (diarization + intents) (Whisper + WhisperX + Praat + Librosa + TextBlob)
     │           │                             │
     └───────────┴───────────┬─────────────────┘
                             ▼
     features ──▶ SecureFeatureVault (AES-256-GCM encrypted cache, keyed by meeting id)
                             ▼
             recordings/<stem>_complete_analysis.txt
                             ▼
                 Groq llama-3.1-8b ──▶ summary + to-do list
                             ▼
             MeetingChatbot (Groq llama-3.3-70b) answers questions
```

### Tech stack by layer

| Layer | Module(s) | Tech |
|---|---|---|
| Desktop UI / orchestrator | `app.py` | Tkinter, threading |
| Meeting capture | `meet3.py` | Selenium, PyAutoGUI, OpenCV, sounddevice, soundfile, MoviePy, FFmpeg |
| ASR | `Transcript.py` | Groq API `whisper-large-v3`, pydub (chunking) |
| Prosody | `prosody_extraction_2.py` | Whisper + WhisperX (word alignment), Praat/Parselmouth, Librosa, TextBlob, regex lexicons |
| Diarization + intent | `DeepgramProcessor` in `app.py` | `deepgram-sdk==2.12.0`, model `nova-3` |
| Summarization | `summarize_transcript.py` | Groq `llama-3.1-8b-instant` |
| Chatbot | `meeting_chatbot.py` (shared) + `bot4.py`, `bot_UI.py`, `desktop_chatbot_tkinter*.py` | Groq `llama-3.3-70b-versatile` |
| Encrypted cache | `secure_lfv.py` | `cryptography` AES-256-GCM, PBKDF2-HMAC-SHA256 |
| Evaluation | `evaluation_suite.py`, `run_evaluation.py`, `signal_quality_eval.py` | WER/CER, SciPy, LLM-as-judge |

External APIs: **Groq** (`api.groq.com`) and **Deepgram** (`api.deepgram.com`).
Everything else (Praat/Librosa signal analysis, the feature vault) runs locally.

## Setup

Requires **Python 3.11** and **FFmpeg** on `PATH`.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (source .venv/bin/activate on Unix)
pip install -r requirements.txt
python -m textblob.download_corpora

cp .env.example .env              # then edit .env with your own keys
```

### Configuration (`.env`)

| Variable | Purpose |
|---|---|
| `GROQ_API_KEY` | **required** — LLM + Whisper ASR |
| `DEEPGRAM_API_KEY` | optional — diarization + intent (disabled if absent) |
| `LFV_PASSWORD` | optional — password for the encrypted feature vault |
| `ENABLE_LFV_CACHE` | optional — `true`/`false` |
| `NEUROSCRIBE_RECORDINGS_DIR` | optional — override the recordings/transcripts folder |
| `CHROME_PROFILE_PATH` | optional — Chrome profile for a logged-in recorder session |

> The `.env` file is git-ignored. Never commit real credentials.

## Usage

```bash
python app.py            # full desktop app (record / upload / analyse / chat)
python bot4.py           # terminal chatbot over ./recordings
python bot_UI.py         # Flask web chatbot  (FLASK_DEBUG=1 for debug mode)
python run_evaluation.py # evaluate the most recent processed meeting
```

## Repository

<https://github.com/anishkaagarwal/neuroscribe>

## Notes

- `Transcript/Transcript.py` is a legacy duplicate of `Transcript.py`; prefer the latter.
- The `evaluation_suite.py` overall score is the plain mean of the available
  dimension scores (0–100), no manual adjustments.
- No secrets, generated media, `recordings/`, `secure_vault/`, or virtualenvs
  are tracked — see `.gitignore`.
