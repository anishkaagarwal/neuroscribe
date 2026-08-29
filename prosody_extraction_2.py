# -*- coding: utf-8 -*-
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prosody extraction single-file script.
Default audio (if --audio not provided):
    /content/Two south Bombay guys having verbal fight during  online class.mp3
Usage:
    python prosody_extraction.py --audio "/path/to/audio.mp3"
"""

import sys
import subprocess
import importlib
import argparse
import os
import glob

# ---- Helper to ensure packages are installed (minimal, only when missing) ----
def ensure_package(pkg_name, import_name=None, pip_name=None):
    import_name = import_name or pkg_name
    pip_name = pip_name or pkg_name
    try:
        importlib.import_module(import_name)
    except Exception:
        print(f"Installing {pip_name} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])

# Try to make the environment ready (you can remove these installs if you manage dependencies yourself)
ensure_package("torch")
ensure_package("librosa")
ensure_package("parselmouth", pip_name="praat-parselmouth")
ensure_package("whisper", pip_name="git+https://github.com/openai/whisper.git")
ensure_package("whisperx", pip_name="git+https://github.com/m-bain/whisperX.git")
ensure_package("textblob")
ensure_package("numpy")
ensure_package("matplotlib")

# Ensure textblob corpora (best-effort)
try:
    subprocess.check_call([sys.executable, "-m", "textblob.download_corpora"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
except Exception:
    pass

# ---- Imports (after installs) ----
import torch, whisper, whisperx, librosa, parselmouth
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
import json, re, math
import numpy as np
import matplotlib.pyplot as plt
from textblob import TextBlob
from collections import Counter, defaultdict

# ---------------------------
# Audio selection: CLI or default path
# ---------------------------
parser = argparse.ArgumentParser(description="Prosody extraction and annotation pipeline")
parser.add_argument("--audio", "-a", required=False, help="Path to input audio file (wav/mp3). If omitted, default path will be used.")
parser.add_argument("--whisper_model", default="small", help="Whisper model size to use (default: small)")
args = parser.parse_args()

# Default audio: override with the AUDIO env var, else fall back to the first
# audio file found in the current directory (see logic below).
DEFAULT_AUDIO = os.getenv("AUDIO", "")

if args.audio:
    audio_path = args.audio
    print(f"Using audio from CLI argument: {audio_path}")
else:
    # prefer DEFAULT_AUDIO if it exists
    if os.path.exists(DEFAULT_AUDIO):
        audio_path = DEFAULT_AUDIO
        print(f"No --audio provided. Using default audio path: {audio_path}")
    else:
        # fallback: try to pick first audio from current directory
        SUPPORTED_EXTS = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac")
        candidates = [f for f in glob.glob(os.path.join(".", "*")) if f.lower().endswith(SUPPORTED_EXTS)]
        candidates.sort()
        if candidates:
            audio_path = candidates[0]
            print(f"No --audio and default not found. Using first audio in cwd: {audio_path}")
        else:
            # interactive prompt if possible
            if sys.stdin and sys.stdin.isatty():
                audio_path = input("Enter path to audio file (wav/mp3): ").strip()
            else:
                raise SystemExit("No audio provided, default audio not found, and no files in cwd. Provide --audio or place an audio file in the current directory.")

WHISPER_MODEL = args.whisper_model

# --- Support video input: convert to mp3 if needed ---
if not os.path.exists(audio_path):
    raise SystemExit(f"Audio file not found: {audio_path}")

# If user passed a video file, convert it to a mono 16kHz mp3 for processing
_video_exts = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".flv", ".mpeg", ".mpg"}
_root, _ext = os.path.splitext(audio_path)
_ext = _ext.lower()

if _ext in _video_exts:
    mp3_path = _root + ".mp3"
    # If mp3 already exists, reuse it; otherwise run ffmpeg conversion
    if not os.path.exists(mp3_path):
        print(f"Converting video -> audio: '{audio_path}' -> '{mp3_path}' (requires ffmpeg on PATH)...")
        try:
            # Use libmp3lame if available; adjust bitrate/sample if you want.
            subprocess.run([
                "ffmpeg", "-y", "-i", audio_path,
                "-vn",                    # no video
                "-acodec", "libmp3lame",  # encode mp3
                "-ar", "16000",           # sample rate 16 kHz
                "-ac", "1",               # mono
                mp3_path
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            raise SystemExit(f"ffmpeg conversion failed: {e}\nMake sure ffmpeg is installed and on PATH.")
    else:
        print(f"Found existing converted file, using: {mp3_path}")
    # point pipeline to the extracted mp3
    audio_path = mp3_path


# -- Constants (kept from original) --
RATE = 16000
FRAME_LENGTH = 1024
HOP_LENGTH = 512

# --- Dataclass and helpers (kept intact) ---
@dataclass
class WordProsody:
    word: str
    start: float
    end: float
    duration: float
    pitch_median: float
    pitch_mean: float
    energy_mean: float
    emphasis: bool
    score: Dict[str, float]

def normalize_token(token: str):
    token = token.strip()
    token = re.sub(r"^[^\w']+|[^\w']+$", "", token)
    return token

def load_audio(audio_path: str, sr: int = RATE):
    y, sr_ret = librosa.load(audio_path, sr=sr, mono=True)
    return y, sr_ret

def extract_energy(y, sr):
    rms = librosa.feature.rms(y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=HOP_LENGTH, n_fft=FRAME_LENGTH)
    return rms, times

def extract_pitch_parselmouth(audio_path: str, pitch_floor: float = 75.0, pitch_ceiling: float = 500.0):
    snd = parselmouth.Sound(audio_path)
    pitch_obj = snd.to_pitch(time_step=None, pitch_floor=pitch_floor, pitch_ceiling=pitch_ceiling)
    times = pitch_obj.xs()
    values = pitch_obj.selected_array['frequency']
    values = np.array(values, dtype=float)
    values[values == 0] = np.nan
    return times, values

def transcribe_and_align(audio_path: str, model_size: str = WHISPER_MODEL, device: str = None, print_progress=True):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"🎙 Loading Whisper model '{model_size}' on {device}...")
    model = whisper.load_model(model_size, device=device)

    print("📝 Transcribing audio with Whisper...")
    result = model.transcribe(audio_path)

    print("🔗 Loading whisperx alignment model (this may download a large file)...")
    model_a, metadata = whisperx.load_align_model(language_code=result.get("language", "en"), device=device)

    if print_progress:
        print("🧩 Performing alignment with whisperx...")
    aligned_result = whisperx.align(result["segments"], model_a, metadata, audio_path, device,
                                    return_char_alignments=False, print_progress=print_progress)

    words = aligned_result.get("words", aligned_result.get("word_segments", []))
    aligned_segments = aligned_result.get("segments", result["segments"])
    return words, aligned_segments, result

def map_prosody_to_words(words, audio_path: str):
    y, sr = load_audio(audio_path)
    rms, rms_times = extract_energy(y, sr)
    pitch_times, pitch_vals = extract_pitch_parselmouth(audio_path)
    prosody_list = []
    for w in words:
        raw_token = w.get("word", "")
        token = normalize_token(raw_token)
        if token == "":
            continue
        start, end = float(w.get("start", 0.0)), float(w.get("end", 0.0))
        duration = max(1e-6, end - start)
        idx_e = np.where((rms_times >= start) & (rms_times <= end))[0]
        if len(idx_e) > 0:
            energy_mean = float(np.mean(rms[idx_e]))
        else:
            s = int(max(0, math.floor(start * sr)))
            e = int(min(len(y), math.ceil(end * sr)))
            energy_mean = float(np.mean(y[s:e]**2)) if e > s else 0.0
        idx_p = np.where((pitch_times >= start) & (pitch_times <= end))[0]
        if len(idx_p) > 0:
            pvals = pitch_vals[idx_p]
            pvals_nonan = pvals[~np.isnan(pvals)]
            pitch_mean = float(np.mean(pvals_nonan)) if len(pvals_nonan) > 0 else float("nan")
            pitch_median = float(np.median(pvals_nonan)) if len(pvals_nonan) > 0 else float("nan")
        else:
            pitch_mean = float("nan")
            pitch_median = float("nan")

        prosody_list.append(WordProsody(
            word=token,
            start=start,
            end=end,
            duration=duration,
            pitch_median=pitch_median,
            pitch_mean=pitch_mean,
            energy_mean=energy_mean,
            emphasis=False,
            score={}
        ))
    return prosody_list

def detect_emphasis(prosody_list: List[WordProsody], z_thresh=1.0, require_n_signals=1):
    pitch_arr = np.array([w.pitch_mean for w in prosody_list], dtype=float)
    energy_arr = np.array([w.energy_mean for w in prosody_list], dtype=float)
    dur_arr = np.array([w.duration for w in prosody_list], dtype=float)

    def zscore_safe(a):
        a_nonan = a[~np.isnan(a)]
        if len(a_nonan) == 0:
            return np.zeros_like(a)
        mu, sd = np.nanmean(a), np.nanstd(a)
        if sd == 0:
            return np.zeros_like(a)
        return (a - mu) / (sd + 1e-12)

    z_pitch = zscore_safe(pitch_arr)
    z_energy = zscore_safe(energy_arr)
    z_dur = zscore_safe(dur_arr)

    for i, w in enumerate(prosody_list):
        zp, ze, zd = float(np.nan_to_num(z_pitch[i])), float(np.nan_to_num(z_energy[i])), float(np.nan_to_num(z_dur[i]))
        signals = [zp > z_thresh, ze > z_thresh, zd > z_thresh]
        w.score = {"z_pitch": zp, "z_energy": ze, "z_duration": zd, "combined": float(max(0.0, zp) + max(0.0, ze) + max(0.0, zd))}
        w.emphasis = sum(signals) >= require_n_signals and (w.energy_mean > 1e-8)
    return prosody_list

def extractive_summary_from_segments(segments, prosody_list, top_k_sentences=3):
    emph_times = [(w.start, w.end) for w in prosody_list if w.emphasis]
    seg_scores = []
    for seg in segments:
        seg_start = seg.get("start", 0.0)
        seg_end = seg.get("end", 0.0)
        count = sum(1 for (s,e) in emph_times if s >= seg_start and s <= seg_end)
        seg_scores.append({"text": seg.get("text", "").strip(), "start": seg_start, "end": seg_end, "score": count})
    seg_scores_sorted = sorted(seg_scores, key=lambda x: (-x["score"], x["start"]))
    top = [s["text"] for s in seg_scores_sorted[:top_k_sentences]]
    return " ".join(top), seg_scores_sorted

def plot_prosody(audio_path, prosody_list, sr=RATE):
    y, _ = librosa.load(audio_path, sr=sr)
    t = np.linspace(0, len(y)/sr, num=len(y))
    rms = librosa.feature.rms(y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=HOP_LENGTH, n_fft=FRAME_LENGTH)
    pitch_times = []
    pitch_vals = []
    emph_times = []
    for w in prosody_list:
        if not math.isnan(w.pitch_mean):
            pitch_times.append((w.start + w.end)/2.0)
            pitch_vals.append(w.pitch_mean)
        if w.emphasis:
            emph_times.append((w.start, w.end, w.word))

    fig, axs = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    axs[0].plot(t, y)
    axs[0].set(title="Waveform", ylabel="amplitude")
    axs[1].plot(rms_times, rms)
    axs[1].set(title="RMS energy", ylabel="rms")
    axs[2].scatter(pitch_times, pitch_vals, s=10)
    axs[2].set(title="Pitch (word-level mean)", ylabel="Hz", xlabel="time (s)")

    for (s,e,token) in emph_times:
        for ax in axs:
            ax.axvspan(s, e, alpha=0.2, color='orange')
        if len(pitch_vals) > 0:
            axs[2].text((s+e)/2, max(pitch_vals)*0.95, token, rotation=90, verticalalignment='top', fontsize=8)

    plt.tight_layout()
    plt.show()

def save_results_json(audio_path, summary, prosody_list, out_file="prosody_results.json"):
    out = {
        "audio": os.path.abspath(audio_path),
        "summary": summary,
        "words": [asdict(w) for w in prosody_list]
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("Saved JSON to:", out_file)
    return out_file

def write_webvtt(prosody_list, out_vtt="prosody.vtt", window_ms=1200):
    def fmt(t):
        mm = int(t // 60)
        ss = int(t % 60)
        ms = int((t - int(t)) * 1000)
        return f"{mm:02d}:{ss:02d}.{ms:03d}"
    cues = []
    for w in prosody_list:
        if not w.emphasis: continue
        start = max(0.0, w.start - window_ms/2000.0)
        end = w.end + window_ms/2000.0
        text = f"... <b>{w.word}</b> ..."
        cues.append((start, end, text))
    with open(out_vtt, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for i, (s,e,t) in enumerate(cues):
            f.write(f"{i+1}\n")
            f.write(f"{fmt(s)} --> {fmt(e)}\n")
            f.write(t + "\n\n")
    print("Saved VTT to:", out_vtt)
    return out_vtt

def write_srt(prosody_list, out_srt="prosody.srt"):
    def fmt_srt(t):
        h = int(t//3600); m = int((t%3600)//60); s = int(t%60); ms = int((t - int(t))*1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
    entries = []
    idx = 1
    for w in prosody_list:
        start, end = w.start, w.end
        token = w.word.upper() if w.emphasis else w.word
        entries.append(f"{idx}\n{fmt_srt(start)} --> {fmt_srt(end)}\n{token}\n\n")
        idx += 1
    with open(out_srt, "w", encoding="utf-8") as fh:
        fh.writelines(entries)
    print("Saved SRT to:", out_srt)
    return out_srt

# ----------------------------
# Main pipeline run (same flow as your notebook)
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on device:", device)

words, segments, raw_whisper = transcribe_and_align(audio_path, model_size=WHISPER_MODEL, device=device, print_progress=True)
prosody_list = map_prosody_to_words(words, audio_path)
prosody_list = detect_emphasis(prosody_list, z_thresh=1.0, require_n_signals=1)
summary, scored_segments = extractive_summary_from_segments(segments, prosody_list, top_k_sentences=3)

for w in prosody_list:
    if w.emphasis:
        pitch_str = f"{w.pitch_mean:.1f}" if not np.isnan(w.pitch_mean) else "NaN"

# Commented out: disable writing/downloading non-.txt outputs (.json, .vtt, .srt)
# json_file = save_results_json(audio_path, summary, prosody_list, out_file="prosody_results.json")
# vtt_file = write_webvtt(prosody_list, out_vtt="prosody.vtt", window_ms=1200)
# srt_file = write_srt(prosody_list, out_srt="prosody.srt")

# --- Lexical sentiment and meeting lexicons (TextBlob) ---
EMOTION_KEYWORDS = {
    "frustration": {
        "blocked","blocking","blocker","stuck","can't reproduce","can't","don't know","no idea",
        "not working","broken","error","fail","failing","annoying","frustrating","frustration",
        "unable","won't","doesn't work","does not work","problem","problems"
    },
    "concern": {
        "concern","worried","worry","problematic","risky","risk","issue","issues","impact","regression",
        "security","privacy","data loss","compliance","breach","outage","latency","slow","downtime"
    },
    "confusion": {
        "confused","confusing","not sure","don't understand","don't get","what do you mean",
        "clarify","clarification","question","questions","how do we","how can we"
    },
    "agreement": {
        "yes","ok","okay","sounds good","agreed","agree","done","will do","thanks","thank you","cool"
    },
    "determination": {
        "we will","we'll","let's","lets","i will","i'll","i can","i'll take","i will take","i'll take"
    },
    "joy": {
        "great","awesome","nice","good","perfect","excellent","fantastic","love","happy","pleased"
    }
}

URGENCY_KEYWORDS = {
    "urgent","immediately","asap","now","right now","today","tonight","deadline","due","must",
    "blocker","blocking","blockers","critical","critical bug","sev1","sev-1","sev0","sev-0",
    "p0","p1","p2","priority 0","priority 1","hotfix","hot fix","incident","outage","deploy now",
    "rollback","production","prod issue","regression","stop the release","stop release","ship hold"
}

VIOLENCE_KEYWORDS = {"kill","murder","beat","stab","shoot","harm"}

def phrase_to_regex_phrase(p):
    esc = re.escape(p)
    esc = esc.replace("\\\'", "'").replace("\\-", "-")
    if re.search(r"\s", p):
        return r"\b" + esc + r"\b"
    else:
        return r"\b" + esc + r"\b"

EMOTION_REGEX = {emo: [re.compile(phrase_to_regex_phrase(p), flags=re.IGNORECASE) for p in sorted(phrases, key=lambda x: -len(x))] for emo,phrases in EMOTION_KEYWORDS.items()}
URGENCY_REGEX = [re.compile(phrase_to_regex_phrase(p), flags=re.IGNORECASE) for p in sorted(URGENCY_KEYWORDS, key=lambda x: -len(x))]
VIOLENCE_REGEX = [re.compile(phrase_to_regex_phrase(p), flags=re.IGNORECASE) for p in sorted(VIOLENCE_KEYWORDS, key=lambda x: -len(x))]

def contains_any_compiled(text: str, compiled_patterns):
    if not text:
        return False
    t = text
    for pat in compiled_patterns:
        if pat.search(t):
            return True
    return False

def find_emotions_in_text(text: str):
    found = []
    for emo, patterns in EMOTION_REGEX.items():
        for pat in patterns:
            if pat.search(text):
                found.append(emo)
                break
    return found

def has_urgency_in_text(text: str):
    return contains_any_compiled(text, URGENCY_REGEX)

def has_violence_in_text(text: str):
    return contains_any_compiled(text, VIOLENCE_REGEX)

# ----------------------------
# Per-word lexical sentiment + keyword emotion
# ----------------------------
try:
    _ = prosody_list
except NameError:
    raise RuntimeError("prosody_list not found. Run the transcription/prosody pipeline first.")

for w in prosody_list:
    tb = TextBlob(w.word if isinstance(w.word, str) else "")
    w.lexical_polarity = float(tb.sentiment.polarity)
    w.lexical_subjectivity = float(tb.sentiment.subjectivity)
    w.keyword_emotions = find_emotions_in_text(w.word)
    w.violent_flag = has_violence_in_text(w.word)
    w._lc = (w.word.lower() if isinstance(w.word, str) else "")

# ----------------------------
# 2) Aggregate per-segment emotion + urgency
# ----------------------------
prosody_list_sorted = sorted(prosody_list, key=lambda x: x.start)
segment_annotations = []
for seg in segments:
    seg_start = seg.get("start", 0.0)
    seg_end = seg.get("end", 0.0)
    seg_text = seg.get("text", "").strip()
    # find words inside the segment
    words_in_seg = [w for w in prosody_list_sorted if (w.start >= seg_start and w.start < seg_end)]
    if len(words_in_seg) == 0:
        # fallback: try token match
        seg_tokens = re.findall(r"\w[\w']+", seg_text.lower())
        words_in_seg = [w for w in prosody_list_sorted if w._lc in seg_tokens]
    if words_in_seg:
        avg_polarity = float(np.nanmean([getattr(w, "lexical_polarity", 0.0) for w in words_in_seg]))
        avg_subjectivity = float(np.nanmean([getattr(w, "lexical_subjectivity", 0.0) for w in words_in_seg]))
        emo_counts = Counter(e for w in words_in_seg for e in getattr(w, "keyword_emotions", []))
        dominant_emo = emo_counts.most_common(1)[0][0] if emo_counts else None
        violent_present = any(getattr(w, "violent_flag", False) for w in words_in_seg)
        combined_vals = [w.score.get("combined", 0.0) for w in words_in_seg]
        avg_combined = float(np.mean(combined_vals)) if combined_vals else 0.0
        has_urgency_word = has_urgency_in_text(seg_text) or any(has_urgency_in_text(w.word) for w in words_in_seg)
        segment_annotations.append({
            "start": seg_start,
            "end": seg_end,
            "text": seg_text,
            "words": words_in_seg,
            "avg_polarity": avg_polarity,
            "avg_subjectivity": avg_subjectivity,
            "dominant_emo": dominant_emo,
            "violent": violent_present,
            "avg_combined": avg_combined,
            "has_urgency_word": has_urgency_word
        })
    else:
        segment_annotations.append({
            "start": seg_start,
            "end": seg_end,
            "text": seg_text,
            "words": [],
            "avg_polarity": 0.0,
            "avg_subjectivity": 0.0,
            "dominant_emo": None,
            "violent": False,
            "avg_combined": 0.0,
            "has_urgency_word": has_urgency_in_text(seg_text)
        })

# Normalize avg_combined across segments
combined_vals_all = np.array([s["avg_combined"] for s in segment_annotations])
if len(combined_vals_all) > 0:
    mu_c, sd_c = float(np.nanmean(combined_vals_all)), float(np.nanstd(combined_vals_all) + 1e-12)
else:
    mu_c, sd_c = 0.0, 1.0

for s in segment_annotations:
    z_comb = (s["avg_combined"] - mu_c) / sd_c
    # urgency raw score: prosody z (0.6) + lexical urgency presence (0.4)
    raw_urgency = 0.6 * max(0.0, z_comb) + 0.4 * (1.0 if s["has_urgency_word"] else 0.0)
    if s["violent"]:
        raw_urgency += 1.0
    if raw_urgency >= 1.0:
        urgency_label = "High"
    elif raw_urgency >= 0.25:
        urgency_label = "Medium"
    else:
        urgency_label = "Low"
    s["z_combined"] = float(z_comb)
    s["raw_urgency"] = float(raw_urgency)
    s["urgency_label"] = urgency_label

# ----------------------------
# 3) Annotate words with segment-derived labels and per-word urgency
# ----------------------------
for w in prosody_list_sorted:
    seg_hit = next((s for s in segment_annotations if w.start >= s["start"] and w.start < s["end"]), None)
    if seg_hit:
        w.segment_emotion = seg_hit["dominant_emo"]
        w.segment_urgency = seg_hit["urgency_label"]
        w.segment_raw_urgency = seg_hit["raw_urgency"]
    else:
        w.segment_emotion = None
        w.segment_urgency = "Low"
        w.segment_raw_urgency = 0.0
    word_combined = w.score.get("combined", 0.0)
    w.z_combined = (word_combined - mu_c) / (sd_c + 1e-12)
    w.word_urgency_score = float(0.6 * max(0.0, w.z_combined) + 0.4 * (1.0 if has_urgency_in_text(w.word) else 0.0))
    w.word_violence_flag = bool(has_violence_in_text(w.word))

# ----------------------------
# 4) Annotate the extractive summary sentences
# ----------------------------
summary_text = summary if isinstance(summary, str) else str(summary)
summary_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', summary_text) if s.strip()]
annotated_summary_sentences = []
for sent in summary_sentences:
    sent_lower = sent.lower()
    matching_segs = [s for s in segment_annotations if any(tok in s["text"].lower() for tok in sent_lower.split()[:6])]
    if not matching_segs:
        matching_segs = [s for s in segment_annotations if any(w.emphasis and w.word.lower() in sent_lower for w in s["words"])]
    if matching_segs:
        urg_vals = [1.0 if x["urgency_label"]=="High" else (0.5 if x["urgency_label"]=="Medium" else 0.0) for x in matching_segs]
        max_urg = max(urg_vals)
        if max_urg >= 1.0:
            sent_urgency = "High"
        elif max_urg >= 0.5:
            sent_urgency = "Medium"
        else:
            sent_urgency = "Low"
        emos = [x["dominant_emo"] for x in matching_segs if x["dominant_emo"]]
        dom_emo = Counter(emos).most_common(1)[0][0] if emos else None
    else:
        sent_urgency = "Low"
        dom_emo = None
    annotated_summary_sentences.append({
        "sentence": sent,
        "urgency": sent_urgency,
        "emotion": dom_emo
    })

# ----------------------------
# 5) Save enriched JSON + VTT + SRT (with tags)
# ----------------------------
enriched = {
    "audio": audio_path,
    "summary": summary,
    "annotated_summary_sentences": annotated_summary_sentences,
    "segments": [
        {
            "start": s["start"], "end": s["end"], "text": s["text"],
            "urgency": s["urgency_label"], "dominant_emotion": s["dominant_emo"], "violent": s["violent"],
            "avg_polarity": s["avg_polarity"], "avg_subjectivity": s["avg_subjectivity"]
        } for s in segment_annotations
    ],
    "words": [
        {
            "word": w.word, "start": w.start, "end": w.end, "emphasis": w.emphasis,
            "score": w.score, "lexical_polarity": getattr(w,"lexical_polarity",None),
            "segment_urgency": getattr(w,"segment_urgency",None),
            "word_urgency_score": getattr(w,"word_urgency_score",0.0),
            "segment_emotion": getattr(w,"segment_emotion",None),
            "violent": getattr(w,"word_violence_flag",False)
        } for w in prosody_list_sorted
    ]
}
# Commented out: disable writing enriched JSON output
# with open("prosody_enriched.json","w",encoding="utf-8") as fh:
#     json.dump(enriched, fh, indent=2, ensure_ascii=False)
# print("Saved prosody_enriched.json")

def write_enriched_vtt(segment_annotations, out_vtt="prosody_enriched.vtt"):
    def fmt(t):
        mm = int(t // 60)
        ss = int(t % 60)
        ms = int((t - int(t)) * 1000)
        return f"{mm:02d}:{ss:02d}.{ms:03d}"
    with open(out_vtt, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for i, s in enumerate(segment_annotations):
            tag = f"[URG:{s['urgency_label']}]"
            emo = f"[EMO:{s['dominant_emo']}]" if s['dominant_emo'] else ""
            text = f"{tag}{emo} {s['text']}"
            f.write(f"{i+1}\n")
            f.write(f"{fmt(s['start'])} --> {fmt(s['end'])}\n")
            f.write(text + "\n\n")
    print("Saved", out_vtt)

def write_enriched_srt(segment_annotations, out_srt="prosody_enriched.srt"):
    def fmt_srt(t):
        h = int(t//3600); m = int((t%3600)//60); s = int(t%60); ms = int((t - int(t))*1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
    entries = []
    idx = 1
    for s in segment_annotations:
        tag = f"[URG:{s['urgency_label']}]"
        emo = f"[EMO:{s['dominant_emo']}]" if s['dominant_emo'] else ""
        text = f"{tag}{emo} {s['text']}"
        entries.append(f"{idx}\n{fmt_srt(s['start'])} --> {fmt_srt(s['end'])}\n{text}\n\n")
        idx += 1
    with open(out_srt, "w", encoding="utf-8") as fh:
        fh.writelines(entries)
    print("Saved", out_srt)

# Run enriched writers (disabled non-.txt outputs)
# write_enriched_vtt(segment_annotations, out_vtt="prosody_enriched.vtt")
# write_enriched_srt(segment_annotations, out_srt="prosody_enriched.srt")

# Build the Prosody_annotations.txt in requested multi-line format and save
segment_lines = ["=== Segment annotations ===\n"]
for s in segment_annotations:
    txt_block = (
        f"[{s['start']:.2f}-{s['end']:.2f}] text={s['text'][:120]} \n"
        f" urg={s['urgency_label']} \n"
        f" emo={s['dominant_emo']} \n "
    )
    segment_lines.append(txt_block)

summary_lines = []
summary_lines.append("\n=== Annotated summary sentences ===\n")
for a in annotated_summary_sentences:
    urg = a.get("urgency", "Low")
    emo = a.get("emotion") if a.get("emotion") is not None else ""
    sent = a.get("sentence", "").strip()
    summary_lines.append(f"({urg}) [{emo}] {sent}")

content = "\n".join(segment_lines) + "\n" + "\n".join(summary_lines) + "\n"
with open("Prosody_annotations.txt", "w", encoding="utf-8") as f:
    f.write(content)

print("Saved Prosody_annotations.txt")
print("All outputs:")
print(" - prosody_results.json")
print(" - prosody.vtt")
print(" - prosody.srt")
print(" - prosody_enriched.json")
print(" - prosody_enriched.vtt")
print(" - prosody_enriched.srt")
print(" - Prosody_annotations.txt")
