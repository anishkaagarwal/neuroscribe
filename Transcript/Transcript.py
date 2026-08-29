import os
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Folder that contains files to transcribe (override with NEUROSCRIBE_RECORDINGS_DIR).
# NOTE: this is a legacy duplicate of ../Transcript.py; prefer importing that module.
FOLDER_PATH = os.getenv(
    "NEUROSCRIBE_RECORDINGS_DIR",
    str(Path(__file__).resolve().parents[1] / "recordings"),
)

def transcribe_file(file_path):
    """Send a file to WhisperX model on Groq API and return transcript."""
    with open(file_path, "rb") as f:
        transcript = client.audio.transcriptions.create(
            file=f,
            model="whisper-large-v3",
            response_format="verbose_json"
        )

    # transcript.text contains final text
    return transcript.text

def process_folder(folder):
    supported_ext = (".wav", ".mp3", ".mp4", ".avi", ".m4a", ".mov")

    for file in os.listdir(folder):
        if file.lower().endswith(supported_ext):
            full_path = os.path.join(folder, file)
            print(f"Transcribing → {file}")

            text = transcribe_file(full_path)

            # Save transcript in same folder as text file
            text_file_path = os.path.join(folder, f"{os.path.splitext(file)[0]}.txt")

            with open(text_file_path, "w", encoding="utf-8") as out:
                out.write(text)

            print(f"✅ Saved transcript as: {text_file_path}")


if __name__ == "__main__":
    process_folder(FOLDER_PATH)
