import os
from pydub import AudioSegment
from groq import Groq
from dotenv import load_dotenv

from paths import recordings_dir

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

FOLDER_PATH = str(recordings_dir())
MAX_FILE_SIZE_MB = 20  # Safe limit below 25MB

def get_file_size_mb(file_path):
    return os.path.getsize(file_path) / (1024 * 1024)

def split_audio_file(file_path, chunk_duration_ms=120000):  # 10 minutes per chunk
    """Split large audio into smaller chunks"""
    audio = AudioSegment.from_file(file_path)
    chunks = []
    
    for i, start_ms in enumerate(range(0, len(audio), chunk_duration_ms)):
        chunk = audio[start_ms:start_ms + chunk_duration_ms]
        chunk_path = f"{file_path}_chunk_{i}.wav"
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)
        print(f"  Created chunk {i+1}: {chunk_path}")
    
    return chunks

def transcribe_file(file_path):
    """Transcribe a single file with size check"""
    file_size_mb = get_file_size_mb(file_path)
    
    if file_size_mb > MAX_FILE_SIZE_MB:
        print(f"⚠️ File too large ({file_size_mb:.2f} MB). Splitting...")
        chunks = split_audio_file(file_path)
        
        full_transcript = ""
        for i, chunk_path in enumerate(chunks):
            print(f"Transcribing chunk {i+1}/{len(chunks)}...")
            with open(chunk_path, "rb") as f:
                response = client.audio.transcriptions.create(
                    file=f,
                    model="whisper-large-v3",
                    response_format="verbose_json"
                )
                full_transcript += response.text + "\n"
            
            # Clean up chunk file
            os.remove(chunk_path)
        
        return full_transcript
    
    else:
        # File is small enough, transcribe normally
        with open(file_path, "rb") as f:
            response = client.audio.transcriptions.create(
                file=f,
                model="whisper-large-v3",
                response_format="verbose_json"
            )
            return response.text

def process_folder(folder):
    supported_ext = (".wav", ".mp3", ".mp4", ".avi", ".m4a", ".mov")

    for file in os.listdir(folder):
        if file.lower().endswith(supported_ext):
            full_path = os.path.join(folder, file)
            file_size = get_file_size_mb(full_path)
            
            print(f"\n Processing: {file} ({file_size:.2f} MB)")

            try:
                text = transcribe_file(full_path)
                
                # Save transcript
                text_file_path = os.path.join(folder, f"{os.path.splitext(file)[0]}.txt")
                with open(text_file_path, "w", encoding="utf-8") as out:
                    out.write(text)

                print(f" Saved transcript: {text_file_path}")
                
            except Exception as e:
                print(f" Error processing {file}: {e}")

if __name__ == "__main__":
    process_folder(FOLDER_PATH)