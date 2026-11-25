import os
from groq import Groq
from dotenv import load_dotenv

from paths import recordings_dir

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

FOLDER_PATH = str(recordings_dir())

def find_transcript_files(folder):
    txt_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".txt")]
    return txt_files

def read_files(files):
    content = ""
    for f in files:
        with open(f, "r", encoding="utf-8") as file:
            content += "\n\n" + file.read()
    return content.strip()

def summarize(transcript):
    prompt = f"""
You are an AI meeting assistant.

INPUT TRANSCRIPT:
{transcript}

TASKS:
1. Provide a meeting summary. Scale its length to the transcript: a short
   transcript gets a short summary. Use only what is actually said in the
   transcript; never invent participants, agreements, decisions or context.
2. Provide a bullet-style To-Do list with owners, only for tasks that were
   actually stated. If none were stated, write "No action items were mentioned."
3. Always write full sentences, not single words.
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    files = find_transcript_files(FOLDER_PATH)

    if not files:
        print("⚠ No .txt transcript files found in folder.")
        exit()

    transcript_text = read_files(files)
    output = summarize(transcript_text)

    print("\n===== SUMMARY & ACTION ITEMS =====\n")
    print(output)
