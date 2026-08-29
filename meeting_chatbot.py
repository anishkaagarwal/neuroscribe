"""
Shared MeetingChatbot used by every NeuroscribeAI front-end
(app.py, bot4.py CLI, bot_UI.py Flask, the Tkinter / CustomTkinter GUIs).

Loads meeting transcript files from a directory, prioritising prosody-enhanced
versions, and answers questions about them via the Groq LLM API.
"""

import os
from pathlib import Path

from groq import Groq
from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = "llama-3.3-70b-versatile"
TRANSCRIPT_EXTENSIONS = (".txt", ".md", ".log")
_PROSODY_MARKERS = ("_with_prosody", "_prosody", "_complete_analysis")


class MeetingChatbot:
    def __init__(self, api_key=None, transcripts_dir=None, model=DEFAULT_MODEL):
        api_key = api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "A Groq API key is required. Set GROQ_API_KEY in your .env file."
            )

        if transcripts_dir is None:
            from paths import recordings_dir

            transcripts_dir = recordings_dir()

        self.client = Groq(api_key=api_key)
        self.model = model
        self.transcripts_dir = Path(transcripts_dir)
        self.transcripts = self._load_transcripts()
        self.conversation_history = []

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def _load_transcripts(self):
        if not self.transcripts_dir.exists():
            raise ValueError(f"Directory not found: {self.transcripts_dir}")

        # Group related files by base name so we can prefer the richest version.
        groups = {}
        for path in self.transcripts_dir.iterdir():
            if not (path.is_file() and path.suffix.lower() in TRANSCRIPT_EXTENSIONS):
                continue
            base = path.stem
            for marker in (*_PROSODY_MARKERS, "_audio", "_video", "_final",
                           "_diarized", "_intents"):
                base = base.replace(marker, "")
            groups.setdefault(base, []).append(path)

        def priority(p: Path) -> int:
            name = p.name.lower()
            if "_complete_analysis" in name:
                return 0
            if "_with_prosody" in name:
                return 1
            if "_prosody" in name:
                return 2
            return 3

        transcripts = []
        for base, files in groups.items():
            selected = sorted(files, key=priority)[0]
            try:
                content = selected.read_text(encoding="utf-8")
            except Exception as exc:  # pragma: no cover - I/O edge case
                print(f"Error loading {selected.name}: {exc}")
                continue
            has_prosody = priority(selected) <= 2
            transcripts.append(
                {
                    "filename": selected.name,
                    "base_name": base,
                    "content": content,
                    "has_prosody": has_prosody,
                }
            )
            print(f"Loaded: {selected.name}{' [WITH PROSODY]' if has_prosody else ''}")

        if not transcripts:
            raise ValueError(
                f"No transcript files ({', '.join(TRANSCRIPT_EXTENSIONS)}) "
                f"found in {self.transcripts_dir}"
            )

        prosody_count = sum(1 for t in transcripts if t["has_prosody"])
        print(
            f"\n[chatbot] {len(transcripts)} transcript(s) loaded "
            f"({prosody_count} with prosody analysis)"
        )
        return transcripts

    # ------------------------------------------------------------------
    # Prompting
    # ------------------------------------------------------------------
    def _create_context(self):
        context = (
            "Meeting transcripts. Some include prosody analysis with urgency "
            "levels (urg=High/Medium/Low), emotion tags (emo=...), and segment "
            "time ranges [start-end].\n\n"
        )
        for i, t in enumerate(self.transcripts, 1):
            context += f"=== File {i}: {t['filename']} ===\n"
            if t["has_prosody"]:
                context += "[includes prosody analysis]\n"
            context += t["content"] + "\n\n"
        return context

    def ask(self, question):
        system_message = (
            "You are a helpful assistant that answers questions about meeting "
            "transcripts. When prosody markers are present, use urgency and "
            "emotion information to give context-aware answers and to "
            "distinguish urgent items from routine discussion. Answer with full "
            "sentences. If the answer is not in the transcripts, say so.\n\n"
            + self._create_context()
        )

        self.conversation_history.append({"role": "user", "content": question})
        messages = [{"role": "system", "content": system_message}] + self.conversation_history

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=1024,
            )
            answer = response.choices[0].message.content
            self.conversation_history.append({"role": "assistant", "content": answer})
            return answer
        except Exception as exc:
            return f"Error: {exc}"

    def reset_conversation(self):
        self.conversation_history = []
        print("Conversation history cleared.")

    # ------------------------------------------------------------------
    # Terminal helper
    # ------------------------------------------------------------------
    def run_interactive_chat(self):
        print(f"\nLoaded {len(self.transcripts)} transcript file(s).")
        print("Ask questions about the meeting ('quit' to exit, 'reset' to clear).\n")
        while True:
            try:
                question = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break
            if question.lower() in {"quit", "exit", "q"}:
                print("Goodbye!")
                break
            if question.lower() == "reset":
                self.reset_conversation()
                continue
            if not question:
                continue
            print("\nAssistant:", self.ask(question), "\n")


def main():
    transcripts_dir = os.getenv("NEUROSCRIBE_RECORDINGS_DIR") or input(
        "Path to your transcripts directory (blank = ./recordings): "
    ).strip() or None
    try:
        MeetingChatbot(transcripts_dir=transcripts_dir).run_interactive_chat()
    except Exception as exc:
        print(f"Error: {exc}")


if __name__ == "__main__":
    main()
