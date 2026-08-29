"""
Flask web UI for the NeuroscribeAI meeting chatbot.

Uses the shared MeetingChatbot from meeting_chatbot.py.
Run:  python bot_UI.py   (set NEUROSCRIBE_RECORDINGS_DIR to point at your transcripts)
"""

import os

from flask import Flask, render_template, request

from meeting_chatbot import MeetingChatbot

app = Flask(__name__)

# Transcripts location: NEUROSCRIBE_RECORDINGS_DIR env var, else ./recordings
chatbot = MeetingChatbot(transcripts_dir=os.getenv("NEUROSCRIBE_RECORDINGS_DIR"))


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_message = request.form["message"]
        bot_response = chatbot.ask(user_message)
        return render_template(
            "index.html", user_message=user_message, bot_response=bot_response
        )
    return render_template("index.html")


@app.route("/reset", methods=["POST"])
def reset():
    chatbot.reset_conversation()
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=bool(os.getenv("FLASK_DEBUG")))
