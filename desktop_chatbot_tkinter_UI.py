"""
Modern Desktop Chatbot with CustomTkinter
Install: pip install customtkinter
"""

import os
import customtkinter as ctk
from dotenv import load_dotenv
import threading

from meeting_chatbot import MeetingChatbot

load_dotenv()


class ModernChatbotGUI:
    def __init__(self):
        # Set appearance
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # Create window
        self.root = ctk.CTk()
        self.root.title("Meeting Chatbot")
        self.root.geometry("700x800")
        
        # Initialize chatbot (transcripts dir: NEUROSCRIBE_RECORDINGS_DIR or ./recordings)
        transcript_folder = os.getenv("NEUROSCRIBE_RECORDINGS_DIR")

        try:
            self.chatbot = MeetingChatbot(transcripts_dir=transcript_folder)
            self.create_widgets()
        except Exception as e:
            self.show_error(f"Failed to initialize: {e}")
            
    def create_widgets(self):
        # Header
        header = ctk.CTkFrame(self.root, height=70, corner_radius=0)
        header.pack(fill="x")
        header.pack_propagate(False)
        
        title = ctk.CTkLabel(
            header,
            text="🤖 Meeting Chatbot",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title.pack(pady=20)
        
        # Chat area
        chat_frame = ctk.CTkFrame(self.root, fg_color="transparent")
        chat_frame.pack(fill="both", expand=True, padx=15, pady=15)
        
        self.chat_display = ctk.CTkTextbox(
            chat_frame,
            font=ctk.CTkFont(size=13),
            wrap="word",
            corner_radius=10
        )
        self.chat_display.pack(fill="both", expand=True)
        
        # Input area
        input_frame = ctk.CTkFrame(self.root, fg_color="transparent")
        input_frame.pack(fill="x", padx=15, pady=(0, 15))
        
        # Buttons frame
        btn_frame = ctk.CTkFrame(input_frame, fg_color="transparent")
        btn_frame.pack(fill="x", pady=(0, 10))
        
        reset_btn = ctk.CTkButton(
            btn_frame,
            text="🔄 Reset",
            command=self.reset_conversation,
            fg_color="#dc3545",
            hover_color="#c82333",
            width=100,
            height=35
        )
        reset_btn.pack(side="right")
        
        files_label = ctk.CTkLabel(
            btn_frame,
            text=f"📁 {len(self.chatbot.transcripts)} files loaded",
            font=ctk.CTkFont(size=11)
        )
        files_label.pack(side="left")
        
        # Message input
        message_frame = ctk.CTkFrame(input_frame, fg_color="transparent")
        message_frame.pack(fill="x")
        
        self.message_input = ctk.CTkEntry(
            message_frame,
            placeholder_text="Ask about your meetings...",
            font=ctk.CTkFont(size=13),
            height=45
        )
        self.message_input.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self.message_input.bind('<Return>', lambda e: self.send_message())
        
        send_btn = ctk.CTkButton(
            message_frame,
            text="Send",
            command=self.send_message,
            width=100,
            height=45,
            font=ctk.CTkFont(size=14, weight="bold")
        )
        send_btn.pack(side="right")
        
        # Welcome message
        self.add_message("👋 Welcome! Ask me anything about your meetings.", "bot")
        
    def add_message(self, text, sender):
        self.chat_display.configure(state="normal")
        
        if sender == "user":
            self.chat_display.insert("end", "\n🧑 You:\n", "user_label")
            self.chat_display.insert("end", f"{text}\n", "user_msg")
        else:
            self.chat_display.insert("end", "\n🤖 Bot:\n", "bot_label")
            self.chat_display.insert("end", f"{text}\n", "bot_msg")
        
        self.chat_display.insert("end", "─" * 60 + "\n", "separator")
        
        # Styling
        self.chat_display.tag_config("user_label", foreground="#4dabf7", font=ctk.CTkFont(size=13, weight="bold"))
        self.chat_display.tag_config("user_msg", foreground="#e0e0e0")
        self.chat_display.tag_config("bot_label", foreground="#51cf66", font=ctk.CTkFont(size=13, weight="bold"))
        self.chat_display.tag_config("bot_msg", foreground="#e0e0e0")
        self.chat_display.tag_config("separator", foreground="#404040")
        
        self.chat_display.configure(state="disabled")
        self.chat_display.see("end")
        
    def send_message(self):
        message = self.message_input.get().strip()
        if not message:
            return
        
        self.message_input.delete(0, "end")
        self.add_message(message, "user")
        
        # Show thinking
        self.chat_display.configure(state="normal")
        self.chat_display.insert("end", "\n🤖 Bot:\n", "bot_label")
        self.chat_display.insert("end", "⏳ Thinking...\n", "thinking")
        self.chat_display.tag_config("thinking", foreground="#ffd43b")
        self.chat_display.configure(state="disabled")
        self.chat_display.see("end")
        
        def get_response():
            response = self.chatbot.ask(message)
            
            # Remove thinking message
            self.chat_display.configure(state="normal")
            self.chat_display.delete("end-3l", "end-1l")
            self.chat_display.configure(state="disabled")
            
            self.add_message(response, "bot")
        
        thread = threading.Thread(target=get_response, daemon=True)
        thread.start()
        
    def reset_conversation(self):
        self.chatbot.conversation_history = []
        self.chat_display.configure(state="normal")
        self.chat_display.delete("1.0", "end")
        self.chat_display.configure(state="disabled")
        self.add_message("Conversation reset! Start fresh.", "bot")
        
    def show_error(self, message):
        error_window = ctk.CTkToplevel(self.root)
        error_window.title("Error")
        error_window.geometry("400x200")
        
        label = ctk.CTkLabel(error_window, text=message, wraplength=350)
        label.pack(pady=40)
        
        btn = ctk.CTkButton(error_window, text="OK", command=error_window.destroy)
        btn.pack(pady=20)
        
    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = ModernChatbotGUI()
    app.run()