import os
import tkinter as tk
from tkinter import scrolledtext, messagebox
import threading

from dotenv import load_dotenv

from meeting_chatbot import MeetingChatbot

# Load environment variables
load_dotenv()


class ChatbotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🤖 Meeting Chatbot")
        self.root.geometry("600x700")
        self.root.configure(bg='#f5f5f5')

        # Initialize chatbot (transcripts dir: NEUROSCRIBE_RECORDINGS_DIR or ./recordings)
        transcript_folder = os.getenv("NEUROSCRIBE_RECORDINGS_DIR")

        try:
            self.chatbot = MeetingChatbot(transcripts_dir=transcript_folder)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize chatbot: {e}")
            self.root.destroy()
            return
        
        # Create GUI elements
        self.create_widgets()
        
    def create_widgets(self):
        # Title
        title_frame = tk.Frame(self.root, bg='#0078ff', height=60)
        title_frame.pack(fill=tk.X)
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(
            title_frame, 
            text="🤖 Meeting Chatbot with Prosody", 
            bg='#0078ff', 
            fg='white',
            font=('Arial', 16, 'bold')
        )
        title_label.pack(pady=15)
        
        # Chat display area
        chat_frame = tk.Frame(self.root, bg='#f5f5f5')
        chat_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.chat_display = scrolledtext.ScrolledText(
            chat_frame,
            wrap=tk.WORD,
            font=('Arial', 10),
            bg='#ffffff',
            state=tk.DISABLED,
            relief=tk.FLAT,
            borderwidth=2
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True)
        
        # Configure tags for styling
        self.chat_display.tag_config('user', foreground='#0078ff', font=('Arial', 10, 'bold'))
        self.chat_display.tag_config('bot', foreground='#333333', font=('Arial', 10))
        self.chat_display.tag_config('separator', foreground='#cccccc')
        self.chat_display.tag_config('info', foreground='#666666', font=('Arial', 9, 'italic'))
        
        # Input area
        input_frame = tk.Frame(self.root, bg='#f5f5f5')
        input_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        # Reset button
        reset_btn = tk.Button(
            input_frame,
            text="🔄 Reset",
            command=self.reset_conversation,
            bg='#ff4444',
            fg='white',
            font=('Arial', 9),
            relief=tk.FLAT,
            cursor='hand2',
            padx=10,
            pady=5
        )
        reset_btn.pack(side=tk.RIGHT, padx=(5, 0))
        
        # Send button
        send_btn = tk.Button(
            input_frame,
            text="Send",
            command=self.send_message,
            bg='#0078ff',
            fg='white',
            font=('Arial', 10, 'bold'),
            relief=tk.FLAT,
            cursor='hand2',
            padx=20,
            pady=5
        )
        send_btn.pack(side=tk.RIGHT)
        
        # Input field
        self.message_input = tk.Entry(
            input_frame,
            font=('Arial', 11),
            relief=tk.FLAT,
            borderwidth=2
        )
        self.message_input.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=8)
        self.message_input.bind('<Return>', lambda e: self.send_message())
        self.message_input.focus()
        
        # Welcome message with prosody info
        prosody_count = sum(1 for t in self.chatbot.transcripts if t.get('has_prosody', False))
        welcome_msg = f"Welcome! I have access to {len(self.chatbot.transcripts)} meeting transcript(s)."
        if prosody_count > 0:
            welcome_msg += f"\n{prosody_count} transcript(s) include prosody analysis with urgency and emotion markers."
            welcome_msg += "\n\nYou can ask about:"
            welcome_msg += "\n• What were the most urgent topics?"
            welcome_msg += "\n• What emotions were expressed?"
            welcome_msg += "\n• Which parts of the meeting were high priority?"
            welcome_msg += "\n• Any specific questions about the meeting content"
        
        self.display_message(welcome_msg, "bot")
        
    def display_message(self, message, sender):
        self.chat_display.config(state=tk.NORMAL)
        
        if sender == "user":
            self.chat_display.insert(tk.END, "You: ", 'user')
            self.chat_display.insert(tk.END, f"{message}\n")
        else:
            self.chat_display.insert(tk.END, "Bot: ", 'bot')
            self.chat_display.insert(tk.END, f"{message}\n")
        
        self.chat_display.insert(tk.END, "-" * 80 + "\n", 'separator')
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)
        
    def send_message(self):
        message = self.message_input.get().strip()
        if not message:
            return
        
        # Clear input
        self.message_input.delete(0, tk.END)
        
        # Display user message
        self.display_message(message, "user")
        
        # Show "thinking" indicator
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, "Bot is thinking...\n", 'bot')
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)
        
        # Get response in thread to prevent GUI freezing
        def get_response():
            response = self.chatbot.ask(message)
            
            # Remove "thinking" message
            self.chat_display.config(state=tk.NORMAL)
            self.chat_display.delete("end-2l", "end-1l")
            self.chat_display.config(state=tk.DISABLED)
            
            # Display bot response
            self.display_message(response, "bot")
        
        thread = threading.Thread(target=get_response, daemon=True)
        thread.start()
        
    def reset_conversation(self):
        if messagebox.askyesno("Reset", "Clear conversation history?"):
            self.chatbot.conversation_history = []
            self.chat_display.config(state=tk.NORMAL)
            self.chat_display.delete(1.0, tk.END)
            self.chat_display.config(state=tk.DISABLED)
            
            # Re-display welcome message
            prosody_count = sum(1 for t in self.chatbot.transcripts if t.get('has_prosody', False))
            welcome_msg = f"Conversation reset! I have access to {len(self.chatbot.transcripts)} meeting transcript(s)."
            if prosody_count > 0:
                welcome_msg += f"\n{prosody_count} include prosody analysis."
            self.display_message(welcome_msg, "bot")


def main():
    root = tk.Tk()
    app = ChatbotGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()