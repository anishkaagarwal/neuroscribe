"""
NeuroscribeAI - Complete Meeting Recording and Analysis Application
Main entry point with Deepgram Speaker Diarization and Intent Detection
"""

import os
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
from pathlib import Path
from datetime import datetime
import threading
import numpy as np
from groq import Groq
from dotenv import load_dotenv
import subprocess
import sys
import json
import asyncio
from secure_lfv import SecureFeatureVault

# Import your custom modules
try:
    from meet3 import MeetingRecorder
    from Transcript import transcribe_file
    from summarize_transcript import summarize
    from meeting_chatbot import MeetingChatbot
except ImportError as e:
    print(f"Warning: Some modules couldn't be imported: {e}")
    print("Make sure all project files are in the same directory")

load_dotenv()


class DeepgramProcessor:
    """Handler for Deepgram speaker diarization and intent detection"""
    
    def __init__(self, api_key):
        try:
            from deepgram import Deepgram
            self.Deepgram = Deepgram
            self.api_key = api_key
        except ImportError:
            raise ImportError("deepgram-sdk not installed. Run: pip install deepgram-sdk==2.12.0")
    
    async def process_audio_async(self, audio_path):
        """Process audio with Deepgram for diarization and intents"""
        deepgram = self.Deepgram(self.api_key)
        
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
        
        # Determine mimetype
        suffix = Path(audio_path).suffix.lower()
        mimetype_map = {
            '.mp3': 'audio/mp3',
            '.mp4': 'audio/mp4',
            '.wav': 'audio/wav',
            '.m4a': 'audio/m4a',
            '.avi': 'audio/avi',
            '.mov': 'audio/mov'
        }
        mimetype = mimetype_map.get(suffix, 'audio/mp4')
        
        options = {
            "model": "nova-3",
            "language": "en",
            "intents": True,
            "diarize": True,
            "punctuate": True
        }
        
        response = await deepgram.transcription.prerecorded(
            {"buffer": audio_bytes, "mimetype": mimetype},
            options
        )
        
        return response
    
    def process_audio(self, audio_path):
        """Synchronous wrapper for async processing"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(self.process_audio_async(audio_path))
            return result
        finally:
            loop.close()
    
    def create_diarized_transcript(self, response_json):
        """Create speaker-diarized transcript from Deepgram response"""
        lines = []
        try:
            words = response_json["results"]["channels"][0]["alternatives"][0]["words"]
            curr_speaker = None
            curr_line = ''
            
            for word_struct in words:
                word_speaker = word_struct.get("speaker", 0)
                word = word_struct.get("punctuated_word", word_struct.get("word", ""))
                
                if word_speaker == curr_speaker:
                    curr_line += ' ' + word
                else:
                    if curr_speaker is not None:
                        tag = f'SPEAKER {curr_speaker}:'
                        full_line = tag + curr_line
                        lines.append(full_line)
                    curr_speaker = word_speaker
                    curr_line = ' ' + word
            
            # Add final line
            if curr_speaker is not None:
                lines.append(f'SPEAKER {curr_speaker}:' + curr_line)
            
            return '\n\n'.join(lines)
        except Exception as e:
            print(f"Error creating diarized transcript: {e}")
            return ""
    
    def extract_intents(self, response_json):
        """Extract intent segments from Deepgram response"""
        def find_segments_with_intents(obj):
            found = []
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k == "segments" and isinstance(v, list):
                        for seg in v:
                            if isinstance(seg, dict) and "text" in seg and "intents" in seg:
                                found.append(seg)
                    else:
                        found.extend(find_segments_with_intents(v))
            elif isinstance(obj, list):
                for item in obj:
                    found.extend(find_segments_with_intents(item))
            return found
        
        segments = find_segments_with_intents(response_json)
        
        # Format intents nicely
        intent_text = "=== INTENT ANALYSIS ===\n\n"
        
        if not segments:
            intent_text += "No intents detected.\n"
        else:
            for i, seg in enumerate(segments, start=1):
                text = seg.get("text", "").strip()
                intent_text += f"--- Segment {i} ---\n"
                intent_text += f"Text: {text}\n\n"
                intent_text += "Intents:\n"
                
                intents = seg.get("intents", [])
                if not intents:
                    intent_text += "  (no intents)\n"
                else:
                    for it in intents:
                        label = it.get("intent") if isinstance(it, dict) else str(it)
                        conf = None
                        if isinstance(it, dict):
                            for k in ("confidence_score", "confidence", "score"):
                                if k in it:
                                    conf = it[k]
                                    break
                        if conf is None:
                            intent_text += f"  - {label}\n"
                        else:
                            try:
                                intent_text += f"  - {label} \n"
                            except Exception:
                                intent_text += f"  - {label} \n"
                intent_text += "\n"
        
        return intent_text, segments


class NeuroscribeAI:
    def __init__(self, root):
        self.root = root
        self.root.title("NeuroscribeAI - Meeting Assistant")
        self.root.geometry("900x700")
        self.root.configure(bg='#1a1a2e')
        
        # State variables
        self.recorder = None
        self.recording_active = False
        self.transcripts_dir = Path("recordings")
        self.transcripts_dir.mkdir(exist_ok=True)
        self.current_files = []
        
        # API keys
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.deepgram_api_key = os.getenv("DEEPGRAM_API_KEY")
        
        if not self.groq_api_key:
            messagebox.showerror("Error", "GROQ_API_KEY not found in .env file!")
            self.root.destroy()
            return
        
        if not self.deepgram_api_key:
            messagebox.showwarning("Warning", "DEEPGRAM_API_KEY not found. Speaker diarization will be disabled.")
            self.deepgram_processor = None
        else:
            try:
                self.deepgram_processor = DeepgramProcessor(self.deepgram_api_key)
            except ImportError as e:
                messagebox.showwarning("Warning", f"Deepgram not available: {e}")
                self.deepgram_processor = None
        
        # Initialize Secure Feature Vault
        self.lfv = SecureFeatureVault(
            vault_dir="secure_vault",
            user_password=os.getenv("LFV_PASSWORD")  # Optional: use password from .env
        )
        self.log_message("🔒 Secure Feature Vault initialized")
        
        self.create_ui()
        
    def create_ui(self):
        """Create the main user interface"""
        
        # Header
        header_frame = tk.Frame(self.root, bg='#16213e', height=80)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(
            header_frame,
            text="🧠 NeuroscribeAI",
            font=('Arial', 24, 'bold'),
            bg='#16213e',
            fg='#00d9ff'
        )
        title_label.pack(pady=20)
        
        subtitle = tk.Label(
            header_frame,
            text="AI-Powered Meeting Analysis with Prosody, Diarization & Intent Detection",
            font=('Arial', 10),
            bg='#16213e',
            fg='#a0a0a0'
        )
        subtitle.place(relx=0.5, rely=0.9, anchor='center')
        
        # Main content area with notebook
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Configure ttk styles
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TNotebook', background='#1a1a2e', borderwidth=0)
        style.configure('TNotebook.Tab', background='#0f3460', foreground='white', 
                       padding=[20, 10], font=('Arial', 10, 'bold'))
        style.map('TNotebook.Tab', background=[('selected', '#16213e')])
        
        # Tab 1: Record Meeting
        self.record_tab = tk.Frame(self.notebook, bg='#1a1a2e')
        self.notebook.add(self.record_tab, text='🔹 Record Meeting')
        self.create_record_tab()
        
        # Tab 2: Upload Files
        self.upload_tab = tk.Frame(self.notebook, bg='#1a1a2e')
        self.notebook.add(self.upload_tab, text='📁 Upload Files')
        self.create_upload_tab()
        
        # Tab 3: Results
        self.results_tab = tk.Frame(self.notebook, bg='#1a1a2e')
        self.notebook.add(self.results_tab, text='📊 Results')
        self.create_results_tab()
        
    def create_record_tab(self):
        """Create the meeting recording interface"""
        
        # Container frame
        container = tk.Frame(self.record_tab, bg='#1a1a2e')
        container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Meeting Configuration Section
        config_frame = tk.LabelFrame(
            container,
            text="Meeting Configuration",
            font=('Arial', 12, 'bold'),
            bg='#16213e',
            fg='#00d9ff',
            padx=20,
            pady=20
        )
        config_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Meeting URL
        tk.Label(config_frame, text="Meeting URL:", bg='#16213e', fg='white', 
                font=('Arial', 10)).grid(row=0, column=0, sticky='w', pady=10)
        self.url_entry = tk.Entry(config_frame, width=50, font=('Arial', 10))
        self.url_entry.grid(row=0, column=1, pady=10, padx=10)
        self.url_entry.insert(0, "https://meet.google.com/xxx-xxxx-xxx")
        
        # Platform Selection
        tk.Label(config_frame, text="Platform:", bg='#16213e', fg='white',
                font=('Arial', 10)).grid(row=1, column=0, sticky='w', pady=10)
        platform_frame = tk.Frame(config_frame, bg='#16213e')
        platform_frame.grid(row=1, column=1, sticky='w', pady=10, padx=10)
        
        self.platform_var = tk.StringVar(value="google_meet")
        tk.Radiobutton(platform_frame, text="Google Meet", variable=self.platform_var,
                      value="google_meet", bg='#16213e', fg='white', 
                      selectcolor='#0f3460', font=('Arial', 10)).pack(side=tk.LEFT, padx=10)
        tk.Radiobutton(platform_frame, text="Zoom", variable=self.platform_var,
                      value="zoom", bg='#16213e', fg='white',
                      selectcolor='#0f3460', font=('Arial', 10)).pack(side=tk.LEFT)
        
        # Password (optional)
        tk.Label(config_frame, text="Password (if any):", bg='#16213e', fg='white',
                font=('Arial', 10)).grid(row=2, column=0, sticky='w', pady=10)
        self.password_entry = tk.Entry(config_frame, width=50, font=('Arial', 10), show='*')
        self.password_entry.grid(row=2, column=1, pady=10, padx=10)
        
        # Recorder Name
        tk.Label(config_frame, text="Your Name:", bg='#16213e', fg='white',
                font=('Arial', 10)).grid(row=3, column=0, sticky='w', pady=10)
        self.name_entry = tk.Entry(config_frame, width=50, font=('Arial', 10))
        self.name_entry.grid(row=3, column=1, pady=10, padx=10)
        self.name_entry.insert(0, "Meeting Recorder Bot")
        
        # Duration
        tk.Label(config_frame, text="Duration (seconds):", bg='#16213e', fg='white',
                font=('Arial', 10)).grid(row=4, column=0, sticky='w', pady=10)
        duration_frame = tk.Frame(config_frame, bg='#16213e')
        duration_frame.grid(row=4, column=1, sticky='w', pady=10, padx=10)
        
        self.duration_entry = tk.Entry(duration_frame, width=15, font=('Arial', 10))
        self.duration_entry.pack(side=tk.LEFT)
        self.duration_entry.insert(0, "3600")
        tk.Label(duration_frame, text="(or 0 for manual stop)", bg='#16213e', 
                fg='#a0a0a0', font=('Arial', 9)).pack(side=tk.LEFT, padx=10)
        
        # Control Buttons
        button_frame = tk.Frame(container, bg='#1a1a2e')
        button_frame.pack(pady=20)
        
        self.start_btn = tk.Button(
            button_frame,
            text="▶️ Start Recording",
            command=self.start_recording,
            bg='#e63946',
            fg='white',
            font=('Arial', 12, 'bold'),
            width=20,
            height=2,
            cursor='hand2',
            relief=tk.FLAT
        )
        self.start_btn.pack(side=tk.LEFT, padx=10)
        
        self.stop_btn = tk.Button(
            button_frame,
            text="⏹️ Stop Recording",
            command=self.stop_recording,
            bg='#457b9d',
            fg='white',
            font=('Arial', 12, 'bold'),
            width=20,
            height=2,
            cursor='hand2',
            relief=tk.FLAT,
            state=tk.DISABLED
        )
        self.stop_btn.pack(side=tk.LEFT, padx=10)
        
        # Status Area
        status_frame = tk.LabelFrame(
            container,
            text="Status",
            font=('Arial', 12, 'bold'),
            bg='#16213e',
            fg='#00d9ff',
            padx=20,
            pady=20
        )
        status_frame.pack(fill=tk.BOTH, expand=True)
        
        self.status_text = scrolledtext.ScrolledText(
            status_frame,
            height=10,
            font=('Courier', 9),
            bg='#0f3460',
            fg='#00ff00',
            wrap=tk.WORD
        )
        self.status_text.pack(fill=tk.BOTH, expand=True)
        self.log_message("Ready to record. Configure settings and click Start.")
        
    def create_upload_tab(self):
        """Create the file upload interface"""
        
        container = tk.Frame(self.upload_tab, bg='#1a1a2e')
        container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Instructions
        instructions = tk.Label(
            container,
            text="Upload existing meeting recordings (audio/video files)",
            font=('Arial', 12, 'bold'),
            bg='#1a1a2e',
            fg='#00d9ff'
        )
        instructions.pack(pady=20)
        
        # Upload button
        upload_btn = tk.Button(
            container,
            text="📂 Select Files to Upload",
            command=self.upload_files,
            bg='#16213e',
            fg='white',
            font=('Arial', 12, 'bold'),
            width=30,
            height=3,
            cursor='hand2',
            relief=tk.FLAT
        )
        upload_btn.pack(pady=20)
        
        # File list
        list_frame = tk.LabelFrame(
            container,
            text="Uploaded Files",
            font=('Arial', 11, 'bold'),
            bg='#16213e',
            fg='#00d9ff',
            padx=10,
            pady=10
        )
        list_frame.pack(fill=tk.BOTH, expand=True, pady=20)
        
        self.file_listbox = tk.Listbox(
            list_frame,
            font=('Arial', 10),
            bg='#0f3460',
            fg='white',
            selectbackground='#00d9ff',
            selectforeground='black'
        )
        self.file_listbox.pack(fill=tk.BOTH, expand=True)
        
        # Process button
        process_btn = tk.Button(
            container,
            text="⚙️ Process Files",
            command=self.process_uploaded_files,
            bg='#2a9d8f',
            fg='white',
            font=('Arial', 12, 'bold'),
            width=30,
            height=2,
            cursor='hand2',
            relief=tk.FLAT
        )
        process_btn.pack(pady=10)
        
    def create_results_tab(self):
        """Create the results display interface"""
    
        container = tk.Frame(self.results_tab, bg='#1a1a2e')
        container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
    
        # Transcript Section with Prosody
        transcript_frame = tk.LabelFrame(
            container,
            text="📝 Transcript with Analysis (Prosody, Diarization, Intents)",
            font=('Arial', 12, 'bold'),
            bg='#16213e',
            fg='#00d9ff',
            padx=10,
            pady=10
        )
        transcript_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
    
        self.transcript_text = scrolledtext.ScrolledText(
            transcript_frame,
            height=12,
            font=('Arial', 10),
            bg='#0f3460',
            fg='white',
            wrap=tk.WORD
        )
        self.transcript_text.pack(fill=tk.BOTH, expand=True)
    
        # Summary & To-Do Section
        summary_frame = tk.LabelFrame(
            container,
            text="📋 Summary & To-Do List",
            font=('Arial', 12, 'bold'),
            bg='#16213e',
            fg='#00d9ff',
            padx=10,
            pady=10
        )
        summary_frame.pack(fill=tk.BOTH, expand=True, pady=10)
    
        self.summary_text = scrolledtext.ScrolledText(
            summary_frame,
            height=10,
            font=('Arial', 10),
            bg='#0f3460',
            fg='white',
            wrap=tk.WORD
        )
        self.summary_text.pack(fill=tk.BOTH, expand=True)
    
        # Add Cache Management Section
        cache_frame = tk.LabelFrame(
            container,
            text="🔒 Secure Feature Vault",
            font=('Arial', 12, 'bold'),
            bg='#16213e',
            fg='#00d9ff',
            padx=10,
            pady=10
        )
        cache_frame.pack(fill=tk.X, pady=10)
        
        # Cache statistics display
        self.cache_stats_label = tk.Label(
            cache_frame,
            text="Cache statistics will appear here",
            font=('Arial', 10),
            bg='#16213e',
            fg='white',
            justify=tk.LEFT
        )
        self.cache_stats_label.pack(pady=10)
        
        # Cache control buttons
        cache_btn_frame = tk.Frame(cache_frame, bg='#16213e')
        cache_btn_frame.pack(pady=10)
        
        refresh_cache_btn = tk.Button(
            cache_btn_frame,
            text="🔄 Refresh Stats",
            command=self.refresh_cache_stats,
            bg='#457b9d',
            fg='white',
            font=('Arial', 10, 'bold'),
            width=15,
            cursor='hand2',
            relief=tk.FLAT
        )
        refresh_cache_btn.pack(side=tk.LEFT, padx=5)
        
        view_cache_btn = tk.Button(
            cache_btn_frame,
            text="📋 View Cache",
            command=self.view_cache_entries,
            bg='#6a4c93',
            fg='white',
            font=('Arial', 10, 'bold'),
            width=15,
            cursor='hand2',
            relief=tk.FLAT
        )
        view_cache_btn.pack(side=tk.LEFT, padx=5)
        
        clear_cache_btn = tk.Button(
            cache_btn_frame,
            text="🗑️ Clear Cache",
            command=self.clear_cache_confirm,
            bg='#e63946',
            fg='white',
            font=('Arial', 10, 'bold'),
            width=15,
            cursor='hand2',
            relief=tk.FLAT
        )
        clear_cache_btn.pack(side=tk.LEFT, padx=5)
        
        export_key_btn = tk.Button(
            cache_btn_frame,
            text="🔑 Export Key",
            command=self.export_encryption_key,
            bg='#2a9d8f',
            fg='white',
            font=('Arial', 10, 'bold'),
            width=15,
            cursor='hand2',
            relief=tk.FLAT
        )
        export_key_btn.pack(side=tk.LEFT, padx=5)
    
        # Action buttons
        action_frame = tk.Frame(container, bg='#1a1a2e')
        action_frame.pack(fill=tk.X, pady=10)
    
        button_container = tk.Frame(action_frame, bg='#1a1a2e')
        button_container.pack(expand=True)
    
        chatbot_btn = tk.Button(
            button_container,
            text="💬 Open Chatbot",
            command=self.open_chatbot,
            bg='#6a4c93',
            fg='white',
            font=('Arial', 11, 'bold'),
            width=20,
            height=2,
            cursor='hand2',
            relief=tk.FLAT
        )
        chatbot_btn.pack(side=tk.LEFT, padx=10)
    
        export_btn = tk.Button(
            button_container,
            text="💾 Export Results",
            command=self.export_results,
            bg='#2a9d8f',
            fg='white',
            font=('Arial', 11, 'bold'),
            width=20,
            height=2,
            cursor='hand2',
            relief=tk.FLAT
        )
        export_btn.pack(side=tk.LEFT, padx=10)
        
        # Initialize cache stats on load
        self.refresh_cache_stats()
        
    def log_message(self, message):
        """Add message to status log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.status_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.status_text.see(tk.END)
        self.root.update()
        
    def start_recording(self):
        """Start meeting recording"""
        # Get configuration
        meeting_url = self.url_entry.get().strip()
        platform = self.platform_var.get()
        password = self.password_entry.get().strip() or None
        name = self.name_entry.get().strip()
        
        try:
            duration = int(self.duration_entry.get().strip())
            if duration == 0:
                duration = None
        except ValueError:
            messagebox.showerror("Error", "Invalid duration value!")
            return
        
        if not meeting_url:
            messagebox.showerror("Error", "Please enter a meeting URL!")
            return
        
        # Disable start button, enable stop button
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.recording_active = True
        
        # Start recording in a separate thread
        def record_thread():
            try:
                self.log_message("Initializing meeting recorder...")
                self.recorder = MeetingRecorder(
                    output_dir=str(self.transcripts_dir),
                    auto_merge=True,
                    keep_temp_files=False
                )
                
                self.log_message("Setting up browser...")
                self.recorder.setup_browser()
                
                self.log_message(f"Joining {platform} meeting...")
                if platform == "zoom":
                    success = self.recorder.join_zoom(meeting_url, name, password)
                else:
                    success = self.recorder.join_google_meet(meeting_url, name)
                
                if not success:
                    self.log_message("❌ Failed to join meeting!")
                    self.recording_active = False
                    return
                
                self.log_message("✅ Joined meeting successfully!")
                self.log_message("🔴 Recording started...")
                
                self.recorder.start_recording(duration=duration)
                
                # Wait for recording to complete or manual stop
                if duration:
                    import time
                    time.sleep(duration)
                    if self.recording_active:
                        self.stop_recording()
                        
            except Exception as e:
                self.log_message(f"❌ Error: {str(e)}")
                self.recording_active = False
                self.start_btn.config(state=tk.NORMAL)
                self.stop_btn.config(state=tk.DISABLED)
        
        thread = threading.Thread(target=record_thread, daemon=True)
        thread.start()
        
    def stop_recording(self):
        """Stop the recording"""
        if not self.recording_active or not self.recorder:
            return
        
        self.log_message("⏹️ Stopping recording...")
        self.recording_active = False
        
        def stop_thread():
            try:
                self.recorder.stop_recording()
                self.log_message("✅ Recording stopped and saved!")
                self.log_message("⚙️ Processing recording...")
                
                # Process the recording
                self.process_recording()
                
            except Exception as e:
                self.log_message(f"❌ Error stopping: {str(e)}")
            finally:
                self.start_btn.config(state=tk.NORMAL)
                self.stop_btn.config(state=tk.DISABLED)
                if self.recorder:
                    self.recorder.cleanup()
        
        thread = threading.Thread(target=stop_thread, daemon=True)
        thread.start()
        
    def process_recording(self):
        """Process the recorded file"""
        # Find the most recent recording
        files = list(self.transcripts_dir.glob("*.mp4")) + \
                list(self.transcripts_dir.glob("*.avi")) + \
                list(self.transcripts_dir.glob("*.wav"))
        
        if not files:
            self.log_message("❌ No recording files found!")
            return
        
        latest_file = max(files, key=lambda x: x.stat().st_mtime)
        self.log_message(f"📁 Processing: {latest_file.name}")
        
        self.transcribe_and_summarize([latest_file])
        
    def upload_files(self):
        """Handle file upload"""
        filetypes = [
            ("Audio/Video files", "*.mp4 *.avi *.wav *.mp3 *.m4a *.mov"),
            ("All files", "*.*")
        ]
        
        files = filedialog.askopenfilenames(
            title="Select Meeting Recordings",
            filetypes=filetypes
        )
        
        if files:
            self.current_files = [Path(f) for f in files]
            self.file_listbox.delete(0, tk.END)
            for file in self.current_files:
                self.file_listbox.insert(tk.END, file.name)
            
            messagebox.showinfo("Success", f"✅ Loaded {len(files)} file(s)")
            
    def process_uploaded_files(self):
        """Process uploaded files"""
        if not self.current_files:
            messagebox.showwarning("Warning", "No files selected!")
            return
        
        self.notebook.select(0)  # Switch to record tab to show status
        self.log_message("⚙️ Processing uploaded files...")
        
        def process_thread():
            self.transcribe_and_summarize(self.current_files)
        
        thread = threading.Thread(target=process_thread, daemon=True)
        thread.start()
    
    def run_prosody_analysis(self, audio_file):
        """Run prosody extraction on the audio file"""
        try:
            self.log_message("🎵 Running prosody analysis (this may take a few minutes)...")
            
            # Find prosody script in current directory
            prosody_script = Path("prosody_extraction_2.py")
            
            if not prosody_script.exists():
                self.log_message("⚠️ Warning: prosody_extraction_2.py not found")
                return None
            
            # Clean up any existing prosody output file
            prosody_output = Path("Prosody_annotations.txt")
            if prosody_output.exists():
                try:
                    prosody_output.unlink()
                except Exception as e:
                    self.log_message(f"Could not delete old prosody file: {e}")
            
            # Run prosody extraction
            self.log_message(f"▶️ Executing prosody analysis...")
            
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            
            result = subprocess.run(
                [sys.executable, str(prosody_script), "--audio", str(audio_file), "--whisper_model", "base"],
                capture_output=True,
                text=True,
                timeout=600,
                cwd=str(Path.cwd()),
                env=env,
                encoding='utf-8',
                errors='replace'
            )
            
            if prosody_output.exists():
                with open(prosody_output, 'r', encoding='utf-8') as f:
                    prosody_content = f.read()
                
                if prosody_content.strip():
                    self.log_message("✅ Prosody analysis complete!")
                    return prosody_content
            
            self.log_message("⚠️ No prosody data generated")
            return None
                
        except subprocess.TimeoutExpired:
            self.log_message("⏱️ Prosody analysis timed out")
            return None
        except Exception as e:
            self.log_message(f"❌ Prosody analysis error: {str(e)}")
            return None
    
    def process_deepgram(self, audio_file):
        """Process audio with Deepgram for diarization and intents"""
        if not self.deepgram_processor:
            self.log_message("⚠️ Deepgram processor not available")
            return None, None, None
        
        try:
            self.log_message("🎤 Running Deepgram speaker diarization and intent detection...")
            
            # Process with Deepgram
            response = self.deepgram_processor.process_audio(str(audio_file))
            
            # Create diarized transcript
            diarized_transcript = self.deepgram_processor.create_diarized_transcript(response)
            
            # Extract intents
            intent_text, intent_segments = self.deepgram_processor.extract_intents(response)
            
            self.log_message(f"✅ Deepgram processing complete!")
            self.log_message(f"   Found {len(intent_segments)} intent segments")
            
            return diarized_transcript, intent_text, response
            
        except Exception as e:
            self.log_message(f"❌ Deepgram processing error: {str(e)}")
            import traceback
            self.log_message(f"Stack trace: {traceback.format_exc()[-300:]}")
            return None, None, None
        
    def transcribe_and_summarize(self, files):
        """Transcribe files and generate summary with LFV caching"""
        try:
            transcripts = []
            all_prosody_content = []
            all_diarized_content = []
            all_intent_content = []
            
            for file in files:
                meeting_id = file.stem  # Use filename as unique ID
                
                # ===== CHECK CACHE FIRST =====
                cached_features = self.lfv.retrieve_features(meeting_id)
                
                if cached_features:
                    self.log_message(f"📦 Using cached features for: {file.name}")
                    
                    # Reconstruct from cached data
                    prosody_features = cached_features.get('prosody', {})
                    intent_features = cached_features.get('intent', {})
                    speaker_features = cached_features.get('speaker', {})
                    
                    # Get transcript (still need this)
                    self.log_message(f"📝 Transcribing: {file.name}")
                    text = transcribe_file(str(file))
                    txt_file = self.transcripts_dir / f"{file.stem}.txt"
                    with open(txt_file, 'w', encoding='utf-8') as f:
                        f.write(text)
                    transcripts.append(text)
                    self.log_message(f"✅ Transcript saved: {txt_file.name}")
                    
                    # Reconstruct prosody result from cache
                    if 'formatted_text' in prosody_features:
                        prosody_result = prosody_features['formatted_text']
                    else:
                        prosody_result = self._format_prosody_from_cache(prosody_features)
                    
                    # Reconstruct diarized transcript from cache
                    if 'formatted_text' in speaker_features:
                        diarized_transcript = speaker_features['formatted_text']
                    else:
                        diarized_transcript = None
                    
                    # Reconstruct intent text from cache
                    if 'formatted_text' in intent_features:
                        intent_text = intent_features['formatted_text']
                    else:
                        intent_text = None
                    
                    deepgram_response = None  # Skip Deepgram for cached items
                    
                else:
                    # ===== PROCESS NEW FILE =====
                    self.log_message(f"🔄 Processing and caching: {file.name}")
                    
                    self.log_message(f"📝 Transcribing: {file.name}")
                    text = transcribe_file(str(file))
                    
                    # Save basic transcript
                    txt_file = self.transcripts_dir / f"{file.stem}.txt"
                    with open(txt_file, 'w', encoding='utf-8') as f:
                        f.write(text)
                    
                    transcripts.append(text)
                    self.log_message(f"✅ Transcript saved: {txt_file.name}")
                    
                    # Run Deepgram processing
                    diarized_transcript, intent_text, deepgram_response = self.process_deepgram(file)
                    
                    # Run prosody analysis (moved from later in code)
                    prosody_result = self.run_prosody_analysis(file)
                    
                    # ===== EXTRACT AND CACHE FEATURES =====
                    try:
                        speaker_features = self._extract_speaker_features(file, diarized_transcript)
                        prosody_features = self._parse_prosody_features(prosody_result or "", file.name)
                        intent_features = self._extract_intent_features(intent_text or "", prosody_result or "")
                        
                        # Store in secure vault
                        cache_success = self.lfv.store_features(
                            meeting_id=meeting_id,
                            speaker_features=speaker_features,
                            prosody_features=prosody_features,
                            intent_features=intent_features,
                            metadata={
                                'filename': file.name,
                                'file_size': file.stat().st_size,
                                'duration': prosody_features.get('duration', 0)
                            }
                        )
                        
                        if cache_success:
                            self.log_message(f"🔒 Features encrypted and cached for: {file.name}")
                    except Exception as cache_error:
                        self.log_message(f"⚠️ Caching failed (continuing): {str(cache_error)}")
                
                # Continue with existing file saving and display formatting
                if diarized_transcript:
                    formatted_diarized = (
                        f"\n{'='*80}\n"
                        f"SPEAKER DIARIZATION FOR: {file.name}\n"
                        f"{'='*80}\n\n"
                        f"{diarized_transcript}"
                    )
                    all_diarized_content.append(formatted_diarized)
                    
                    # Save diarized transcript
                    diarized_file = self.transcripts_dir / f"{file.stem}_diarized.txt"
                    with open(diarized_file, 'w', encoding='utf-8') as f:
                        f.write(diarized_transcript)
                    self.log_message(f"✅ Diarized transcript saved: {diarized_file.name}")
                
                if intent_text:
                    formatted_intents = (
                        f"\n{'='*80}\n"
                        f"INTENT DETECTION FOR: {file.name}\n"
                        f"{'='*80}\n\n"
                        f"{intent_text}"
                    )
                    all_intent_content.append(formatted_intents)
                    
                    # Save intent analysis
                    intent_file = self.transcripts_dir / f"{file.stem}_intents.txt"
                    with open(intent_file, 'w', encoding='utf-8') as f:
                        f.write(intent_text)
                    self.log_message(f"✅ Intent analysis saved: {intent_file.name}")
                
                if deepgram_response:
                    # Save full Deepgram JSON response
                    json_file = self.transcripts_dir / f"{file.stem}_deepgram.json"
                    with open(json_file, 'w', encoding='utf-8') as f:
                        json.dump(deepgram_response, f, indent=2)
                    self.log_message(f"✅ Deepgram JSON saved: {json_file.name}")
                
                # Format and save prosody analysis (prosody_result already obtained above)
                if prosody_result:
                    formatted_prosody = (
                        f"\n{'='*80}\n"
                        f"PROSODY ANALYSIS FOR: {file.name}\n"
                        f"{'='*80}\n\n"
                        f"{prosody_result}"
                    )
                    all_prosody_content.append(formatted_prosody)
                    
                    # Save prosody analysis
                    prosody_file = self.transcripts_dir / f"{file.stem}_prosody.txt"
                    with open(prosody_file, 'w', encoding='utf-8') as f:
                        f.write(prosody_result)
                    self.log_message(f"✅ Prosody analysis saved: {prosody_file.name}")
                    
                    # Save combined file with ALL analyses for chatbot
                    combined_file = self.transcripts_dir / f"{file.stem}_complete_analysis.txt"
                    with open(combined_file, 'w', encoding='utf-8') as f:
                        f.write(f"COMPLETE MEETING ANALYSIS\n")
                        f.write(f"{'='*80}\n\n")
                        f.write(f"FILE: {file.name}\n")
                        f.write(f"DATE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                        f.write(f"{'='*80}\n\n")
                        
                        if diarized_transcript:
                            f.write(f"SPEAKER DIARIZED TRANSCRIPT:\n{diarized_transcript}\n\n")
                            f.write(f"{'='*80}\n\n")
                        else:
                            f.write(f"RAW TRANSCRIPT:\n{text}\n\n")
                            f.write(f"{'='*80}\n\n")
                        
                        if intent_text:
                            f.write(f"{intent_text}\n\n")
                            f.write(f"{'='*80}\n\n")
                        
                        if prosody_result:
                            f.write(f"PROSODY ANALYSIS:\n{prosody_result}\n")
                    
                    self.log_message(f"✅ Complete analysis saved: {combined_file.name}")
            
            # Combine all transcripts for summary
            combined_transcript = "\n\n=== Next Recording ===\n\n".join(transcripts)
            
            # Create the display content for transcript window
            display_content = "="*80 + "\n"
            display_content += "COMPLETE MEETING ANALYSIS\n"
            display_content += "="*80 + "\n\n"
            
            # Add Diarized Transcripts
            if all_diarized_content:
                display_content += "\n\n".join(all_diarized_content)
                display_content += "\n\n" + "="*80 + "\n\n"
            
            # Add Intent Analysis
            if all_intent_content:
                display_content += "\n\n".join(all_intent_content)
                display_content += "\n\n" + "="*80 + "\n\n"
            
            # Add Prosody Analysis
            if all_prosody_content:
                display_content += "\n\n".join(all_prosody_content)
                display_content += "\n\n" + "="*80 + "\n\n"
            
            # Add raw transcript if no other analysis available
            if not all_diarized_content:
                display_content += "RAW TRANSCRIPT TEXT\n"
                display_content += "="*80 + "\n\n"
                display_content += combined_transcript
            
            self.log_message("📊 Generating summary and to-do list...")
            summary = summarize(combined_transcript)
            
            # Display results in the transcript window
            self.transcript_text.delete('1.0', tk.END)
            self.transcript_text.insert('1.0', display_content)
            
            # Add color coding for different sections
            self.highlight_analysis_sections()
            
            self.summary_text.delete('1.0', tk.END)
            self.summary_text.insert('1.0', summary)
            
            # Switch to results tab
            self.notebook.select(2)
            
            # Display cache statistics
            try:
                stats = self.lfv.get_statistics()
                self.log_message(f"📊 Cache Stats: {stats['cached_meetings']} meetings, "
                                f"Hit Rate: {stats.get('hit_rate', 0):.1%}")
            except Exception as stats_error:
                self.log_message(f"⚠️ Could not get cache stats: {stats_error}")
            
            self.log_message("✅ Processing complete!")
            messagebox.showinfo("Success", "✅ Files processed successfully with complete analysis!")
            
        except Exception as e:
            self.log_message(f"❌ Error: {str(e)}")
            import traceback
            self.log_message(f"Stack trace: {traceback.format_exc()}")
            messagebox.showerror("Error", f"Processing failed: {str(e)}")
    
    def _extract_speaker_features(self, audio_file, diarized_transcript=None) -> np.ndarray:
        """
        Extract speaker embeddings from audio
        
        Args:
            audio_file: Path to audio file
            diarized_transcript: Optional diarized transcript text
            
        Returns:
            np.ndarray: Speaker embedding vector
        """
        try:
            import librosa
            
            # Load audio (first 60 seconds for efficiency)
            y, sr = librosa.load(str(audio_file), sr=16000, duration=60)
            
            # Extract MFCC features (speaker characteristics)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
            
            # Extract spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            
            # Aggregate features
            speaker_embedding = np.concatenate([
                np.mean(mfccs, axis=1),
                np.std(mfccs, axis=1),
                [np.mean(spectral_centroids)],
                [np.mean(spectral_rolloff)]
            ])
            
            # Add speaker count from diarization if available
            if diarized_transcript:
                speaker_count = len(set(
                    line.split(':')[0].strip()
                    for line in diarized_transcript.split('\n')
                    if line.startswith('SPEAKER')
                ))
                speaker_embedding = np.append(speaker_embedding, speaker_count)
            
            self.log_message(f"✅ Extracted speaker features: shape={speaker_embedding.shape}")
            return speaker_embedding
            
        except Exception as e:
            self.log_message(f"⚠️ Speaker feature extraction failed: {e}")
            return np.zeros(83)  # Return zero vector as fallback
    
    def _parse_prosody_features(self, prosody_text: str, filename: str) -> dict:
        """
        Parse prosody annotations into structured format for caching
        
        Args:
            prosody_text: Raw prosody analysis text
            filename: Original audio filename
            
        Returns:
            dict: Structured prosody features
        """
        if not prosody_text:
            return {'segments': [], 'summary': {}, 'formatted_text': ''}
        
        prosody_dict = {
            'filename': filename,
            'formatted_text': prosody_text,  # Store for display
            'segments': [],
            'summary': {
                'high_urgency_count': 0,
                'medium_urgency_count': 0,
                'low_urgency_count': 0,
                'emotions_detected': set(),
                'total_segments': 0
            },
            'duration': 0
        }
        
        lines = prosody_text.split('\n')
        
        for line in lines:
            # Parse segment annotations: [0.00-2.02] text=... urg=High emo=frustration
            if line.strip().startswith('[') and '-' in line and ']' in line:
                try:
                    # Extract time range
                    time_part = line.split(']')[0].strip('[')
                    start, end = map(float, time_part.split('-'))
                    
                    # Extract urgency
                    urgency = 'Low'
                    if 'urg=High' in line:
                        urgency = 'High'
                        prosody_dict['summary']['high_urgency_count'] += 1
                    elif 'urg=Medium' in line:
                        urgency = 'Medium'
                        prosody_dict['summary']['medium_urgency_count'] += 1
                    else:
                        prosody_dict['summary']['low_urgency_count'] += 1
                    
                    # Extract emotion
                    emotion = None
                    if 'emo=' in line:
                        emotion = line.split('emo=')[1].split()[0].strip()
                        if emotion not in ['None', '']:
                            prosody_dict['summary']['emotions_detected'].add(emotion)
                    
                    # Extract text
                    text = ''
                    if 'text=' in line:
                        text = line.split('text=')[1].split('urg=')[0].strip()
                    
                    prosody_dict['segments'].append({
                        'start': start,
                        'end': end,
                        'duration': end - start,
                        'text': text,
                        'urgency': urgency,
                        'emotion': emotion
                    })
                    
                    prosody_dict['duration'] = max(prosody_dict['duration'], end)
                    
                except Exception as e:
                    self.log_message(f"⚠️ Failed to parse prosody line: {line[:50]}... ({e})")
        
        prosody_dict['summary']['total_segments'] = len(prosody_dict['segments'])
        prosody_dict['summary']['emotions_detected'] = list(prosody_dict['summary']['emotions_detected'])
        
        return prosody_dict
    
    def _extract_intent_features(self, intent_text: str, prosody_text: str) -> dict:
        """
        Extract intent and action items from analysis
        
        Args:
            intent_text: Deepgram intent detection output
            prosody_text: Prosody analysis output
            
        Returns:
            dict: Intent features
        """
        intent_dict = {
            'deepgram_intents': [],
            'prosody_based_intents': {
                'high_urgency_items': [],
                'emotional_highlights': []
            },
            'summary': {
                'total_intents': 0,
                'urgent_action_items': 0
            }
        }
        
        # Parse Deepgram intents
        if intent_text:
            lines = intent_text.split('\n')
            current_segment = None
            
            for line in lines:
                if line.startswith('--- Segment'):
                    current_segment = {'text': '', 'intents': []}
                elif line.startswith('Text:') and current_segment is not None:
                    current_segment['text'] = line.replace('Text:', '').strip()
                elif line.strip().startswith('- ') and current_segment is not None:
                    intent = line.strip()[2:].strip()
                    current_segment['intents'].append(intent)
                elif line.strip() == '' and current_segment and current_segment['text']:
                    intent_dict['deepgram_intents'].append(current_segment)
                    current_segment = None
        
        # Extract high urgency items from prosody
        if prosody_text:
            for line in prosody_text.split('\n'):
                if 'urg=High' in line:
                    text = ''
                    if 'text=' in line:
                        text = line.split('text=')[1].split('urg=')[0].strip()
                    if text:
                        intent_dict['prosody_based_intents']['high_urgency_items'].append(text)
                
                # Extract emotional highlights
                if 'emo=' in line and 'emo=None' not in line:
                    emotion = line.split('emo=')[1].split()[0].strip()
                    text = ''
                    if 'text=' in line:
                        text = line.split('text=')[1].split('urg=')[0].strip()
                    if text and emotion:
                        intent_dict['prosody_based_intents']['emotional_highlights'].append({
                            'emotion': emotion,
                            'text': text
                        })
        
        # Calculate summary
        intent_dict['summary']['total_intents'] = len(intent_dict['deepgram_intents'])
        intent_dict['summary']['urgent_action_items'] = len(
            intent_dict['prosody_based_intents']['high_urgency_items']
        )
        
        return intent_dict
    
    def _format_prosody_from_cache(self, prosody_features: dict) -> str:
        """
        Reconstruct prosody display text from cached features
        
        Args:
            prosody_features: Cached prosody dictionary
            
        Returns:
            str: Formatted prosody text for display
        """
        if 'formatted_text' in prosody_features:
            return prosody_features['formatted_text']
        
        # Reconstruct from segments if formatted_text not available
        output = "=== Segment annotations ===\n\n"
        
        for seg in prosody_features.get('segments', []):
            output += (
                f"[{seg['start']:.2f}-{seg['end']:.2f}] "
                f"text={seg['text']} \n"
                f" urg={seg['urgency']} \n"
                f" emo={seg.get('emotion', 'None')} \n\n"
            )
        
        summary = prosody_features.get('summary', {})
        output += "\n=== Summary ===\n"
        output += f"Total segments: {summary.get('total_segments', 0)}\n"
        output += f"High urgency: {summary.get('high_urgency_count', 0)}\n"
        output += f"Medium urgency: {summary.get('medium_urgency_count', 0)}\n"
        output += f"Emotions detected: {', '.join(summary.get('emotions_detected', []))}\n"
        
        return output
    
    def highlight_analysis_sections(self):
        """Add color highlighting to different analysis sections"""
        try:
            # Configure tags for different sections
            self.transcript_text.tag_configure("header", foreground="#00d9ff", font=('Arial', 11, 'bold'))
            self.transcript_text.tag_configure("speaker", foreground="#ffaa00", font=('Arial', 10, 'bold'))
            self.transcript_text.tag_configure("intent", foreground="#51cf66", font=('Arial', 10, 'bold'))
            self.transcript_text.tag_configure("high_urgency", foreground="#ff4444", font=('Arial', 10, 'bold'))
            self.transcript_text.tag_configure("medium_urgency", foreground="#ffaa00", font=('Arial', 10, 'bold'))
            self.transcript_text.tag_configure("low_urgency", foreground="#00ff00")
            self.transcript_text.tag_configure("emotion", foreground="#00d9ff", font=('Arial', 10, 'italic'))
            
            content = self.transcript_text.get('1.0', tk.END)
            
            # Highlight headers
            for match in ['COMPLETE MEETING ANALYSIS', 'SPEAKER DIARIZATION', 'INTENT DETECTION', 
                          'PROSODY ANALYSIS', 'RAW TRANSCRIPT', 'Segment annotations', 
                          'Annotated summary sentences']:
                start_idx = '1.0'
                while True:
                    start_idx = self.transcript_text.search(match, start_idx, tk.END)
                    if not start_idx:
                        break
                    end_idx = f"{start_idx}+{len(match)}c"
                    self.transcript_text.tag_add("header", start_idx, end_idx)
                    start_idx = end_idx
            
            # Highlight speakers
            start_idx = '1.0'
            while True:
                start_idx = self.transcript_text.search("SPEAKER", start_idx, tk.END)
                if not start_idx:
                    break
                end_idx = self.transcript_text.search(":", start_idx, tk.END)
                if end_idx:
                    end_idx = f"{end_idx}+1c"
                    self.transcript_text.tag_add("speaker", start_idx, end_idx)
                    start_idx = end_idx
                else:
                    break
            
            # Highlight intents
            start_idx = '1.0'
            while True:
                start_idx = self.transcript_text.search("Intents:", start_idx, tk.END)
                if not start_idx:
                    break
                end_idx = f"{start_idx}+8c"
                self.transcript_text.tag_add("intent", start_idx, end_idx)
                start_idx = end_idx
            
            # Highlight urgency levels
            for urgency, tag in [("High", "high_urgency"), ("Medium", "medium_urgency"), ("Low", "low_urgency")]:
                pattern = f"urg={urgency}"
                start_idx = '1.0'
                while True:
                    start_idx = self.transcript_text.search(pattern, start_idx, tk.END)
                    if not start_idx:
                        break
                    end_idx = f"{start_idx}+{len(pattern)}c"
                    self.transcript_text.tag_add(tag, start_idx, end_idx)
                    start_idx = end_idx
            
            # Highlight emotions
            start_idx = '1.0'
            while True:
                start_idx = self.transcript_text.search("emo=", start_idx, tk.END)
                if not start_idx:
                    break
                end_idx = self.transcript_text.search("\n", start_idx, tk.END)
                if end_idx:
                    self.transcript_text.tag_add("emotion", start_idx, end_idx)
                    start_idx = end_idx
                else:
                    break
                    
        except Exception as e:
            self.log_message(f"⚠️ Could not apply syntax highlighting: {str(e)}")
            
    def open_chatbot(self):
        """Open the chatbot interface"""
        try:
            # Check if there are transcript files
            txt_files = list(self.transcripts_dir.glob("*.txt"))
            if not txt_files:
                messagebox.showwarning("Warning", "No transcripts available! Process a recording first.")
                return
            
            # Create chatbot window
            chatbot_window = tk.Toplevel(self.root)
            chatbot_window.title("💬 Meeting Chatbot")
            chatbot_window.geometry("700x800")
            chatbot_window.configure(bg='#1a1a2e')
            
            # Initialize chatbot
            chatbot = MeetingChatbot(self.groq_api_key, str(self.transcripts_dir))
            
            # Create chat interface
            self.create_chatbot_ui(chatbot_window, chatbot)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open chatbot: {str(e)}")
            
    def create_chatbot_ui(self, window, chatbot):
        """Create chatbot UI in the given window"""
        
        # Header
        header = tk.Frame(window, bg='#16213e', height=60)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        
        tk.Label(
            header,
            text="💬 Meeting Chatbot with Full Analysis",
            font=('Arial', 16, 'bold'),
            bg='#16213e',
            fg='#00d9ff'
        ).pack(pady=15)
        
        # Chat display
        chat_frame = tk.Frame(window, bg='#1a1a2e')
        chat_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        chat_display = scrolledtext.ScrolledText(
            chat_frame,
            font=('Arial', 10),
            bg='#0f3460',
            fg='white',
            wrap=tk.WORD
        )
        chat_display.pack(fill=tk.BOTH, expand=True)
        chat_display.config(state=tk.DISABLED)
        
        # Input area
        input_frame = tk.Frame(window, bg='#1a1a2e')
        input_frame.pack(fill=tk.X, padx=15, pady=(0, 15))
        
        message_entry = tk.Entry(
            input_frame,
            font=('Arial', 11),
            bg='#16213e',
            fg='white',
            insertbackground='white'
        )
        message_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=10)
        
        def send_message():
            message = message_entry.get().strip()
            if not message:
                return
            
            message_entry.delete(0, tk.END)
            
            # Display user message
            chat_display.config(state=tk.NORMAL)
            chat_display.insert(tk.END, f"\n👤 You: {message}\n", 'user')
            chat_display.insert(tk.END, "\n🤖 Bot: Thinking...\n", 'bot')
            chat_display.config(state=tk.DISABLED)
            chat_display.see(tk.END)
            
            def get_response():
                response = chatbot.ask(message)
                chat_display.config(state=tk.NORMAL)
                chat_display.delete("end-2l", "end-1l")
                chat_display.insert(tk.END, f"\n🤖 Bot: {response}\n", 'bot')
                chat_display.insert(tk.END, "-" * 80 + "\n", 'separator')
                chat_display.config(state=tk.DISABLED)
                chat_display.see(tk.END)
            
            thread = threading.Thread(target=get_response, daemon=True)
            thread.start()
        
        send_btn = tk.Button(
            input_frame,
            text="Send",
            command=send_message,
            bg='#6a4c93',
            fg='white',
            font=('Arial', 10, 'bold'),
            width=10,
            cursor='hand2',
            relief=tk.FLAT
        )
        send_btn.pack(side=tk.RIGHT, padx=(10, 0))
        
        message_entry.bind('<Return>', lambda e: send_message())
        
        # Welcome message
        chat_display.config(state=tk.NORMAL)
        welcome_msg = (
            "Welcome! I have access to your meeting transcripts with:\n"
            "  • Speaker diarization (who said what)\n"
            "  • Intent detection (what people wanted)\n"
            "  • Prosody analysis (urgency and emotions)\n\n"
            "Ask me anything about your meetings!"
        )
        chat_display.insert(tk.END, welcome_msg + "\n")
        chat_display.insert(tk.END, "-" * 80 + "\n", 'separator')
        chat_display.config(state=tk.DISABLED)
        
    def export_results(self):
        """Export transcript and summary to file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_file = self.transcripts_dir / f"meeting_report_{timestamp}.txt"
            
            transcript = self.transcript_text.get('1.0', tk.END)
            summary = self.summary_text.get('1.0', tk.END)
            
            with open(export_file, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("NEUROSCRIBEAI - COMPLETE MEETING REPORT\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 80 + "\n\n")
                f.write("ANALYSIS (Diarization, Intents, Prosody)\n")
                f.write("-" * 80 + "\n")
                f.write(transcript)
                f.write("\n\n")
                f.write("SUMMARY & TO-DO LIST\n")
                f.write("-" * 80 + "\n")
                f.write(summary)
            
            messagebox.showinfo("Success", f"✅ Report exported to:\n{export_file}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Export failed: {str(e)}")
    
    def refresh_cache_stats(self):
        """Update cache statistics display"""
        try:
            stats = self.lfv.get_statistics()
            
            stats_text = (
                f"📊 Cache Statistics\n"
                f"─────────────────────────────\n"
                f"Cached Meetings: {stats['cached_meetings']}\n"
                f"Total Reads: {stats.get('total_reads', 0)}\n"
                f"Total Writes: {stats.get('total_writes', 0)}\n"
                f"Cache Hits: {stats.get('cache_hits', stats.get('hits', 0))}\n"
                f"Cache Misses: {stats.get('cache_misses', stats.get('misses', 0))}\n"
                f"Hit Rate: {stats.get('hit_rate', 0):.1%}\n"
                f"Total Size: {stats.get('total_size_mb', 0):.2f} MB\n"
                f"Vault Dir: {stats.get('vault_dir', 'N/A')}"
            )
            
            self.cache_stats_label.config(text=stats_text)
            self.log_message("✅ Cache statistics refreshed")
            
        except Exception as e:
            self.cache_stats_label.config(text=f"⚠️ Error loading cache stats: {str(e)}")
            self.log_message(f"❌ Failed to refresh cache stats: {e}")
    
    def view_cache_entries(self):
        """Display list of cached meetings in a popup window"""
        try:
            cached_meetings = self.lfv.list_cached_meetings()
            
            if not cached_meetings:
                messagebox.showinfo("Cache Empty", "No meetings cached yet.")
                return
            
            # Create popup window
            cache_window = tk.Toplevel(self.root)
            cache_window.title("🔒 Cached Meetings")
            cache_window.geometry("800x600")
            cache_window.configure(bg='#1a1a2e')
            
            # Header
            header = tk.Label(
                cache_window,
                text="🔒 Secure Feature Vault - Cached Meetings",
                font=('Arial', 16, 'bold'),
                bg='#16213e',
                fg='#00d9ff',
                pady=15
            )
            header.pack(fill=tk.X)
            
            # Scrollable list
            list_frame = tk.Frame(cache_window, bg='#1a1a2e')
            list_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
            
            scrollbar = tk.Scrollbar(list_frame)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            cache_listbox = tk.Listbox(
                list_frame,
                font=('Courier', 9),
                bg='#0f3460',
                fg='white',
                selectbackground='#00d9ff',
                selectforeground='black',
                yscrollcommand=scrollbar.set
            )
            cache_listbox.pack(fill=tk.BOTH, expand=True)
            scrollbar.config(command=cache_listbox.yview)
            
            # Populate list with better formatting
            for meeting in cached_meetings:
                meeting_id = meeting.get('meeting_id', 'Unknown')
                metadata = meeting.get('metadata', {})
                timestamp = meeting.get('timestamp', 'N/A')[:19]
                
                # Count keys in features
                prosody_keys = len(meeting.get('prosody', {})) if isinstance(meeting.get('prosody'), dict) else 0
                intent_keys = len(meeting.get('intent', {})) if isinstance(meeting.get('intent'), dict) else 0
                speaker_keys = len(meeting.get('speaker', {})) if isinstance(meeting.get('speaker'), dict) else 0
                
                entry = (
                    f"ID: {meeting_id:<30} | "
                    f"Time: {timestamp} | "
                    f"P:{prosody_keys} I:{intent_keys} S:{speaker_keys}"
                )
                cache_listbox.insert(tk.END, entry)
            
            # Button frame
            btn_frame = tk.Frame(cache_window, bg='#1a1a2e')
            btn_frame.pack(fill=tk.X, padx=15, pady=10)
            
            def delete_selected():
                selection = cache_listbox.curselection()
                if not selection:
                    messagebox.showwarning("No Selection", "Please select a meeting to delete.")
                    return
                
                idx = selection[0]
                if idx < len(cached_meetings):
                    meeting_id = cached_meetings[idx].get('meeting_id', 'Unknown')
                    
                    if messagebox.askyesno("Confirm Delete", 
                                          f"Delete cached features for:\n{meeting_id}?"):
                        if self.lfv.delete_features(meeting_id):
                            cache_listbox.delete(idx)
                            cached_meetings.pop(idx)
                            messagebox.showinfo("Success", "Cache entry deleted.")
                            self.refresh_cache_stats()
                            self.log_message(f"🗑️ Deleted cache entry: {meeting_id}")
                        else:
                            messagebox.showerror("Error", "Failed to delete cache entry")
            
            delete_btn = tk.Button(
                btn_frame,
                text="🗑️ Delete Selected",
                command=delete_selected,
                bg='#e63946',
                fg='white',
                font=('Arial', 10, 'bold'),
                width=20,
                cursor='hand2',
                relief=tk.FLAT
            )
            delete_btn.pack(side=tk.LEFT, padx=5)
            
            close_btn = tk.Button(
                btn_frame,
                text="✖️ Close",
                command=cache_window.destroy,
                bg='#457b9d',
                fg='white',
                font=('Arial', 10, 'bold'),
                width=20,
                cursor='hand2',
                relief=tk.FLAT
            )
            close_btn.pack(side=tk.RIGHT, padx=5)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to view cache: {str(e)}")
            self.log_message(f"❌ Failed to view cache entries: {e}")
    
    def clear_cache_confirm(self):
        """Clear all cached features with confirmation"""
        if not messagebox.askyesno(
            "Confirm Clear Cache",
            "⚠️ This will delete ALL cached features!\n\n"
            "You will need to re-process all meetings.\n\n"
            "Continue?"
        ):
            return
        
        try:
            if self.lfv.clear_all():
                messagebox.showinfo("Success", "✅ Cache cleared successfully!")
                self.refresh_cache_stats()
                self.log_message("🗑️ All cached features cleared")
            else:
                messagebox.showerror("Error", "Failed to clear cache")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to clear cache: {str(e)}")
            self.log_message(f"❌ Failed to clear cache: {e}")
    
    def export_encryption_key(self):
        """Export master encryption key for backup"""
        try:
            output_path = filedialog.asksaveasfilename(
                title="Export Encryption Key",
                defaultextension=".key",
                filetypes=[("Key files", "*.key"), ("All files", "*.*")],
                initialfile="neuroscribe_master.key"
            )
            
            if not output_path:
                return
            
            if self.lfv.export_key(output_path):
                messagebox.showinfo(
                    "Success",
                    f"✅ Encryption key exported to:\n{output_path}\n\n"
                    "⚠️ Keep this file SECURE!\n"
                    "It can decrypt all your cached features."
                )
                self.log_message(f"🔑 Encryption key exported: {output_path}")
            else:
                messagebox.showerror("Error", "Failed to export key")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export key: {str(e)}")
            self.log_message(f"❌ Failed to export encryption key: {e}")


def main():
    """Main entry point"""
    root = tk.Tk()
    app = NeuroscribeAI(root)
    root.mainloop()


if __name__ == "__main__":
    main()