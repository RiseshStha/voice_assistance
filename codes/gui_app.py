"""
Simple GUI for the Nepali Student Voice Assistant.
Uses tkinter for a lightweight, cross-platform interface.
"""

import sys
import threading
import tkinter as tk
from tkinter import scrolledtext, messagebox
from pathlib import Path

# Add codes/ to path so we can import our pipeline modules
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from step5_pipeline import load_all_components, run_query
    from voice_assistance import transcribe_microphone
except ImportError as e:
    messagebox.showerror("Import Error", f"Failed to load pipeline modules: {e}")
    sys.exit(1)


class VoiceAssistantGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Nepali Voice Assistant")
        self.root.geometry("600x700")
        self.root.configure(bg="#F2F2F7")

        # Models
        self.asr_model = None
        self.svm_pipe = None
        self.kb_index = None

        self._build_ui()
        self._load_system_async()

    def _build_ui(self):
        """Construct the tkinter widgets."""
        # Top banner
        header = tk.Frame(self.root, bg="#007AFF", height=60)
        header.pack(fill=tk.X)
        header.pack_propagate(False)

        title_lbl = tk.Label(
            header,
            text="नेपाली विद्यार्थी सहायक (Nepali Student Assistant)",
            fg="white", bg="#007AFF", font=("Helvetica", 16, "bold")
        )
        title_lbl.pack(pady=15)

        # Chat history display
        chat_frame = tk.Frame(self.root, bg="#F2F2F7", padx=15, pady=15)
        chat_frame.pack(fill=tk.BOTH, expand=True)

        self.chat_display = scrolledtext.ScrolledText(
            chat_frame, wrap=tk.WORD, font=("Helvetica", 12),
            bg="white", relief=tk.FLAT, padx=10, pady=10, state=tk.DISABLED
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True)

        # Bottom Input Area
        input_frame = tk.Frame(self.root, bg="#F2F2F7", padx=15, pady=10)
        input_frame.pack(fill=tk.X, side=tk.BOTTOM)

        # Status label
        self.status_lbl = tk.Label(
            input_frame, text="Loading models... Please wait.",
            fg="#8E8E93", bg="#F2F2F7", font=("Helvetica", 10, "italic")
        )
        self.status_lbl.pack(anchor="w", pady=(0, 5))

        controls = tk.Frame(input_frame, bg="#F2F2F7")
        controls.pack(fill=tk.X)

        self.text_input = tk.Entry(
            controls, font=("Helvetica", 13), relief=tk.FLAT,
            highlightbackground="#D1D1D6", highlightthickness=1
        )
        self.text_input.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=8, padx=(0, 10))
        self.text_input.bind("<Return>", lambda event: self.handle_text_query())

        self.send_btn = tk.Button(
            controls, text="Send", font=("Helvetica", 12, "bold"),
            bg="#34C759", fg="white", relief=tk.FLAT, width=8,
            command=self.handle_text_query, state=tk.DISABLED
        )
        self.send_btn.pack(side=tk.LEFT, ipady=4, padx=(0, 10))

        self.mic_btn = tk.Button(
            controls, text="🎙️ Speak", font=("Helvetica", 12, "bold"),
            bg="#007AFF", fg="white", relief=tk.FLAT, width=10,
            command=self.handle_voice_query, state=tk.DISABLED
        )
        self.mic_btn.pack(side=tk.LEFT, ipady=4)

    def append_message(self, sender: str, text: str, color: str = "black", is_system: bool = False):
        """Append a message line to the chat display."""
        self.chat_display.config(state=tk.NORMAL)
        
        if is_system:
            self.chat_display.insert(tk.END, f"\n[System] {text}\n", "system")
            self.chat_display.tag_config("system", foreground="#8E8E93", font=("Helvetica", 10, "italic"))
        else:
            self.chat_display.insert(tk.END, f"\n{sender}:\n", "sender_tag")
            self.chat_display.insert(tk.END, f"{text}\n")
            
            # Simple alternating colours for user vs assistant
            tag_name = f"color_{color}"
            self.chat_display.tag_config(tag_name, foreground=color)
            
            # Colour just the sender name for visual clarity
            start_index = self.chat_display.index("end-3lines")
            end_index = self.chat_display.index("end-2lines")
            self.chat_display.tag_add(tag_name, start_index, end_index)

        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)

    def _load_system_async(self):
        """Load large AI models in a background thread to keep UI responsive."""
        def task():
            try:
                self.asr_model, self.svm_pipe, self.kb_index = load_all_components()
                self.root.after(0, self._on_models_loaded)
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Load Error", f"Model loading failed: {e}"))
                self.root.after(0, lambda: self.status_lbl.config(text="Model load failed.", fg="red"))

        threading.Thread(target=task, daemon=True).start()

    def _on_models_loaded(self):
        """Enable inputs once models finish loading."""
        self.status_lbl.config(text="System ready! Type a question or click Speak.", fg="#34C759")
        self.send_btn.config(state=tk.NORMAL)
        self.mic_btn.config(state=tk.NORMAL)
        self.append_message("", "System fully operational.", is_system=True)

    def disable_inputs(self, message: str):
        self.status_lbl.config(text=message, fg="#007AFF")
        self.text_input.config(state=tk.DISABLED)
        self.send_btn.config(state=tk.DISABLED)
        self.mic_btn.config(state=tk.DISABLED)

    def enable_inputs(self):
        self.status_lbl.config(text="Ready.", fg="#34C759")
        self.text_input.config(state=tk.NORMAL)
        self.send_btn.config(state=tk.NORMAL)
        self.mic_btn.config(state=tk.NORMAL)
        self.text_input.delete(0, tk.END)

    def handle_text_query(self):
        query = self.text_input.get().strip()
        if not query:
            return

        self.append_message("तपाईं (You)", query, color="#007AFF")
        self.disable_inputs("Processing...")

        def task():
            try:
                # Text query uses intent+retrieval+TTS pipeline
                res = run_query(query, self.svm_pipe, self.kb_index, speak=True, verbose=False)
                self.root.after(0, lambda: self._on_result(res))
            except Exception as e:
                self.root.after(0, lambda: self.append_message("Error", str(e), is_system=True))
                self.root.after(0, self.enable_inputs)

        threading.Thread(target=task, daemon=True).start()

    def handle_voice_query(self):
        self.disable_inputs("Listening... Please speak now.")
        self.append_message("", "🎙️ Listening...", is_system=True)

        def task():
            try:
                # Vosk microphone block
                audio_text = transcribe_microphone(self.asr_model, duration_hint=5)
                
                if not audio_text.strip():
                    self.root.after(0, lambda: self.append_message("", "I couldn't hear anything.", is_system=True))
                    self.root.after(0, self.enable_inputs)
                    return
                
                self.root.after(0, lambda: self.append_message("तपाईं (You)", audio_text, color="#007AFF"))
                self.root.after(0, lambda: self.status_lbl.config(text="Processing intent & answer..."))

                # Standard pipeline execution
                res = run_query(audio_text, self.svm_pipe, self.kb_index, speak=True, verbose=False)
                self.root.after(0, lambda: self._on_result(res))

            except Exception as e:
                self.root.after(0, lambda: self.append_message("Error", f"Voice error: {e}", is_system=True))
                self.root.after(0, self.enable_inputs)

        threading.Thread(target=task, daemon=True).start()

    def _on_result(self, res_dict: dict):
        answer = res_dict.get("answer", "Error retrieving answer.")
        intent = res_dict.get("intent", "Unknown")
        intent_conf = res_dict.get("intent_conf", 0.0)

        sys_msg = f"Intent identified: {intent} ({intent_conf:.1%})"
        self.append_message("", sys_msg, is_system=True)
        
        self.append_message("सहायक (Assistant)", answer, color="#FF9500")
        self.enable_inputs()


if __name__ == "__main__":
    app_root = tk.Tk()
    gui = VoiceAssistantGUI(app_root)
    app_root.bind("<Escape>", lambda e: app_root.quit())
    app_root.mainloop()
