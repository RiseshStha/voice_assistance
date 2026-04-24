"""
app.py — Part 4: Final GUI Application
======================================
The main entry point for the Nepali Voice Assistant.
Integrates ML Classification, Voice Engine, and User UI.
"""
import sys, pickle
import re
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import threading
from pathlib import Path

# Project imports
from config import MODEL_PATH, KB_INDEX_PATH
import voice_engine

class NepaliAssistantApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Nepali Educational Student Assistant")
        self.root.geometry("820x620")
        self.root.minsize(780, 560)
        
        # Load Models
        print("Initializing AI components...")
        try:
            with open(MODEL_PATH, "rb") as f:
                self.clf = pickle.load(f)
            self.stt = voice_engine.load_stt_model()
            self.kb = self._load_or_build_kb()
            hotword_texts = list(self.kb.get("questions", [])) + list(self.kb.get("answers", []))
            voice_engine.set_domain_hotwords(hotword_texts)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load models: {e}")
            sys.exit(1)

        self._build_ui()
        self._load_audio_devices()
        if not voice_engine.has_native_nepali_tts():
            self._append_line(
                "System note: Nepali TTS is unavailable because eSpeak/eSpeak-NG is not installed. "
                "Current fallback may not speak Nepali correctly.\n\n"
            )

    def _build_ui(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        # Style
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except Exception:
            pass

        header = ttk.Frame(self.root, padding=(16, 14))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)

        ttk.Label(header, text="Nepali Student Voice Assistant", font=("Segoe UI", 18, "bold")).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            header,
            text="Type a question or use the microphone to classify the intent.",
            font=("Segoe UI", 10),
            foreground="#444",
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))

        # Status + confidence
        status_row = ttk.Frame(self.root, padding=(16, 0, 16, 8))
        status_row.grid(row=0, column=0, sticky="e", padx=0, pady=0)

        self.status_var = tk.StringVar(value="Ready")
        self.status_lbl = ttk.Label(status_row, textvariable=self.status_var, foreground="#1f6feb")
        self.status_lbl.grid(row=0, column=0, sticky="e")

        body = ttk.Frame(self.root, padding=(16, 10))
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(0, weight=1)
        body.rowconfigure(1, weight=1)

        # Input panel
        input_panel = ttk.Labelframe(body, text="Input", padding=(12, 10))
        input_panel.grid(row=0, column=0, sticky="ew")
        input_panel.columnconfigure(1, weight=1)

        ttk.Label(input_panel, text="Prompt").grid(row=0, column=0, sticky="w")
        self.prompt_var = tk.StringVar()
        self.prompt_entry = ttk.Entry(input_panel, textvariable=self.prompt_var)
        self.prompt_entry.grid(row=0, column=1, sticky="ew", padx=(8, 8))
        self.prompt_entry.bind("<Return>", lambda _e: self.on_send_prompt())

        self.send_btn = ttk.Button(input_panel, text="Send", command=self.on_send_prompt)
        self.send_btn.grid(row=0, column=2, sticky="e")

        ttk.Label(input_panel, text="Max listen time").grid(row=1, column=0, sticky="w", pady=(10, 0))
        self.mic_seconds = tk.IntVar(value=10)
        self.mic_spin = ttk.Spinbox(input_panel, from_=2, to=12, textvariable=self.mic_seconds, width=6)
        self.mic_spin.grid(row=1, column=1, sticky="w", padx=(8, 0), pady=(10, 0))

        self.listen_btn = ttk.Button(input_panel, text="Start Listening", command=self.on_listen)
        self.listen_btn.grid(row=1, column=2, sticky="e", pady=(10, 0))

        ttk.Label(input_panel, text="Input device").grid(row=2, column=0, sticky="w", pady=(10, 0))
        self.mic_device_var = tk.StringVar()
        self.mic_device_combo = ttk.Combobox(
            input_panel,
            textvariable=self.mic_device_var,
            state="readonly",
            width=48,
        )
        self.mic_device_combo.grid(row=2, column=1, columnspan=2, sticky="ew", padx=(8, 0), pady=(10, 0))

        # Output panel
        output_panel = ttk.Labelframe(body, text="Conversation / Output", padding=(12, 10))
        output_panel.grid(row=1, column=0, sticky="nsew", pady=(12, 0))
        output_panel.columnconfigure(0, weight=1)
        output_panel.rowconfigure(0, weight=1)

        self.text_area = tk.Text(output_panel, wrap="word", font=("Segoe UI", 10))
        self.text_area.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(output_panel, orient="vertical", command=self.text_area.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.text_area.configure(yscrollcommand=scroll.set)

        self.conf_bar = ttk.Progressbar(body, mode="determinate", maximum=100)
        self.conf_bar.grid(row=2, column=0, sticky="ew", pady=(10, 0))

        self.prompt_entry.focus_set()

    def _load_or_build_kb(self):
        """
        Load retrieval KB if present; otherwise build it from CLEAN_XL.
        Keeps the app usable even if the user forgets to run build_kb.py.
        """
        try:
            if Path(KB_INDEX_PATH).exists():
                with open(KB_INDEX_PATH, "rb") as f:
                    kb = pickle.load(f)
                required = {"vectorizer", "X", "questions", "answers", "intents"}
                if required.issubset(set(kb.keys())) and len(kb["questions"]) > 0:
                    return kb
        except Exception:
            pass

        # Fallback: build quickly at startup
        import pandas as pd
        from sklearn.feature_extraction.text import TfidfVectorizer

        from config import CLEAN_XL

        df = pd.read_excel(CLEAN_XL).dropna(subset=["question", "answer", "intent"])
        questions = df["question"].astype(str).tolist()
        answers = df["answer"].astype(str).tolist()
        intents = df["intent"].astype(str).tolist()

        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            sublinear_tf=True,
            min_df=2,
        )
        X = vectorizer.fit_transform(questions)
        return {"vectorizer": vectorizer, "X": X, "questions": questions, "answers": answers, "intents": intents}

    def _load_audio_devices(self):
        self.audio_devices = voice_engine.list_input_devices()
        values = []
        selected_value = None
        preferred_id = voice_engine.choose_default_input_device()
        for dev in self.audio_devices:
            label = (
                f"{dev['id']}: {dev['name']} [{dev['hostapi']}, "
                f"{int(dev['default_samplerate'])} Hz]"
            )
            values.append(label)
            if dev["id"] == preferred_id:
                selected_value = label

        self.mic_device_combo["values"] = values
        if selected_value:
            self.mic_device_var.set(selected_value)
        elif values:
            self.mic_device_combo.current(0)

    def _selected_device_id(self):
        value = (self.mic_device_var.get() or "").strip()
        if not value:
            return None
        try:
            return int(value.split(":", 1)[0])
        except Exception:
            return None

    def _normalize_query(self, text: str) -> str:
        """
        Align live user input with the cleaned dataset used for training/KB lookup.
        """
        text = str(text or "").strip()
        text = re.sub(r"^[\d\u0966-\u096F]+[\.\)\s\u0964]+", "", text).strip()
        noise = [
            "लेख्नुहोस्।",
            "लेख्नुहोस्",
            "बताउनुहोस्।",
            "बताउनुहोस्",
            "उदाहरण दिनुहोस्।",
            "उदाहरण दिनुहोस्",
            "उल्लेख गर्नुहोस्।",
            "उल्लेख गर्नुहोस्",
        ]
        for word in noise:
            text = text.replace(word, "").strip()

        text = re.sub(r"\s+", " ", text)
        return text.strip(" ?।.!," )

    def _retrieve_answer(self, user_text: str, predicted_intent: str, min_sim: float = 0.25):
        """
        Retrieve the best matching answer from the KB (optionally restricted to predicted intent).
        Returns (answer_text, similarity, matched_question) or (None, sim, q).
        """
        user_text = (user_text or "").strip()
        if not user_text:
            return None, 0.0, None

        kb = self.kb
        vec = kb["vectorizer"]
        X = kb["X"]

        q_vec = vec.transform([user_text])

        # Prefer same-intent retrieval to reduce wrong-topic answers.
        intents = kb.get("intents")
        mask_idx = None
        if intents is not None and predicted_intent:
            mask_idx = [i for i, it in enumerate(intents) if it == predicted_intent]

        if mask_idx:
            X_sub = X[mask_idx]
            sims_sub = (X_sub @ q_vec.T).toarray().reshape(-1)  # cosine sim with L2-normalized tfidf
            best_local = int(np.argmax(sims_sub))
            best_idx = mask_idx[best_local]
            best_sim = float(sims_sub[best_local])
        else:
            sims = (X @ q_vec.T).toarray().reshape(-1)
            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])

        if best_sim < min_sim:
            return None, best_sim, kb["questions"][best_idx]

        return kb["answers"][best_idx], best_sim, kb["questions"][best_idx]

    def on_listen(self):
        if getattr(self, "_busy", False):
            if getattr(self, "_listening", False):
                self.status_var.set("Stopping recording...")
                self._stop_listen_event.set()
            return

        self._busy = True
        self._listening = True
        self._stop_listen_event = threading.Event()
        self.listen_btn.configure(text="Stop Listening")
        self.send_btn.configure(state="disabled")
        self.status_var.set("Listening... speak, then click Stop Listening when you finish.")

        def worker():
            try:
                seconds = int(self.mic_seconds.get())
                device_id = self._selected_device_id()
                text = voice_engine.listen_and_transcribe(
                    self.stt,
                    duration_s=seconds,
                    device=device_id,
                    stop_event=self._stop_listen_event,
                )
                self._ui(lambda: self._append_line(f"User (voice): {text or '[no speech detected]'}\n"))
                self._ui(lambda: self.classify_and_respond(text))
            except Exception as e:
                self._ui(lambda: messagebox.showerror("Listening Error", str(e)))
            finally:
                self._ui(self._set_ready)

        threading.Thread(target=worker, daemon=True).start()

    def on_send_prompt(self):
        text = (self.prompt_var.get() or "").strip()
        if not text:
            return
        self.prompt_var.set("")
        self._append_line(f"User (typed): {text}\n")
        self.classify_and_respond(text)

    def _append_line(self, s: str):
        self.text_area.insert(tk.END, s)
        self.text_area.see(tk.END)

    def _ui(self, fn):
        self.root.after(0, fn)

    def _set_ready(self):
        self.status_var.set("Ready")
        self.listen_btn.configure(state="normal", text="Start Listening")
        self.send_btn.configure(state="normal")
        self._busy = False
        self._listening = False
        
    def classify_and_respond(self, text):
        original_text = (text or "").strip()
        text = self._normalize_query(original_text)
        if not text:
            self.status_var.set("Ready (no input)")
            self.conf_bar["value"] = 0
            return
        
        # Predict
        self.status_var.set("Processing...")
        intent = self.clf.predict([text])[0]
        self.status_var.set(f"Identified Intent: {intent}")

        # Confidence (best-effort; LinearSVC has margins not probs)
        conf = None
        try:
            if hasattr(self.clf, "decision_function"):
                scores = np.asarray(self.clf.decision_function([text])).reshape(-1)
                margin = float(np.max(scores))
                conf = float(1.0 / (1.0 + np.exp(-margin)))  # squashed margin
        except Exception:
            conf = None

        if conf is not None:
            self.conf_bar["value"] = int(round(conf * 100))
        else:
            self.conf_bar["value"] = 0

        answer, sim, matched_q = self._retrieve_answer(text, intent)

        # Display Result
        header = f"Assistant: intent={intent}" + (f", confidence≈{conf:.2f}\n" if conf is not None else "\n")
        self._append_line(header)

        if answer:
            self._append_line(f"Answer: {answer}\n")
            self._append_line(f"(matched sim={sim:.2f})\n\n")
            voice_engine.speak_text_async(answer)
        else:
            self._append_line("Answer: माफ गर्नुहोस्, यस प्रश्नको ठ्याक्कै उत्तर डाटासेटमा भेटिएन।\n")
            if matched_q:
                self._append_line(f"(closest sim={sim:.2f}, closest question: {matched_q})\n\n")
            else:
                self._append_line("\n")
            voice_engine.speak_text_async("माफ गर्नुहोस्, यस प्रश्नको ठ्याक्कै उत्तर भेटिएन।")

if __name__ == "__main__":
    root = tk.Tk()
    app = NepaliAssistantApp(root)
    root.mainloop()
