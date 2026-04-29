"""
app.py - Enhanced GUI application for the Nepali Voice Assistant.
"""

from __future__ import annotations

import pickle
import re
import sys
import threading
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import messagebox, ttk
from typing import List, Optional

import numpy as np

from config import CLEAN_XL, KB_INDEX_PATH, MODEL_PATH
import voice_engine


DEMO_QUESTIONS = [
    "विज्ञान सिक्ने पहिलो चरण के हो?",
    "बिरुवाका मुख्य भागहरू के के हुन्?",
    "हात धुनु किन आवश्यक छ?",
    "नेपालको राष्ट्रिय झण्डा कस्तो हुन्छ?",
    "व्यायाम गर्दा के फाइदा हुन्छ?",
    "रङ मिसाउँदा नयाँ रङ कसरी बन्छ?",
]


class Theme:
    BG = "#f6f7ef"
    CARD_BG = "#fffdf6"
    TEXT_DARK = "#213547"
    TEXT_MID = "#5f6f52"
    ACCENT = "#2f6f62"
    ACCENT_SOFT = "#d8efe5"
    BORDER = "#d9e3cf"
    LIST_BG = "#f9fff7"


INTENT_LABELS = {
    "science_question": "विज्ञान",
    "social_question": "सामाजिक",
    "nepali_language_question": "नेपाली",
    "health_question": "स्वास्थ्य",
    "physical_education_question": "शारीरिक शिक्षा",
    "arts_question": "कला",
}

NOISE_PHRASES = [
    "लेख्नुहोस्।",
    "लेख्नुहोस्",
    "बताउनुहोस्।",
    "बताउनुहोस्",
    "उदाहरण दिनुहोस्।",
    "उदाहरण दिनुहोस्",
    "उल्लेख गर्नुहोस्।",
    "उल्लेख गर्नुहोस्",
]

SPOKEN_FILLERS = [
    "ए",
    "हजुर",
    "ए हजुर",
    "ल",
    "अनि",
    "उम",
    "उम्",
    "अब",
    "कृपया",
    "सर",
    "मिस",
    "म्याम",
    "please",
    "teacher",
]

SPOKEN_ENDINGS = [
    "भन्नुहोस्",
    "बताइदिनुहोस्",
    "बताउनुहोला",
    "बताइदिनुहोला",
    "जान्न चाहन्छु",
]


@dataclass
class ClassificationResult:
    intent: str
    confidence: Optional[float] = None

    @property
    def intent_nepali(self) -> str:
        return INTENT_LABELS.get(self.intent, self.intent)


@dataclass
class RetrievalResult:
    answer: Optional[str]
    similarity: float
    matched_question: Optional[str]
    matched_intent: Optional[str] = None
    strategy: str = "global"

    @property
    def is_found(self) -> bool:
        return self.answer is not None


class KnowledgeBase:
    def __init__(self):
        self.vectorizer = None
        self.X = None
        self.roman_vectorizer = None
        self.roman_X = None
        self.questions: List[str] = []
        self.answers: List[str] = []
        self.intents: List[str] = []

    def load_or_build(self, kb_index_path: Path) -> "KnowledgeBase":
        if self._try_load_cache(kb_index_path):
            return self
        self._build_from_excel()
        self._save_cache(kb_index_path)
        return self

    def _try_load_cache(self, path: Path) -> bool:
        if not path.exists():
            return False
        try:
            with open(path, "rb") as f:
                kb = pickle.load(f)
            required = {"vectorizer", "X", "questions", "answers", "intents"}
            if required.issubset(kb.keys()) and len(kb["questions"]) > 0:
                self.vectorizer = kb["vectorizer"]
                self.X = kb["X"]
                self.roman_vectorizer = kb.get("roman_vectorizer")
                self.roman_X = kb.get("roman_X")
                self.questions = kb["questions"]
                self.answers = kb["answers"]
                self.intents = kb["intents"]
                return True
        except Exception:
            pass
        return False

    def _build_from_excel(self):
        import pandas as pd
        from sklearn.feature_extraction.text import TfidfVectorizer

        df = pd.read_excel(CLEAN_XL).dropna(subset=["question", "answer", "intent"])
        self.questions = df["question"].astype(str).tolist()
        self.answers = df["answer"].astype(str).tolist()
        self.intents = df["intent"].astype(str).tolist()

        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            sublinear_tf=True,
            min_df=2,
        )
        self.X = self.vectorizer.fit_transform(self.questions)

        roman_questions = [TextNormalizer.romanize_for_matching(q) for q in self.questions]
        self.roman_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            sublinear_tf=True,
            min_df=1,
        )
        self.roman_X = self.roman_vectorizer.fit_transform(roman_questions)

    def _save_cache(self, path: Path):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump(
                    {
                        "vectorizer": self.vectorizer,
                        "X": self.X,
                        "roman_vectorizer": self.roman_vectorizer,
                        "roman_X": self.roman_X,
                        "questions": self.questions,
                        "answers": self.answers,
                        "intents": self.intents,
                    },
                    f,
                )
        except Exception as e:
            print(f"Warning: Could not cache knowledge base: {e}")

    def _retrieve_single(
        self,
        query: str,
        predicted_intent: Optional[str] = None,
        restrict_to_intent: bool = True,
        use_romanized_index: bool = False,
    ) -> RetrievalResult:
        if use_romanized_index:
            vectorizer = self.roman_vectorizer
            matrix = self.roman_X
            query = TextNormalizer.romanize_for_matching(query)
        else:
            vectorizer = self.vectorizer
            matrix = self.X

        q_vec = vectorizer.transform([query])

        if restrict_to_intent and predicted_intent and self.intents:
            mask_idx = [i for i, it in enumerate(self.intents) if it == predicted_intent]
            if mask_idx:
                X_sub = matrix[mask_idx]
                sims = (X_sub @ q_vec.T).toarray().reshape(-1)
                best_local = int(np.argmax(sims))
                best_idx = mask_idx[best_local]
                best_sim = float(sims[best_local])
                return RetrievalResult(
                    self.answers[best_idx],
                    best_sim,
                    self.questions[best_idx],
                    self.intents[best_idx],
                    "romanized_intent_filtered" if use_romanized_index else "intent_filtered",
                )

        sims = (matrix @ q_vec.T).toarray().reshape(-1)
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        return RetrievalResult(
            self.answers[best_idx],
            best_sim,
            self.questions[best_idx],
            self.intents[best_idx],
            "romanized_global" if use_romanized_index else "global",
        )

    def retrieve(
        self,
        queries: List[str],
        predicted_intent: Optional[str],
        min_similarity: float = 0.25,
        allow_intent_filter: bool = True,
    ) -> RetrievalResult:
        cleaned_queries: List[str] = []
        for query in queries:
            query = str(query or "").strip()
            if query and query not in cleaned_queries:
                cleaned_queries.append(query)

        if not cleaned_queries:
            return RetrievalResult(None, 0.0, None)

        candidates: List[RetrievalResult] = []
        for query in cleaned_queries:
            if allow_intent_filter and predicted_intent:
                candidates.append(self._retrieve_single(query, predicted_intent, restrict_to_intent=True))
            candidates.append(self._retrieve_single(query, predicted_intent, restrict_to_intent=False))
            if self.roman_vectorizer is not None and self.roman_X is not None:
                if allow_intent_filter and predicted_intent:
                    candidates.append(
                        self._retrieve_single(
                            query,
                            predicted_intent,
                            restrict_to_intent=True,
                            use_romanized_index=True,
                        )
                    )
                candidates.append(
                    self._retrieve_single(
                        query,
                        predicted_intent,
                        restrict_to_intent=False,
                        use_romanized_index=True,
                    )
                )

        best = max(candidates, key=lambda item: item.similarity)
        if best.similarity < min_similarity:
            return RetrievalResult(None, best.similarity, best.matched_question, best.matched_intent, best.strategy)
        return best


class TextNormalizer:
    _DEVANAGARI_VOWELS = {
        "अ": "a", "आ": "aa", "इ": "i", "ई": "ii", "उ": "u", "ऊ": "uu",
        "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au", "ऋ": "ri",
    }
    _DEVANAGARI_CONSONANTS = {
        "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ng",
        "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "ny",
        "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
        "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
        "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
        "य": "y", "र": "r", "ल": "l", "व": "v", "श": "sh",
        "ष": "sh", "स": "s", "ह": "h", "ळ": "l",
    }
    _DEVANAGARI_MARKS = {
        "ा": "a", "ि": "i", "ी": "i", "ु": "u", "ू": "u",
        "े": "e", "ै": "ai", "ो": "o", "ौ": "au", "ृ": "ri",
        "ं": "n", "ँ": "n", "ः": "h", "्": "",
    }
    _DEVANAGARI_SPECIALS = {"क्ष": "ksh", "त्र": "tra", "ज्ञ": "gya"}

    @staticmethod
    def normalize(text: str) -> str:
        text = str(text or "").strip()
        text = re.sub(r"^[\d\u0966-\u096F]+[\.\)\s\u0964]+", "", text).strip()
        for phrase in NOISE_PHRASES:
            text = text.replace(phrase, "").strip()
        text = re.sub(r"\s+", " ", text)
        return text.strip(" ?।.!,")

    @staticmethod
    def spoken_variants(text: str) -> List[str]:
        base = TextNormalizer.normalize(text)
        if not base:
            return []

        variants = [base]
        stripped = base
        for filler in SPOKEN_FILLERS:
            stripped = re.sub(rf"(^|\s){re.escape(filler)}(\s|$)", " ", stripped, flags=re.IGNORECASE)
        for ending in SPOKEN_ENDINGS:
            stripped = re.sub(rf"(^|\s){re.escape(ending)}(\s|$)", " ", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s+", " ", stripped).strip(" ?।.!,")
        if stripped and stripped not in variants:
            variants.append(stripped)

        shortened = re.sub(r"\s+(हो|छ|रहेछ)$", "", stripped).strip()
        if shortened and shortened not in variants:
            variants.append(shortened)
        return variants

    @staticmethod
    def romanize_for_matching(text: str) -> str:
        text = TextNormalizer.normalize(text)
        out = []
        i = 0
        while i < len(text):
            pair = text[i : i + 2]
            if pair in TextNormalizer._DEVANAGARI_SPECIALS:
                out.append(TextNormalizer._DEVANAGARI_SPECIALS[pair])
                i += 2
                continue

            ch = text[i]
            if ch in TextNormalizer._DEVANAGARI_VOWELS:
                out.append(TextNormalizer._DEVANAGARI_VOWELS[ch])
                i += 1
                continue
            if ch in TextNormalizer._DEVANAGARI_CONSONANTS:
                base = TextNormalizer._DEVANAGARI_CONSONANTS[ch]
                nxt = text[i + 1] if i + 1 < len(text) else ""
                if nxt in TextNormalizer._DEVANAGARI_MARKS:
                    out.append(base + TextNormalizer._DEVANAGARI_MARKS[nxt])
                    i += 2
                elif nxt == "्":
                    out.append(base)
                    i += 2
                else:
                    out.append(base + "a")
                    i += 1
                continue
            if ch in TextNormalizer._DEVANAGARI_MARKS:
                out.append(TextNormalizer._DEVANAGARI_MARKS[ch])
                i += 1
                continue
            if ch.isascii() and ch.isalpha():
                out.append(ch.lower())
                i += 1
                continue

            out.append(" ")
            i += 1

        text = "".join(out).lower()
        text = text.replace("sikni", "sikhne")
        text = text.replace("sikne", "sikhne")
        text = text.replace("vahila", "pahilo")
        text = text.replace("pahilacaran", "pahilo charan")
        text = text.replace("kihu", "ke ho")
        text = text.replace("ki", "ke")
        text = text.replace("sarang", "charan")
        text = text.replace("biru", "biruwa")
        text = text.replace("biruwa aka", "biruwaka")
        text = text.replace("aga", "aka")
        text = text.replace("mukhe", "mukhya")
        text = text.replace("baagar", "bhagharu")
        text = text.replace("bagar", "bhagharu")
        text = text.replace("baakar", "bhagharu")
        text = text.replace("charankil", "charan ke")
        text = text.replace("ke kun", "ke ke hun")
        text = text.replace("kun hun", "ke ke hun")
        text = re.sub(r"\s+", " ", text).strip()
        return text


class Classifier:
    def __init__(self, model_path: Path):
        with open(model_path, "rb") as f:
            self._clf = pickle.load(f)

    def predict(self, text: str) -> ClassificationResult:
        intent = self._clf.predict([text])[0]
        confidence = self._estimate_confidence(text)
        return ClassificationResult(intent, confidence)

    def _estimate_confidence(self, text: str) -> Optional[float]:
        try:
            if hasattr(self._clf, "decision_function"):
                scores = np.asarray(self._clf.decision_function([text])).reshape(-1)
                margin = float(np.max(scores))
                return float(1.0 / (1.0 + np.exp(-margin)))
        except Exception:
            pass
        return None


class VoiceService:
    def __init__(self):
        self.stt_model = voice_engine.load_stt_model()
        self.audio_devices = voice_engine.list_input_devices()
        self.default_device_id = voice_engine.choose_default_input_device()

    def listen(self, duration: int, device_id, stop_event):
        return voice_engine.listen_and_transcribe(
            self.stt_model,
            duration_s=duration,
            device=device_id
        )

    def speak(self, text: str):
        voice_engine.speak_text_async(text)

    @property
    def has_nepali_tts(self):
        return True

    def get_device_list(self):
        return [f"{d['id']}: {d['name']}" for d in self.audio_devices]

    def find_default_device_label(self):
        for d in self.audio_devices:
            if d["id"] == self.default_device_id:
                return f"{d['id']}: {d['name']}"
        return None

class NepaliAssistantApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self._busy = False
        self._listening = False
        self._stop_listen_event: Optional[threading.Event] = None

        self._setup_window()
        self._load_components()
        self._build_ui()
        self._initialize_voice_system()
        self._configure_audio_devices()
        self._show_tts_status()

    def _setup_window(self):
        self.root.title("नेपाली शैक्षिक विद्यार्थी सहायक")
        self.root.geometry("840x640")
        self.root.minsize(800, 580)
        self.root.configure(bg=Theme.BG)

    def _initialize_voice_system(self):
        """Initialize voice system with noise calibration."""
        self.status_var.set("आवाज प्रणाली सेटअप हुँदैछ (कृपया २ सेकेन्ड शान्त रहनुहोस्)...")
        self.root.update()
        try:
            self.voice.calibrate(duration=2.0)
            self.status_var.set("तयार")
        except Exception as e:
            print(f"Calibration warning: {e}")
            self.status_var.set("तयार (क्यालिब्रेसन त्रुटि)")

    def _load_components(self):
        try:
            self.classifier = Classifier(MODEL_PATH)
            self.voice = VoiceService()
            self.kb = KnowledgeBase().load_or_build(KB_INDEX_PATH)
            
        except FileNotFoundError as e:
            messagebox.showerror("त्रुटि - फाइल भेटिएन", f"आवश्यक फाइल भेटिएन:\n{e}")
            sys.exit(1)
        except Exception as e:
            messagebox.showerror("त्रुटि", f"मोडेल लोड गर्न सकिएन:\n{e}")
            sys.exit(1)

    def _build_ui(self):
        self._setup_styles()
        self._create_layout()

    def _setup_styles(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure("App.TFrame", background=Theme.BG)
        style.configure("Card.TFrame", background=Theme.CARD_BG, relief="flat")
        style.configure("HeaderTitle.TLabel", background=Theme.BG, foreground=Theme.TEXT_DARK, font=("Segoe UI", 20, "bold"))
        style.configure("HeaderSub.TLabel", background=Theme.BG, foreground=Theme.TEXT_MID, font=("Segoe UI", 10))
        style.configure("Status.TLabel", background=Theme.BG, foreground=Theme.ACCENT, font=("Segoe UI", 10, "bold"))
        style.configure("Section.TLabelframe", background=Theme.CARD_BG, borderwidth=1, relief="solid")
        style.configure("Section.TLabelframe.Label", background=Theme.CARD_BG, foreground=Theme.TEXT_DARK, font=("Segoe UI", 10, "bold"))
        style.configure("Field.TLabel", background=Theme.CARD_BG, foreground=Theme.TEXT_MID, font=("Segoe UI", 10))
        style.configure("Primary.TButton", font=("Segoe UI", 10, "bold"), padding=(12, 8), foreground="#ffffff", background=Theme.ACCENT)
        style.configure("Secondary.TButton", font=("Segoe UI", 10), padding=(12, 8), foreground=Theme.TEXT_DARK, background=Theme.ACCENT_SOFT)
        style.map("Primary.TButton", background=[("active", "#275d52"), ("pressed", "#214e45")])
        style.map("Secondary.TButton", background=[("active", "#c6e7d8"), ("pressed", "#b7dbc9")])

    def _create_layout(self):
        header = ttk.Frame(self.root, padding=(18, 16), style="App.TFrame")
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)

        ttk.Label(header, text="नेपाली विद्यार्थी भ्वाइस सहायक", style="HeaderTitle.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(header, text="प्रश्न टाइप गर्नुहोस् वा माइक्रोफोन प्रयोग गरेर सोध्नुहोस्।", style="HeaderSub.TLabel").grid(row=1, column=0, sticky="w", pady=(4, 0))

        self.status_var = tk.StringVar(value="तयार")
        ttk.Label(header, textvariable=self.status_var, style="Status.TLabel").grid(row=0, column=1, rowspan=2, sticky="e")

        body = ttk.Frame(self.root, padding=(18, 10), style="App.TFrame")
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=0)
        body.rowconfigure(1, weight=1)

        self._create_input_panel(body)
        self._create_demo_panel(body)
        self._create_output_panel(body)

        self.conf_bar = ttk.Progressbar(body, mode="determinate", maximum=100)
        self.conf_bar.grid(row=2, column=0, sticky="ew", pady=(10, 0))

    def _create_input_panel(self, parent):
        panel = ttk.Labelframe(parent, text="प्रश्न प्रविष्टि", padding=(14, 12), style="Section.TLabelframe")
        panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        panel.columnconfigure(1, weight=1)

        ttk.Label(panel, text="प्रश्न", style="Field.TLabel").grid(row=0, column=0, sticky="w")
        self.prompt_var = tk.StringVar()
        self.prompt_entry = ttk.Entry(panel, textvariable=self.prompt_var)
        self.prompt_entry.grid(row=0, column=1, sticky="ew", padx=(8, 8))
        self.prompt_entry.bind("<Return>", lambda _e: self.on_send_prompt())

        self.send_btn = ttk.Button(panel, text="पठाउनुहोस्", command=self.on_send_prompt, style="Primary.TButton")
        self.send_btn.grid(row=0, column=2, sticky="e")

        ttk.Label(panel, text="सुन्ने समय सीमा", style="Field.TLabel").grid(row=1, column=0, sticky="w", pady=(12, 0))
        self.mic_seconds = tk.IntVar(value=10)
        self.mic_spin = ttk.Spinbox(panel, from_=2, to=12, textvariable=self.mic_seconds, width=6)
        self.mic_spin.grid(row=1, column=1, sticky="w", padx=(8, 0), pady=(12, 0))

        self.listen_btn = ttk.Button(panel, text="सुन्न सुरु गर्नुहोस्", command=self.on_listen, style="Secondary.TButton")
        self.listen_btn.grid(row=1, column=2, sticky="e", pady=(12, 0))

        ttk.Label(panel, text="आवाज इनपुट उपकरण", style="Field.TLabel").grid(row=2, column=0, sticky="w", pady=(12, 0))
        self.mic_device_var = tk.StringVar()
        self.mic_device_combo = ttk.Combobox(panel, textvariable=self.mic_device_var, state="readonly", width=48)
        self.mic_device_combo.grid(row=2, column=1, columnspan=2, sticky="ew", padx=(8, 0), pady=(10, 0))

    def _create_demo_panel(self, parent):
        panel = ttk.Labelframe(parent, text="छिटो परीक्षणका प्रश्नहरू", padding=(14, 12), style="Section.TLabelframe")
        panel.grid(row=0, column=1, rowspan=2, sticky="nsew")
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(1, weight=1)

        ttk.Label(panel, text="तलका नमुना प्रश्नहरू छान्दा सिधै परीक्षण गर्न सकिन्छ।", style="Field.TLabel", wraplength=260, justify="left").grid(row=0, column=0, sticky="w")

        self.demo_listbox = tk.Listbox(
            panel,
            font=("Segoe UI", 10),
            activestyle="none",
            relief="flat",
            highlightthickness=1,
            highlightbackground=Theme.BORDER,
            selectbackground=Theme.ACCENT_SOFT,
            selectforeground=Theme.TEXT_DARK,
            background=Theme.LIST_BG,
            fg=Theme.TEXT_DARK,
        )
        self.demo_listbox.grid(row=1, column=0, sticky="nsew", pady=(10, 10))
        for question in DEMO_QUESTIONS:
            self.demo_listbox.insert(tk.END, question)
        self.demo_listbox.bind("<Double-Button-1>", lambda _e: self.ask_demo_question())

        actions = ttk.Frame(panel, style="Card.TFrame")
        actions.grid(row=2, column=0, sticky="ew")
        actions.columnconfigure(0, weight=1)
        actions.columnconfigure(1, weight=1)

        ttk.Button(actions, text="प्रश्न राख्नुहोस्", command=self.use_demo_question, style="Secondary.TButton").grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ttk.Button(actions, text="सिधै सोध्नुहोस्", command=self.ask_demo_question, style="Primary.TButton").grid(row=0, column=1, sticky="ew", padx=(6, 0))

    def _create_output_panel(self, parent):
        panel = ttk.Labelframe(parent, text="संवाद / उत्तर", padding=(14, 12), style="Section.TLabelframe")
        panel.grid(row=1, column=0, sticky="nsew", pady=(12, 0), padx=(0, 10))
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(0, weight=1)

        self.text_area = tk.Text(
            panel,
            wrap="word",
            font=("Segoe UI", 10),
            relief="flat",
            bg=Theme.LIST_BG,
            fg=Theme.TEXT_DARK,
            padx=10,
            pady=10,
            insertbackground=Theme.TEXT_DARK,
        )
        self.text_area.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(panel, orient="vertical", command=self.text_area.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.text_area.configure(yscrollcommand=scroll.set)
        self.prompt_entry.focus_set()

    def _configure_audio_devices(self):
        devices = self.voice.get_device_list()
        default_label = self.voice.find_default_device_label()
        self.mic_device_combo["values"] = devices
        if default_label:
            self.mic_device_var.set(default_label)
        elif devices:
            self.mic_device_combo.current(0)

    def _show_tts_status(self):
        if not self.voice.has_nepali_tts:
            self._append_line(
                "सूचना: नेपाली आवाजमा पढ्ने सुविधा अहिले उपलब्ध छैन, किनकि eSpeak/eSpeak-NG इन्स्टल गरिएको छैन। "
                "अहिलेको वैकल्पिक आवाजले नेपाली राम्रोसँग नबोल्न सक्छ।\n\n"
            )

    def _selected_demo_question(self) -> Optional[str]:
        selection = self.demo_listbox.curselection()
        if not selection:
            return None
        return self.demo_listbox.get(selection[0])

    def use_demo_question(self):
        question = self._selected_demo_question()
        if not question:
            messagebox.showinfo("सूचना", "कृपया एउटा नमुना प्रश्न छान्नुहोस्।")
            return
        self.prompt_var.set(question)
        self.prompt_entry.focus_set()

    def ask_demo_question(self):
        question = self._selected_demo_question()
        if not question:
            messagebox.showinfo("सूचना", "कृपया एउटा नमुना प्रश्न छान्नुहोस्।")
            return
        self.prompt_var.set(question)
        self.on_send_prompt()

    def _selected_device_id(self) -> Optional[int]:
        value = (self.mic_device_var.get() or "").strip()
        if not value:
            return None
        try:
            return int(value.split(":", 1)[0])
        except (ValueError, IndexError):
            return None

    def on_listen(self):
        if self._busy:
            if self._listening and self._stop_listen_event is not None:
                self.status_var.set("रेकर्ड रोकिँदैछ...")
                self._stop_listen_event.set()
            return

        self._busy = True
        self._listening = True
        self._stop_listen_event = threading.Event()
        self.listen_btn.configure(text="सुन्न रोक्नुहोस्")
        self.send_btn.configure(state="disabled")
        self.status_var.set("सुनिँदैछ... बोल्नुहोस्, सकिएपछि रोक्नुहोस्।")

        def worker():
            try:
                duration = int(self.mic_seconds.get())
                device_id = self._selected_device_id()
                text = self.voice.listen(duration, device_id, self._stop_listen_event)
                display_text = text or "[कुनै आवाज भेटिएन]"
                self._ui(lambda: self._append_line(f"प्रयोगकर्ता (आवाज): {display_text}\n"))
                self._ui(lambda: self._process_query(text, source="voice"))
            except Exception as e:
                self._ui(lambda: messagebox.showerror("सुन्ने त्रुटि", str(e)))
            finally:
                self._ui(self._set_ready)

        threading.Thread(target=worker, daemon=True).start()

    def on_send_prompt(self):
        text = (self.prompt_var.get() or "").strip()
        if not text:
            return
        self.prompt_var.set("")
        self._append_line(f"प्रयोगकर्ता (टाइप): {text}\n")
        self._process_query(text, source="typed")

    def _process_query(self, text: Optional[str], source: str = "typed"):
        original_text = (text or "").strip()
        normalized = TextNormalizer.normalize(original_text)
        if not normalized:
            self.status_var.set("तयार (इनपुट छैन)")
            self.conf_bar["value"] = 0
            return

        self.status_var.set("प्रशोधन हुँदैछ...")
        query_variants = TextNormalizer.spoken_variants(original_text) if source == "voice" else [normalized]

        if source == "voice":
            direct_retrieval = self.kb.retrieve(
                query_variants,
                predicted_intent=None,
                min_similarity=0.26,
                allow_intent_filter=False,
            )
            if direct_retrieval.is_found:
                result = ClassificationResult(
                    direct_retrieval.matched_intent or "science_question",
                    None,
                )
                self.status_var.set(f"पहिचान गरिएको कक्षा: {result.intent_nepali}")
                self.conf_bar["value"] = min(100, int(round(direct_retrieval.similarity * 100)))
                self._display_result(result, direct_retrieval, source)
                return

        result = self.classifier.predict(normalized)
        self.status_var.set(f"पहिचान गरिएको कक्षा: {result.intent_nepali}")
        self.conf_bar["value"] = int(round(result.confidence * 100)) if result.confidence is not None else 0

        retrieval_threshold = 0.20 if source == "voice" else 0.25
        retrieval = self.kb.retrieve(
            query_variants,
            result.intent,
            min_similarity=retrieval_threshold,
            allow_intent_filter=True,
        )
        self._display_result(result, retrieval, source)

    def _display_result(self, classification: ClassificationResult, retrieval: RetrievalResult, source: str):
        header = f"सहायक: कक्षा={classification.intent_nepali}"
        if classification.confidence is not None:
            header += f", विश्वसनीयता≈{classification.confidence:.2f}"
        self._append_line(header + "\n")

        if retrieval.is_found:
            self._append_line(f"उत्तर: {retrieval.answer}\n")
            extra = f"(मिलान स्कोर={retrieval.similarity:.2f}"
            if source == "voice":
                extra += f", खोज तरिका={retrieval.strategy}"
            extra += ")\n\n"
            self._append_line(extra)
            self.voice.speak(retrieval.answer)
        else:
            self._append_line("उत्तर: माफ गर्नुहोस्, यस प्रश्नको ठ्याक्कै उत्तर डाटासेटमा भेटिएन।\n")
            if retrieval.matched_question:
                self._append_line(
                    f"(सबैभन्दा नजिकको मिलान={retrieval.similarity:.2f}, नजिकको प्रश्न: {retrieval.matched_question})\n\n"
                )
            else:
                self._append_line("\n")
            self.voice.speak("माफ गर्नुहोस्, यस प्रश्नको ठ्याक्कै उत्तर भेटिएन।")

    def _append_line(self, text: str):
        self.text_area.insert(tk.END, text)
        self.text_area.see(tk.END)

    def _ui(self, fn):
        self.root.after(0, fn)

    def _set_ready(self):
        self.status_var.set("तयार")
        self.listen_btn.configure(state="normal", text="सुन्न सुरु गर्नुहोस्")
        self.send_btn.configure(state="normal")
        self._busy = False
        self._listening = False


if __name__ == "__main__":
    root = tk.Tk()
    app = NepaliAssistantApp(root)
    root.mainloop()