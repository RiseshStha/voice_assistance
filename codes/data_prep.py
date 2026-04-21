"""
data_prep.py — Part 1: Data Preprocessing and EDA
=================================================
Technical contribution: Implements automated Devanagari text normalization
and statistical visualization for class distribution.
"""
import pandas as pd
import re, sys, io
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Setup encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Project Path Settings
from config import ROOT, QA_EXCEL as CLEAN_DATA
RAW_DATA = ROOT / "data" / "data_for_training" / "question_answer_final.xlsx"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

def normalize_nepali(text):
    """Strips serial numbers and instruction suffixes."""
    text = str(text).strip()
    # Remove serial numbers
    text = re.sub(r"^[\d\u0966-\u096F]+[\.\)\s\u0964]+", "", text).strip()
    # Remove instructional noise
    noise = ["लेख्नुहोस्।", "लेख्नुहोस्", "बताउनुहोस्", "उदाहरण दिनुहोस्"]
    for word in noise:
        text = text.replace(word, "").strip()
    return text

def main():
    print("Loading raw textbook data...")
    df = pd.read_excel(RAW_DATA)
    df.columns = [c.strip().lower() for c in df.columns]
    
    # Preprocessing
    df['question'] = df['question'].apply(normalize_nepali)
    df = df.dropna(subset=['question', 'intent'])
    df = df[df['question'].str.len() > 3] # Filter outliers

    # EDA Visualizations
    plt.figure(figsize=(10, 6))
    counts = df['intent'].value_counts()
    sns.barplot(x=counts.index, y=counts.values, palette="rocket")
    plt.title("Educational Intent Distribution")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "eda_intent_dist.png")
    
    print(f"Data cleaned. {len(df)} samples remaining.")
    print("EDA Plot saved to outputs/eda_intent_dist.png")
    
    # Save Cleaned Data
    df.to_excel(CLEAN_DATA, index=False)
    print(f"Cleaned dataset saved: {CLEAN_DATA.name}")

if __name__ == "__main__":
    main()
