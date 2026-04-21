"""
build_kb.py
===========
Build a lightweight retrieval index from the cleaned Nepali Q/A dataset.

At runtime, the GUI can retrieve the most similar known question and return its answer.
This turns the prototype from "intent-only" into an assistant that can answer.
"""

import pickle
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

from config import CLEAN_XL, KB_INDEX_PATH


def main():
    df = pd.read_excel(CLEAN_XL)
    df = df.dropna(subset=["question", "answer", "intent"])

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

    kb = {
        "vectorizer": vectorizer,
        "X": X,
        "questions": questions,
        "answers": answers,
        "intents": intents,
    }

    with open(KB_INDEX_PATH, "wb") as f:
        pickle.dump(kb, f)

    print(f"KB index saved: {KB_INDEX_PATH}")
    print(f"Items indexed: {len(questions)}")


if __name__ == "__main__":
    main()

