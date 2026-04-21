"""
STEP 3 — Knowledge Base & Answer Retrieval
==========================================
Loads the 2,098 Q&A pairs from Excel and retrieves the best matching
answer for a given query using BM25 (primary) with TF-IDF cosine
similarity as a fallback.

What it produces
----------------
  outputs/kb_index.pkl   — serialised BM25 index + corpus (fast cold start)

Usage
-----
  # Build / rebuild the index
  python step3_retrieval.py --build

  # Query interactively
  python step3_retrieval.py --query "भूकम्प भनेको के हो?"

  # Query with intent filter
  python step3_retrieval.py --query "जोड कसरी गर्ने?" --intent math_question

  # Top-N results
  python step3_retrieval.py --query "पानी" --top-n 5
"""

import sys, json, pickle, logging, argparse
from pathlib import Path
from typing  import Optional, List

import numpy  as np
import pandas as pd
from rank_bm25           import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise        import cosine_similarity

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import QA_EXCEL, OUTPUTS_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

KB_INDEX_PATH = OUTPUTS_DIR / "kb_index.pkl"


# ─────────────────────────────────────────────────────────────────────────────
# 3.1  Load & clean corpus
# ─────────────────────────────────────────────────────────────────────────────

def load_corpus(excel_path: Path = QA_EXCEL) -> pd.DataFrame:
    """Load the Q&A Excel, return a clean DataFrame."""
    df = pd.read_excel(excel_path, dtype=str)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.dropna(subset=["question", "answer"])
    df["question"] = df["question"].str.strip()
    df["answer"]   = df["answer"].str.strip()
    df["intent"]   = df.get("intent", pd.Series(["unknown"] * len(df))).fillna("unknown").str.strip()
    log.info(f"Corpus loaded: {len(df)} Q&A pairs")
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 3.2  Tokeniser for Devanagari / Nepali
# ─────────────────────────────────────────────────────────────────────────────

def tokenise(text: str) -> List[str]:
    """
    Simple whitespace + punctuation tokeniser for Nepali (Devanagari).
    BM25Okapi expects a list of tokens per document.
    """
    import re
    # Remove punctuation (retain Devanagari, latin, digits)
    text = re.sub(r"[^\u0900-\u097F\w\s]", " ", text)
    return text.lower().split()


# ─────────────────────────────────────────────────────────────────────────────
# 3.3  Build index
# ─────────────────────────────────────────────────────────────────────────────

def build_index(df: pd.DataFrame) -> dict:
    """
    Build and return a dict containing:
      - bm25      : BM25Okapi index over tokenised questions
      - tfidf     : TfidfVectorizer fitted on questions (char n-grams)
      - tfidf_mat : sparse matrix (n_docs × n_features)
      - df        : the Q&A DataFrame (aligned with index rows)
    """
    log.info("Tokenising corpus for BM25 …")
    tokenised = [tokenise(q) for q in df["question"]]
    bm25      = BM25Okapi(tokenised)

    log.info("Fitting TF-IDF vectoriser (char-ngrams fallback) …")
    tfidf = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        min_df=1,
        max_features=50_000,
        sublinear_tf=True,
    )
    tfidf_mat = tfidf.fit_transform(df["question"])

    index = {
        "bm25":       bm25,
        "tokenised":  tokenised,
        "tfidf":      tfidf,
        "tfidf_mat":  tfidf_mat,
        "df":         df,
    }

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(KB_INDEX_PATH, "wb") as f:
        pickle.dump(index, f)
    log.info(f"Index saved → {KB_INDEX_PATH}")
    return index


def load_index() -> dict:
    """Load the serialised index from disk."""
    if not KB_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Index not found at {KB_INDEX_PATH}.\n"
            "Run:  python step3_retrieval.py --build"
        )
    with open(KB_INDEX_PATH, "rb") as f:
        index = pickle.load(f)
    log.info(f"Index loaded from {KB_INDEX_PATH}")
    return index


# ─────────────────────────────────────────────────────────────────────────────
# 3.4  Retrieve
# ─────────────────────────────────────────────────────────────────────────────

def retrieve(
    query:       str,
    index:       dict,
    top_n:       int           = 3,
    intent:      Optional[str] = None,
    bm25_weight: float         = 0.7,
) -> List[dict]:
    """
    Retrieve the top-N matching Q&A pairs for a query.

    Strategy
    --------
    1. Score all docs with BM25.
    2. Score all docs with TF-IDF cosine similarity.
    3. Combine: score = bm25_weight * bm25_norm + (1-bm25_weight) * tfidf_norm
    4. Optionally filter to a specific intent before returning.

    Returns
    -------
    List of dicts: { rank, question, answer, intent, score, bm25_score, tfidf_score }
    """
    df        = index["df"]
    bm25      = index["bm25"]
    tfidf     = index["tfidf"]
    tfidf_mat = index["tfidf_mat"]

    # BM25 scores
    bm25_scores = np.array(bm25.get_scores(tokenise(query)))

    # TF-IDF cosine scores
    q_vec       = tfidf.transform([query])
    tfidf_scores = cosine_similarity(q_vec, tfidf_mat).flatten()

    # Normalise each to [0, 1]
    def _norm(arr):
        mn, mx = arr.min(), arr.max()
        return (arr - mn) / (mx - mn + 1e-9)

    combined = bm25_weight * _norm(bm25_scores) + (1 - bm25_weight) * _norm(tfidf_scores)

    # Intent filter
    if intent:
        mask     = df["intent"].str.lower() == intent.lower()
        combined = combined * mask.values.astype(float)

    top_idx = np.argsort(combined)[::-1][:top_n]
    results = []
    for rank, idx in enumerate(top_idx, start=1):
        row = df.iloc[idx]
        results.append({
            "rank":        rank,
            "question":    row["question"],
            "answer":      row["answer"],
            "intent":      row["intent"],
            "score":       round(float(combined[idx]),    4),
            "bm25_score":  round(float(bm25_scores[idx]), 4),
            "tfidf_score": round(float(tfidf_scores[idx]),4),
        })
    return results


def get_best_answer(query: str, index: dict,
                    intent: Optional[str] = None) -> str:
    """Return the single best answer string for a query."""
    hits = retrieve(query, index, top_n=1, intent=intent)
    return hits[0]["answer"] if hits else "माफ गर्नुहोस्, यस प्रश्नको उत्तर फेला परेन।"


# ─────────────────────────────────────────────────────────────────────────────
# 3.5  CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Step 3 — Nepali Knowledge Base & Retrieval")
    parser.add_argument("--build",   action="store_true",
                        help="Build (or rebuild) the BM25 + TF-IDF index")
    parser.add_argument("--query",   type=str, default=None,
                        help="Nepali question to look up")
    parser.add_argument("--intent",  type=str, default=None,
                        help="Filter results to this intent (optional)")
    parser.add_argument("--top-n",   type=int, default=3,
                        help="Number of top results to return (default: 3)")
    args = parser.parse_args()

    if args.build:
        df = load_corpus()
        build_index(df)
        log.info("Index built successfully.")
        return

    index = load_index()

    if args.query:
        hits = retrieve(args.query, index, top_n=args.top_n, intent=args.intent)
        print(f"\n  Query : {args.query}")
        if args.intent:
            print(f"  Filter: intent = {args.intent}")
        print()
        for h in hits:
            print(f"  [{h['rank']}] score={h['score']:.4f}  intent={h['intent']}")
            print(f"      Q: {h['question']}")
            print(f"      A: {h['answer']}\n")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
