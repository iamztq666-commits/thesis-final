"""
retrieval.py
Build BM25 indices for three political leanings from bypublisher:
  bias=0: right / bias=2: neutral / bias=4: left
Used for Settings 3a, 3b, 3c.
"""

import os, re, pickle
import nltk
from rank_bm25 import BM25Okapi
from nltk.tokenize import sent_tokenize
from bs4 import BeautifulSoup

for _pkg in ("punkt", "punkt_tab"):
    try:
        nltk.data.find(f"tokenizers/{_pkg}")
    except LookupError:
        nltk.download(_pkg, quiet=True)

DATASET_NAME   = "SemEvalWorkshop/hyperpartisan_news_detection"
MAX_ARTICLES   = 3000
MIN_SENT_WORDS = 10
MAX_SENT_WORDS = 60
TOP_K          = 3
INDEX_CACHE    = "bm25_indices.pkl"
BIAS_LABELS    = {0: "right", 2: "neutral", 4: "left"}


def clean_html(text: str) -> str:
    text = BeautifulSoup(text, "html.parser").get_text(separator=" ")
    return re.sub(r"\s+", " ", text).strip()


def build_indices(max_articles: int = MAX_ARTICLES,
                  cache_path: str   = INDEX_CACHE) -> dict:
    """
    Build or load BM25 indices for right / neutral / left.
    Returns:
        {"right": {"bm25":..., "sentences":[...]},
         "neutral": {...}, "left": {...}}
    """
    if os.path.exists(cache_path):
        print(f"Loading BM25 indices from cache ({cache_path})...")
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        for leaning, d in data.items():
            print(f"  {leaning:8s}: {len(d['sentences'])} sentences")
        return data

    print("Building BM25 indices (right / neutral / left) — takes ~5 min...")
    from datasets import load_dataset
    ds = load_dataset(DATASET_NAME, "bypublisher", trust_remote_code=True)

    buckets       = {v: [] for v in BIAS_LABELS}
    article_count = {v: 0  for v in BIAS_LABELS}

    for row in ds["train"]:
        bv = row["bias"]
        if bv not in BIAS_LABELS or article_count[bv] >= max_articles:
            continue
        sents = sent_tokenize(clean_html(row["text"]))
        buckets[bv].extend(
            s for s in sents if MIN_SENT_WORDS <= len(s.split()) <= MAX_SENT_WORDS
        )
        article_count[bv] += 1
        if all(c >= max_articles for c in article_count.values()):
            break

    indices = {}
    for bv, label in BIAS_LABELS.items():
        sents = buckets[bv]
        print(f"  Fitting BM25 for {label}: {len(sents)} sentences...")
        indices[label] = {"bm25": BM25Okapi([s.lower().split() for s in sents]),
                          "sentences": sents}

    with open(cache_path, "wb") as f:
        pickle.dump(indices, f)
    print(f"  Cached to {cache_path}")
    return indices


def retrieve(query: str, indices: dict,
             leaning: str = "neutral", top_k: int = TOP_K) -> list[str]:
    """Retrieve top-k sentences from a specific leaning index."""
    bm25      = indices[leaning]["bm25"]
    sentences = indices[leaning]["sentences"]
    scores    = bm25.get_scores(query.lower().split())
    top_idx   = scores.argsort()[-top_k:][::-1]
    return [sentences[i] for i in top_idx]


def format_bullets(sentences: list[str], header: str = None) -> str:
    prefix  = f"[{header}]\n" if header else ""
    bullets = "\n".join(f"- {s}" for s in sentences)
    return prefix + bullets


if __name__ == "__main__":
    indices = build_indices()
    query   = "Democrats spending bill Congress economy"
    for leaning in ["neutral", "left", "right"]:
        print(f"\n── {leaning.upper()} ──")
        for s in retrieve(query, indices, leaning=leaning):
            print(f"  {s}")
