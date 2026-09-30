"""
data_loader.py
Load and preprocess articles from Article-Bias-Prediction dataset
(Baly et al., EMNLP 2020), filtered to a single topic.
Data path: Article-Bias-Prediction/data/jsons/
"""

import json, os, re
import pandas as pd

DATA_DIR    = "Article-Bias-Prediction/data/jsons"
TOPIC       = "economy_and_jobs"
MAX_WORDS   = 600
RANDOM_SEED = 42
MIN_WORDS   = 100    # filter out articles too short to summarize


def truncate_words(text: str, max_words: int = MAX_WORDS) -> str:
    return " ".join(text.split()[:max_words])


def load_articles(topic: str       = TOPIC,
                  n_samples: int   = None,
                  balanced: bool   = True) -> pd.DataFrame:
    """
    Load articles from Article-Bias-Prediction JSON files.

    Args:
        topic     : topic filter (e.g. 'economy_and_jobs')
        n_samples : total sample size; if balanced, split evenly across 3 classes
        balanced  : if True, sample equal numbers of left/center/right

    Returns:
        DataFrame with columns:
            id, title, bias, bias_text, source, topic, date, url, input_text
    """
    print(f"Loading articles | topic={topic} ...")
    records = []
    for fname in os.listdir(DATA_DIR):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(DATA_DIR, fname)) as f:
            d = json.load(f)
        if d.get("topic") != topic:
            continue
        content = d.get("content_original", "").strip()
        if len(content.split()) < MIN_WORDS:
            continue
        records.append({
            "raw_id"    : d.get("ID"),
            "title"     : d.get("title", ""),
            "bias"      : d.get("bias"),        # 0=left, 1=center, 2=right
            "bias_text" : d.get("bias_text"),   # 'left' / 'center' / 'right'
            "source"    : d.get("source", ""),
            "topic"     : d.get("topic", ""),
            "date"      : d.get("date", ""),
            "url"       : d.get("url", ""),
            "content"   : content,
        })

    df = pd.DataFrame(records)
    print(f"  Found {len(df)} articles "
          f"({(df.bias_text=='left').sum()} left / "
          f"{(df.bias_text=='center').sum()} center / "
          f"{(df.bias_text=='right').sum()} right)")

    if n_samples is not None:
        if balanced:
            per_class = n_samples // 3
            parts = []
            for label in ["left", "center", "right"]:
                subset = df[df.bias_text == label]
                parts.append(subset.sample(
                    n=min(per_class, len(subset)),
                    random_state=RANDOM_SEED
                ))
            df = pd.concat(parts).sample(frac=1, random_state=RANDOM_SEED)
        else:
            df = df.sample(n=min(n_samples, len(df)), random_state=RANDOM_SEED)
        df = df.reset_index(drop=True)
        print(f"  Sampled {len(df)} articles "
              f"({(df.bias_text=='left').sum()} left / "
              f"{(df.bias_text=='center').sum()} center / "
              f"{(df.bias_text=='right').sum()} right)")

    df.insert(0, "id", df.index)
    df["input_text"] = df["content"].apply(truncate_words)

    print(f"  Done. Shape: {df.shape}")
    return df


def inspect(df: pd.DataFrame, n: int = 3) -> None:
    print(f"\nTotal  : {len(df)}")
    print(f"Labels : left={( df.bias_text=='left').sum()}  "
          f"center={(df.bias_text=='center').sum()}  "
          f"right={(df.bias_text=='right').sum()}")
    print(f"Avg words : {df['input_text'].apply(lambda x: len(x.split())).mean():.0f}")
    for _, row in df.head(n).iterrows():
        print(f"\n[{row['id']}] [{row['bias_text']:^6}] {row['title'][:70]}")
        print(f"  source : {row['source']}")
        print(f"  preview: {row['input_text'][:120]}...")


if __name__ == "__main__":
    df = load_articles(n_samples=30, balanced=True)
    inspect(df)
    df.to_parquet("articles.parquet", index=False)
    print("\nSaved to articles.parquet")
