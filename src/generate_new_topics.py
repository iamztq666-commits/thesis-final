"""
generate_new_topics.py
Generate summaries for environment, education, gun_control_and_gun_rights.
150 articles each (50 L / 50 C / 50 R), Qwen-Plus.
"""
import os, re, time, json, warnings
warnings.filterwarnings("ignore")

import pandas as pd
from openai import OpenAI
from retrieval import build_indices
from prompts  import build_all_prompts

API_KEY  = os.environ["DASHSCOPE_API_KEY"]
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL    = "qwen-plus"

N_PER_TOPIC    = 150
MAX_TOKENS     = 150
MAX_TOKENS_COT = 400
TEMPERATURE    = 0.3
SLEEP          = 0.25
SETTINGS       = ["s1","s2","s3a","s3b","s3c","s4","s5_cot","s6"]
STRUCTURED     = {"s5_cot"}
RANDOM_SEED    = 77
DATA_DIR       = "Article-Bias-Prediction/data/jsons"

TOPICS = ["environment", "education", "gun_control_and_gun_rights"]

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def load_topic(topic, n=N_PER_TOPIC):
    records = []
    for fname in os.listdir(DATA_DIR):
        if not fname.endswith(".json"): continue
        d = json.load(open(os.path.join(DATA_DIR, fname)))
        if d.get("topic") != topic: continue
        if len(d.get("content_original","").split()) < 100: continue
        records.append({
            "raw_id":    d.get("ID"),
            "title":     d.get("title",""),
            "bias_text": d.get("bias_text"),
            "source":    d.get("source",""),
            "topic":     topic,
            "date":      d.get("date",""),
            "url":       d.get("url",""),
            "content":   d.get("content_original","").strip(),
        })
    df  = pd.DataFrame(records)
    per = n // 3
    parts = []
    for lbl in ["left","center","right"]:
        sub = df[df.bias_text == lbl]
        parts.append(sub.sample(n=min(per, len(sub)), random_state=RANDOM_SEED))
    df = pd.concat(parts).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    df.insert(0, "id", df.index)
    df["input_text"] = df["content"].apply(lambda t: " ".join(t.split()[:600]))
    print(f"  {topic}: {len(df)} articles "
          f"(L={(df.bias_text=='left').sum()} "
          f"C={(df.bias_text=='center').sum()} "
          f"R={(df.bias_text=='right').sum()})")
    return df

def call_api(prompt, setting):
    tokens = MAX_TOKENS_COT if setting in STRUCTURED else MAX_TOKENS
    for attempt in range(3):
        try:
            r = client.chat.completions.create(
                model=MODEL,
                messages=[{"role":"user","content":prompt}],
                max_tokens=tokens, temperature=TEMPERATURE, timeout=30,
            )
            return r.choices[0].message.content.strip()
        except Exception as e:
            print(f"    attempt {attempt+1} failed: {e}")
            time.sleep(2 ** attempt)
    return ""

def clean(text, setting):
    if setting in STRUCTURED and "Summary:" in text:
        text = text.split("Summary:", 1)[1].strip()
    sents = re.split(r'(?<=[.!?])\s+', text)
    done  = [s for s in sents if s.endswith((".", "!", "?"))]
    return " ".join(done) if done else text

def run(df, topic):
    indices = build_indices()
    records = []
    total   = len(df)
    for i, row in df.iterrows():
        print(f"  [{i+1}/{total}] {row['title'][:60]}...")
        prompts = build_all_prompts(row["input_text"], row["title"],
                                    indices, row["bias_text"])
        rec = {k: row[k] for k in ["id","title","bias_text","input_text",
                                    "source","topic","date","url"]}
        for s in SETTINGS:
            try:
                raw     = call_api(prompts[s], s)
                summary = clean(raw, s)
                rec[f"summary_{s}"] = summary
                rec[f"raw_{s}"]     = raw
                time.sleep(SLEEP)
            except Exception as e:
                print(f"    ERROR {s}: {e}")
                rec[f"summary_{s}"] = ""
                rec[f"raw_{s}"]     = ""
        records.append(rec)
    out  = pd.DataFrame(records)
    path = f"summaries_{topic}.parquet"
    out.to_parquet(path, index=False)
    out.to_csv(path.replace(".parquet",".csv"), index=False)
    print(f"  Saved {len(out)} rows → {path}\n")
    return out

for topic in TOPICS:
    print(f"\n{'='*60}")
    print(f"Topic: {topic}")
    print('='*60)
    df = load_topic(topic)
    run(df, topic)

print("All done.")
