"""
generate_environment_revised.py
Generate revised-prompt summaries for the same 150 environment articles
already in summaries_environment.parquet (50L / 50C / 50R, seed=77).

Uses prompts_revised.py — 8 settings: s1, s2_r, s3a_r, s3b_r, s3c_r, s4_r, s5_r, s6_r
Saves → summaries_environment_revised.parquet
"""
import os, re, time, warnings
warnings.filterwarnings("ignore")

import pandas as pd
from openai import OpenAI
from prompts_revised import build_all_prompts_r, SETTINGS_R

API_KEY      = os.environ["DASHSCOPE_API_KEY"]
BASE_URL     = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL        = "qwen-plus"
MAX_TOKENS   = 150
MAX_TOKENS_L = 600   # structured settings: s4_r, s5_r, s5_adv (3-step needs more)
TEMPERATURE  = 0.3
SLEEP        = 0.25
STRUCTURED   = {"s4_r", "s5_r", "s5_adv"}   # all emit "Summary:" label
IN_FILE      = "summaries_environment.parquet"
OUT_FILE     = "summaries_environment_revised.parquet"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ── load existing 150 articles ──────────────────────────────────────────────
src = pd.read_parquet(IN_FILE)[["id","title","bias_text","input_text","source","topic","date","url"]]
print(f"Loaded {len(src)} articles  "
      f"L={(src.bias_text=='left').sum()} "
      f"C={(src.bias_text=='center').sum()} "
      f"R={(src.bias_text=='right').sum()}")

# ── API helpers ──────────────────────────────────────────────────────────────
def call_api(prompt: str, setting: str) -> str:
    tokens = MAX_TOKENS_L if setting in STRUCTURED else MAX_TOKENS
    for attempt in range(3):
        try:
            r = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=tokens, temperature=TEMPERATURE, timeout=30,
            )
            return r.choices[0].message.content.strip()
        except Exception as e:
            print(f"    attempt {attempt+1} failed: {e}")
            time.sleep(2 ** attempt)
    return ""

def clean(text: str, setting: str) -> str:
    if setting in STRUCTURED and "Summary:" in text:
        text = text.split("Summary:", 1)[1].strip()
    sents = re.split(r'(?<=[.!?])\s+', text)
    done  = [s for s in sents if s.endswith((".", "!", "?"))]
    return " ".join(done) if done else text

# ── generation loop ──────────────────────────────────────────────────────────
records = []
total   = len(src)

for i, row in src.iterrows():
    print(f"[{i+1}/{total}] {row['title'][:65]}...")
    prompts = build_all_prompts_r(row["input_text"])
    rec = {k: row[k] for k in ["id","title","bias_text","input_text","source","topic","date","url"]}
    for s in SETTINGS_R:
        try:
            raw     = call_api(prompts[s], s)
            summary = clean(raw, s)
            rec[f"summary_{s}"] = summary
            rec[f"raw_{s}"]     = raw
            time.sleep(SLEEP)
        except Exception as e:
            print(f"  ERROR {s}: {e}")
            rec[f"summary_{s}"] = ""
            rec[f"raw_{s}"]     = ""
    records.append(rec)

# ── save ─────────────────────────────────────────────────────────────────────
out = pd.DataFrame(records)
out.to_parquet(OUT_FILE, index=False)
out.to_csv(OUT_FILE.replace(".parquet", ".csv"), index=False)
print(f"\nSaved {len(out)} rows → {OUT_FILE}")

# ── verify ────────────────────────────────────────────────────────────────────
for s in SETTINGS_R:
    empty = (out[f"summary_{s}"].fillna("").str.strip() == "").sum()
    if empty:
        print(f"  WARNING: {s} has {empty} empty summaries")
print("Done.")
