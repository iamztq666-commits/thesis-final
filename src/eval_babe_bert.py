"""
eval_babe_bert.py
BABE + BERTScore evaluation for a topic (no LLM Judge).
Usage: python3 eval_babe_bert.py <topic>
  e.g. python3 eval_babe_bert.py environment
       python3 eval_babe_bert.py education
       python3 eval_babe_bert.py gun_control_and_gun_rights
"""
import os, sys, warnings
warnings.filterwarnings("ignore")

import pandas as pd
from transformers import pipeline
from bert_score import score as bert_score

TOPIC     = sys.argv[1] if len(sys.argv) > 1 else "environment"
THRESHOLD = 0.15
SETTINGS  = ["s1","s2","s3a","s3b","s3c","s4","s5_cot","s6"]
LABELS    = ["S1 Baseline","S2 Faithful","S3a Neutral","S3b Spectrum",
             "S3c Spec+I","S4 L/R+C","S5 CoT","S6 Explicit"]

IN_FILE  = f"summaries_{TOPIC}.parquet"
OUT_FILE = f"babe_{TOPIC}.parquet"

print(f"\n{'='*60}")
print(f"Evaluating: {TOPIC}")
print(f"Input:  {IN_FILE}")
print(f"Output: {OUT_FILE}")
print('='*60)

df = pd.read_parquet(IN_FILE)
print(f"Loaded {len(df)} articles  "
      f"L={(df.bias_text=='left').sum()} "
      f"C={(df.bias_text=='center').sum()} "
      f"R={(df.bias_text=='right').sum()}")

# ── BABE ──────────────────────────────────────────────────────
print("\n[1/2] BABE scoring...")
babe = pipeline("text-classification", model="mediabiasgroup/da-roberta-babe-ft",
                truncation=True, max_length=512)

def babe_fn(text):
    try:
        r = babe(str(text)[:512])[0]
        lbl = r["label"].lower()
        return r["score"] if ("biased" in lbl and "non" not in lbl) else 1 - r["score"]
    except:
        return None

df["orig_bias_score"] = [babe_fn(t) for t in df["input_text"]]
print(f"  orig: mean={df['orig_bias_score'].mean():.3f}")

for s in SETTINGS:
    col = f"summary_{s}"
    if col in df.columns:
        df[f"{s}_bias_score"] = [babe_fn(t) for t in df[col]]
        print(f"  {s}: mean={df[f'{s}_bias_score'].mean():.3f}")

# ── BERTScore ─────────────────────────────────────────────────
print("\n[2/2] BERTScore (roberta-large)...")
refs = df["input_text"].tolist()
for s in SETTINGS:
    col = f"summary_{s}"
    if col not in df.columns:
        continue
    hyps = [t if t and len(str(t).strip()) > 5 else "no summary available"
            for t in df[col].fillna("").tolist()]
    _, _, F = bert_score(hyps, refs, lang="en", model_type="roberta-large",
                         verbose=False, batch_size=8)
    df[f"bertscore_F_{s}"] = F.numpy()
    print(f"  {s}: F1={F.mean():.4f}")

df.to_parquet(OUT_FILE, index=False)
df.to_csv(OUT_FILE.replace(".parquet", ".csv"), index=False)
print(f"\nSaved → {OUT_FILE}")

# ── BABE Transition summary ────────────────────────────────────
print(f"\n{'='*60}")
print(f"BABE Transition — {TOPIC}")
print(f"{'Setting':<16} {'Suppress%':>10} {'Inject%':>9} {'AvgDrift%':>10} {'BERTScore':>10}")
print('─'*58)
partisan = df[df.bias_text != 'center']
center   = df[df.bias_text == 'center']
for s, lbl in zip(SETTINGS, LABELS):
    if f"{s}_bias_score" not in df.columns:
        continue
    p_tot = (partisan["orig_bias_score"] >= THRESHOLD).sum()
    sup   = ((partisan["orig_bias_score"] >= THRESHOLD) &
             (partisan[f"{s}_bias_score"]  < THRESHOLD)).sum()
    c_tot = (center["orig_bias_score"] < THRESHOLD).sum()
    inj   = ((center["orig_bias_score"] < THRESHOLD) &
             (center[f"{s}_bias_score"] >= THRESHOLD)).sum()
    sup_p = sup / max(p_tot, 1) * 100
    inj_p = inj / max(c_tot, 1) * 100
    avg   = (sup_p + inj_p) / 2
    bert  = df[f"bertscore_F_{s}"].mean() if f"bertscore_F_{s}" in df.columns else float("nan")
    print(f"{lbl:<16} {sup_p:>9.1f}% {inj_p:>8.1f}% {avg:>9.1f}% {bert:>10.4f}")
