"""
eval_babe_bert_revised.py
BABE + BERTScore for summaries_environment_revised.parquet (9 revised settings).
Output: babe_environment_revised.parquet
"""
import os, sys, warnings
warnings.filterwarnings("ignore")

import pandas as pd
from transformers import pipeline
from bert_score import score as bert_score

THRESHOLD = 0.15
SETTINGS  = ["s1", "s2_r", "s3a_r", "s3b_r", "s3c_r", "s4_r", "s5_r", "s5_adv", "s6_r"]
LABELS    = ["S1 Baseline", "S2_r Faithful", "S3a_r Neutral", "S3b_r Spectrum",
             "S3c_r Spec+I", "S4_r L/R+C", "S5_r CoT", "S5_adv Ev-CoT", "S6_r Explicit"]

IN_FILE  = "summaries_environment_revised.parquet"
OUT_FILE = "babe_environment_revised.parquet"

print(f"\n{'='*60}")
print(f"Evaluating: environment_revised")
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
print("BABE Transition — environment_revised")
print(f"{'Setting':<18} {'Suppress%':>10} {'Inject%':>9} {'AvgDrift%':>10} {'BERTScore':>10}")
print('─'*60)
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
    drift = (sup_p + inj_p) / 2
    bert  = df[f"bertscore_F_{s}"].mean() if f"bertscore_F_{s}" in df.columns else float("nan")
    print(f"  {lbl:<16} {sup_p:>9.1f}% {inj_p:>8.1f}% {drift:>9.1f}% {bert:>10.4f}")
