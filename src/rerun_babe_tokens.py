"""Recompute BABE scores with tokenizer-level truncation at 512 tokens.

Existing BERTScore columns are retained; only BABE score columns are replaced.
Outputs are written to *_tokens.parquet/csv so the previous results remain intact.
"""
import os
import warnings
warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["no_proxy"] = "*"
os.environ["NO_PROXY"] = "*"

import pandas as pd
from transformers import pipeline

MODEL = "mediabiasgroup/da-roberta-babe-ft"
SETTINGS_ORIG = ["s1", "s2", "s3a", "s3b", "s3c", "s4", "s5_cot", "s6"]
SETTINGS_REV = ["s1", "s2_r", "s3a_r", "s3b_r", "s3c_r", "s4_r", "s5_r", "s5_adv", "s6_r"]

print("Loading BABE classifier with tokenizer-level max_length=512 ...")
babe = pipeline(
    "text-classification",
    model=MODEL,
    truncation=True,
    max_length=512,
)

def babe_score(text):
    result = babe(str(text))[0]
    label = result["label"].lower()
    return result["score"] if ("biased" in label and "non" not in label) else 1 - result["score"]

def process(input_file, old_eval_file, output_file, settings):
    src = pd.read_parquet(input_file)
    old = pd.read_parquet(old_eval_file)
    keys = ["id", "title", "bias_text", "input_text"]
    if not src[keys].reset_index(drop=True).equals(old[keys].reset_index(drop=True)):
        raise ValueError(f"Metadata mismatch between {input_file} and {old_eval_file}")

    out = old.copy()
    print(f"\n{input_file}: {len(src)} articles, {len(settings)} settings")
    print("Scoring source texts ...")
    out["orig_bias_score"] = [babe_score(x) for x in src["input_text"]]
    for setting in settings:
        column = f"summary_{setting}"
        print(f"Scoring {column} ...")
        out[f"{setting}_bias_score"] = [babe_score(x) for x in src[column].fillna("")]

    out.to_parquet(output_file, index=False)
    out.to_csv(output_file.replace(".parquet", ".csv"), index=False)
    score_cols = ["orig_bias_score"] + [f"{s}_bias_score" for s in settings]
    print(f"Saved {output_file}; null scores: {int(out[score_cols].isna().sum().sum())}")

process("summaries_environment.parquet", "babe_environment.parquet", "babe_environment_tokens.parquet", SETTINGS_ORIG)
process("summaries_education.parquet", "babe_education.parquet", "babe_education_tokens.parquet", SETTINGS_ORIG)
process("summaries_gun_control_and_gun_rights.parquet", "babe_gun_control_and_gun_rights.parquet", "babe_gun_control_and_gun_rights_tokens.parquet", SETTINGS_ORIG)
process("summaries_environment_revised.parquet", "babe_environment_revised.parquet", "babe_environment_revised_tokens.parquet", SETTINGS_REV)

print("\nCompleted BABE-only token-level rerun.")
