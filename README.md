# Political Framing Fidelity in LLM News Summarisation

Code and de-identified evaluation outputs for the thesis study of political framing fidelity in LLM-generated news summaries.

## What is included

- `src/`: prompt templates, data-loading helpers, generation/evaluation scripts, and `analyze.py` for reproducing descriptive results and paired tests from the released score files.
- `data/article_scores.csv`: de-identified article-level BABE scores and BERTScore values for the original and revised Environment, Education, and Gun Control experiments.
- `data/llm_judgements.csv`: de-identified GPT-4o-mini and DeepSeek-V3 judge outputs for Environment.
- `results/`: descriptive tables, Base-vs-All paired tests, revision-effect tests, and aggregated LLM-judge results.
- `prompts/`: original and revised prompt documentation.
- `docs/source_manifest.json`: checksums and provenance for local source files used to build the release.

The release contains no source article text, source URLs, raw model responses, API keys, model caches, or private credentials. The original news corpus is not redistributed; obtain it from its original source under its own terms.

## Reproduce the released analyses

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/analyze.py
```

`analyze.py` performs no API calls and no model downloads. It reads the released de-identified score files and writes CSV outputs to `results/`. The BABE threshold is 0.15; Suppression and Injection use their respective eligible source subsets. Holm correction is applied within each family of comparisons.

## Regenerating summaries or scores

The scripts in `src/` document the generation and evaluation pipeline, but rerunning generation requires the original article data, model checkpoints, and provider API credentials. Hosted models and model checkpoints can change, so regenerated text and scores may not exactly match the released outputs. Never commit `.env` files or credentials.

## Important provenance note

The released descriptive and inferential results use the tokenizer-level BABE rerun with a 512-token limit. The original untruncated intermediate files are intentionally excluded from this repository.
