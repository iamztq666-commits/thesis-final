# Data dictionary

`article_scores.csv` contains one row per article and prompt condition.

- `id`: internal article identifier
- `topic`: Environment, Education, or Gun Control and Gun Rights
- `version`: `original` or `revised`
- `condition`: prompt setting identifier
- `bias_text`: source label (`left`, `center`, or `right`)
- `orig_bias_score`: BABE score for the source article
- `summary_bias_score`: BABE score for the generated summary
- `bertscore`: BERTScore F1 between source input and summary

`llm_judgements.csv` contains evaluator-level BiasShift and CenterDrift values for the Environment experiment.
