"""
prompts.py — Faithful Representation Framework
Goal: summaries should faithfully reflect the source article's political stance,
      neither amplifying nor suppressing it.

Eight settings:
  S1      — Direct summarization (baseline: what does the model do by default?)
  S2      — Faithfulness instruction (explicit: preserve the original stance)
  S3a     — Neutral calibration anchors (zero-point reference on the spectrum)
  S3b     — Full-spectrum anchors (left/center/right range reference)
  S3c     — Full-spectrum anchors + faithfulness instruction
  S4      — Left/right definition + preserve what is present in the article
  S5_cot  — Chain-of-Thought: identify stance → faithfully reproduce it
  S6      — Explicit leaning preservation (reference condition)
"""

from retrieval import retrieve, format_bullets

FAITHFUL_INSTRUCTION = (
    "Your summary should faithfully represent the original article's perspective and framing. "
    "Preserve its political stance as written — do not amplify it, soften it, or shift it toward neutrality."
)

SUMMARIZE_SUFFIX = "Now summarize the article below in 3 to 5 sentences:\n\nArticle:\n{article}\n\nSummary:"

BIAS_TAG = {
    "left"   : "[left-leaning]",
    "right"  : "[right-leaning]",
    "neutral": "[neutral]",
}

LEFT_RIGHT_DEF = (
    "Left-leaning framing may include: emphasis on systemic inequality, "
    "pro-government-intervention language, anti-corporate rhetoric, or progressive social framing.\n"
    "Centre/neutral framing: balanced reporting without clear partisan framing; "
    "presents multiple perspectives or sticks to factual reporting without editorial slant.\n"
    "Right-leaning framing may include: emphasis on free-market solutions, anti-regulation language, "
    "patriotic or traditional-values framing, or anti-government-spending rhetoric.\n"
)


def format_bullets_labeled(sentences: list[str], leaning: str, header: str = None) -> str:
    tag    = BIAS_TAG.get(leaning, "")
    prefix = f"[{header}]\n" if header else ""
    lines  = "\n".join(f"- {s}  {tag}" for s in sentences)
    return prefix + lines


# ─────────────────────────────────────────
# S1 — Direct summarization (baseline)
# ─────────────────────────────────────────

def prompt_s1(article: str) -> str:
    return (
        "Summarize the following news article in 3 to 5 sentences.\n\n"
        f"Article:\n{article}\n\nSummary:"
    )


# ─────────────────────────────────────────
# S2 — Faithfulness instruction
# ─────────────────────────────────────────

def prompt_s2(article: str) -> str:
    return (
        f"Summarize the following news article in 3 to 5 sentences. "
        f"{FAITHFUL_INSTRUCTION}\n\n"
        f"Article:\n{article}\n\nSummary:"
    )


# ─────────────────────────────────────────
# S3a — Neutral calibration anchors
# The neutral sentences serve as a zero-point reference on the political spectrum.
# The model should gauge where the article sits relative to this baseline
# and preserve that distance faithfully.
# ─────────────────────────────────────────

def prompt_s3a(article: str, title: str, indices: dict) -> str:
    neutral = retrieve(title, indices, leaning="neutral", top_k=3)
    context = format_bullets_labeled(neutral, leaning="neutral")
    return (
        "The following sentences represent neutral, factual reporting on a related topic. "
        "Use them as a reference point to gauge the political spectrum. "
        "If the article you are summarizing is more partisan than these sentences, "
        "your summary should faithfully reflect that difference — do not move it closer to neutral.\n\n"
        f"{context}\n\n"
        + SUMMARIZE_SUFFIX.format(article=article)
    )


# ─────────────────────────────────────────
# S3b — Full-spectrum anchors (left / center / right)
# Shows the range of political expression so the model can locate the article
# on the spectrum and reproduce that position faithfully.
# ─────────────────────────────────────────

def prompt_s3b(article: str, title: str, indices: dict) -> str:
    left    = retrieve(title, indices, leaning="left",    top_k=1)
    right   = retrieve(title, indices, leaning="right",   top_k=1)
    neutral = retrieve(title, indices, leaning="neutral", top_k=1)

    context = (
        format_bullets_labeled(left,    leaning="left",    header="Left-leaning example")    + "\n\n" +
        format_bullets_labeled(neutral, leaning="neutral", header="Neutral example")          + "\n\n" +
        format_bullets_labeled(right,   leaning="right",   header="Right-leaning example")
    )
    return (
        "The following sentences illustrate how the same topic is reported across the political spectrum. "
        "Use them to identify where the article you are summarizing sits on this spectrum, "
        "then summarize it in 3 to 5 sentences while faithfully preserving that position.\n\n"
        f"{context}\n\n"
        + SUMMARIZE_SUFFIX.format(article=article)
    )


# ─────────────────────────────────────────
# S3c — Full-spectrum anchors + faithfulness instruction
# ─────────────────────────────────────────

def prompt_s3c(article: str, title: str, indices: dict) -> str:
    left    = retrieve(title, indices, leaning="left",    top_k=1)
    right   = retrieve(title, indices, leaning="right",   top_k=1)
    neutral = retrieve(title, indices, leaning="neutral", top_k=1)

    context = (
        format_bullets_labeled(left,    leaning="left",    header="Left-leaning example")    + "\n\n" +
        format_bullets_labeled(neutral, leaning="neutral", header="Neutral example")          + "\n\n" +
        format_bullets_labeled(right,   leaning="right",   header="Right-leaning example")
    )
    return (
        "The following sentences illustrate how the same topic is reported across the political spectrum. "
        "Use them to identify where the article sits on this spectrum.\n\n"
        f"{context}\n\n"
        f"Summarize the article below in 3 to 5 sentences. "
        f"{FAITHFUL_INSTRUCTION}\n\n"
        f"Article:\n{article}\n\nSummary:"
    )


# ─────────────────────────────────────────
# S4 — Left/right definition + preserve what is present
# The model identifies which characteristics are present in the article
# and preserves them — neither amplifying nor suppressing.
# ─────────────────────────────────────────

def prompt_s4(article: str) -> str:
    return (
        f"{LEFT_RIGHT_DEF}\n"
        "Summarize the following news article in 3 to 5 sentences. "
        "Identify which of the above framings (left / centre / right) best describes the article, "
        "then write a summary that faithfully preserves that framing — do not amplify or remove it. "
        "Output the summary only, with no preamble or framing analysis.\n\n"
        f"Article:\n{article}\n\nSummary:"
    )


# ─────────────────────────────────────────
# S5_cot — Chain-of-Thought: identify stance → faithfully reproduce
# ─────────────────────────────────────────

def prompt_s5_cot(article: str) -> str:
    return (
        "Read the article below and complete the two steps.\n\n"
        "Step 1 — Stance identification: In one sentence, describe the article's political "
        "stance and key framing choices (e.g., which perspective it emphasises, what language it uses).\n"
        "Step 2 — Faithful summary: Write a 3 to 5 sentence summary that faithfully reflects "
        "that stance as identified in Step 1 — do not amplify it or soften it.\n\n"
        "Format your response exactly as:\n"
        "Stance: <one sentence>\n"
        "Summary: <3-5 sentences>\n\n"
        f"Article:\n{article}"
    )


# ─────────────────────────────────────────
# S6 — Explicit leaning preservation (reference condition)
# Model infers and explicitly preserves the article's political leaning.
# Serves as a reference for what faithful preservation looks like
# when directly instructed.
# ─────────────────────────────────────────

def prompt_s6(article: str, bias_text: str = "") -> str:
    return (
        "Read the article and identify its political leaning. "
        "Then summarize it in 3 to 5 sentences while faithfully preserving that leaning.\n\n"
        f"Article:\n{article}\n\nSummary:"
    )


# ─────────────────────────────────────────
# Build all prompts for one article
# ─────────────────────────────────────────

def build_all_prompts(article: str, title: str, indices: dict,
                      bias_text: str = "center") -> dict[str, str]:
    return {
        "s1"    : prompt_s1(article),
        "s2"    : prompt_s2(article),
        "s3a"   : prompt_s3a(article, title, indices),
        "s3b"   : prompt_s3b(article, title, indices),
        "s3c"   : prompt_s3c(article, title, indices),
        "s4"    : prompt_s4(article),
        "s5_cot": prompt_s5_cot(article),
        "s6"    : prompt_s6(article, bias_text),
    }


# ─────────────────────────────────────────
# Entry point — inspect prompts
# ─────────────────────────────────────────

if __name__ == "__main__":
    from retrieval import build_indices

    indices = build_indices()
    article = (
        "The radical left-wing Democrats are hellbent on destroying everything "
        "that made America great. Their latest scheme — a socialist spending spree "
        "that would bankrupt every hardworking family..."
    )
    title = "Democrats spending bill Congress"

    prompts = build_all_prompts(article, title, indices)
    for name, prompt in prompts.items():
        print(f"\n{'─'*50}\nSetting {name.upper()}\n{'─'*50}")
        print(prompt)
