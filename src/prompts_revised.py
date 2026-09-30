"""
prompts_revised.py — Revised Faithful Representation Framework
Fixes scientific issues in the original prompts.py:

  S1      — unchanged (clean baseline)
  S2_r    — fixed: symmetric instruction (center articles → neutral summary)
  S3a_r   — fixed: definition-based neutral anchor (removes cross-dataset BM25 retrieval)
  S3b_r   — fixed: definition-based full-spectrum (removes "same topic" false claim)
  S3c_r   — fixed: definition-based spectrum + symmetric faithful instruction
  S4_r    — fixed: expose classification step (remove "no preamble" suppression)
  S5_r    — improved: explicit scale (-2 to +2) + evidence grounding
  S6_r    — fixed: preserve ≠ amplify (add explicit center handling)

Changes vs original:
  S2   → added symmetric branch: "if neutral, write neutral summary"
  S3a  → removed BM25 retrieval; replaced with explicit definition of neutral reporting
  S3b  → removed BM25 retrieval + false "same topic" claim; replaced with L/C/R definitions
  S3c  → same as S3b fix, combined with symmetric faithful instruction
  S4   → removed "output summary only" suppression; model now outputs Framing: + Summary:
  S5   → added -2..+2 scale + one-phrase evidence grounding; explicit center branch
  S6   → added explicit "if centre/neutral → neutral summary" branch
"""

LEFT_RIGHT_DEF = (
    "Left-leaning framing may include: emphasis on systemic inequality, "
    "pro-government-intervention language, anti-corporate rhetoric, or progressive social framing.\n"
    "Centre/neutral framing: balanced reporting without clear partisan framing; "
    "presents multiple perspectives or sticks to factual reporting without editorial slant.\n"
    "Right-leaning framing may include: emphasis on free-market solutions, anti-regulation language, "
    "patriotic or traditional-values framing, or anti-government-spending rhetoric.\n"
)

NEUTRAL_DEF = (
    "Neutral, factual news reporting is characterised by: presenting verifiable facts without "
    "editorial judgment, covering multiple perspectives, and avoiding charged or partisan language."
)


# ─────────────────────────────────────────
# S1 — Direct summarization (baseline, unchanged)
# ─────────────────────────────────────────

def prompt_s1(article: str) -> str:
    return (
        "Summarize the following news article in 3 to 5 sentences.\n\n"
        f"Article:\n{article}\n\nSummary:"
    )


# ─────────────────────────────────────────
# S2_r — Symmetric faithful instruction
# Fix: original only said "do not shift toward neutrality",
#      which gave no guidance for centre articles → high inject%.
#      Revised adds explicit centre branch.
# ─────────────────────────────────────────

def prompt_s2_r(article: str) -> str:
    return (
        "Summarize the following news article in 3 to 5 sentences.\n"
        "Your summary should faithfully represent the original article's perspective and framing:\n"
        "- If the article has a partisan stance (left or right leaning), preserve it — "
        "do not soften or amplify it.\n"
        "- If the article is neutral or centrist, write a neutral summary — "
        "do not introduce any partisan framing.\n\n"
        f"Article:\n{article}\n\nSummary:"
    )


# ─────────────────────────────────────────
# S3a_r — Neutral calibration anchor (definition-based)
# Fix: original retrieved sentences from Hyperpartisan News Detection dataset
#      via BM25 keyword match — cross-dataset, off-topic contamination.
#      Revised replaces retrieval with an explicit definition of neutral reporting.
# ─────────────────────────────────────────

def prompt_s3a_r(article: str) -> str:
    return (
        f"{NEUTRAL_DEF}\n\n"
        "Use this as a calibration baseline. "
        "If the article you are summarising departs from neutral reporting — "
        "in a left or right direction — your summary should faithfully reflect that departure. "
        "If the article is neutral, write a neutral summary.\n\n"
        f"Now summarise the article below in 3 to 5 sentences:\n\n"
        f"Article:\n{article}\n\nSummary:"
    )


# ─────────────────────────────────────────
# S3b_r — Full-spectrum calibration (definition-based)
# Fix: original said "how the same topic is reported" — factually false,
#      examples came from a different dataset matched by keyword only.
#      Revised uses L/C/R definitions as the spectrum reference — no retrieval.
# ─────────────────────────────────────────

def prompt_s3b_r(article: str) -> str:
    return (
        "Political news reporting falls on a spectrum. Use the following as a reference:\n\n"
        f"{LEFT_RIGHT_DEF}\n"
        "Identify where this article sits on the spectrum, "
        "then summarise it in 3 to 5 sentences while faithfully preserving that position. "
        "Do not shift it toward or away from the centre.\n\n"
        f"Article:\n{article}\n\nSummary:"
    )


# ─────────────────────────────────────────
# S3c_r — Full-spectrum + symmetric faithful instruction (definition-based)
# Fix: same cross-dataset issue as S3b; also inherits S2's asymmetric instruction.
#      Revised combines definition-based spectrum with symmetric faithful branch.
# ─────────────────────────────────────────

def prompt_s3c_r(article: str) -> str:
    return (
        "Political news reporting falls on a spectrum. Use the following as a reference:\n\n"
        f"{LEFT_RIGHT_DEF}\n"
        "Using this spectrum as a guide, summarise the article below in 3 to 5 sentences:\n"
        "- If it is left or right leaning, preserve that framing faithfully — "
        "do not soften or amplify it.\n"
        "- If it is neutral or centrist, write a neutral summary — "
        "do not introduce any partisan framing.\n\n"
        f"Article:\n{article}\n\nSummary:"
    )


# ─────────────────────────────────────────
# S4_r — Explicit classification + preserve (with visible reasoning)
# Fix: original suppressed classification step ("output summary only,
#      no preamble") — made the reasoning unverifiable.
#      Revised exposes the framing label so it can be audited.
# ─────────────────────────────────────────

def prompt_s4_r(article: str) -> str:
    return (
        f"{LEFT_RIGHT_DEF}\n"
        "Read the article below. First, identify which framing "
        "(left / centre / right) best describes it. "
        "Then write a 3 to 5 sentence summary that faithfully preserves that framing — "
        "do not amplify or remove it.\n\n"
        "Format your response as:\n"
        "Framing: <left / centre / right>\n"
        "Summary: <3-5 sentences>\n\n"
        f"Article:\n{article}"
    )


# ─────────────────────────────────────────
# S5_r — Improved CoT: explicit scale + evidence grounding
# Improvement over S5: adds -2..+2 numerical scale (forces calibration)
#   and requires one cited phrase as evidence (grounding).
#   Explicit centre branch in Step 2 prevents inject for neutral articles.
# ─────────────────────────────────────────

def prompt_s5_r(article: str) -> str:
    return (
        "Read the article below and complete the two steps.\n\n"
        "Step 1 — Stance: Rate the article's political lean:\n"
        "  -2 = strongly left   -1 = leans left   0 = neutral/centre   "
        "+1 = leans right   +2 = strongly right\n"
        "  Then cite one specific phrase or framing choice from the article "
        "that justifies your rating.\n\n"
        "Step 2 — Summary: Write a 3 to 5 sentence summary.\n"
        "  - If rating ≠ 0: preserve the specific framing you cited — "
        "do not soften or amplify it.\n"
        "  - If rating = 0: write a factual, neutral summary — "
        "do not introduce any partisan framing.\n\n"
        "Format your response as:\n"
        "Stance: <rating> — <one-sentence evidence>\n"
        "Summary: <3-5 sentences>\n\n"
        f"Article:\n{article}"
    )


# ─────────────────────────────────────────
# S5_adv — Evidence-Anchored CoT (three-step)
# Advance over S5_r: evidence is extracted FIRST (prior to rating),
#   making the causal chain explicit: evidence → rating → summary.
#   The summary is explicitly bound to Step 1 evidence, preventing both
#   suppression (model ignores framing cues) and amplification (model
#   free-wheels beyond what the evidence supports).
#   Auditable: Evidence + Lean fields can be checked against original.
# ─────────────────────────────────────────

def prompt_s5_adv(article: str) -> str:
    return (
        "Read the article below and complete the three steps.\n\n"
        "Step 1 — Evidence: Quote 2 to 3 specific phrases or sentences "
        "from the article that most clearly signal its political framing "
        "(word choice, emphasis, or perspective). Use the article's exact words.\n\n"
        "Step 2 — Lean rating: Based ONLY on the evidence from Step 1, "
        "rate the overall political lean:\n"
        "  -2 = strongly left   -1 = leans left   0 = neutral/centre   "
        "+1 = leans right   +2 = strongly right\n"
        "  In one sentence, explain how the quoted phrases support this rating.\n\n"
        "Step 3 — Summary: Write a 3 to 5 sentence summary.\n"
        "  - If lean ≠ 0: the summary must reflect the framing signals "
        "identified in Step 1 — do not soften, amplify, or substitute "
        "with different partisan framing.\n"
        "  - If lean = 0: write a factual, balanced summary — "
        "do not introduce any partisan framing.\n\n"
        "Format your response as:\n"
        'Evidence: ["<phrase 1>", "<phrase 2>", ...]\n'
        "Lean: <rating> — <one-sentence justification>\n"
        "Summary: <3-5 sentences>\n\n"
        f"Article:\n{article}"
    )


# ─────────────────────────────────────────
# S6_r — Fixed: preserve not amplify
# Fix: original "identify leaning → preserve leaning" caused the model
#      to amplify any detected lean, including in centre articles (inject% = 89.7%).
#      Revised adds explicit symmetric branch identical to S2_r.
# ─────────────────────────────────────────

def prompt_s6_r(article: str) -> str:
    return (
        "Read the article and identify its political leaning "
        "(left / centre / right).\n"
        "Then summarise it in 3 to 5 sentences:\n"
        "- If left or right leaning: preserve that framing faithfully — "
        "do not amplify or soften it.\n"
        "- If centre or neutral: write a factual, neutral summary — "
        "do not introduce any partisan framing.\n\n"
        f"Article:\n{article}\n\nSummary:"
    )


# ─────────────────────────────────────────
# Build all revised prompts for one article
# ─────────────────────────────────────────

SETTINGS_R   = ["s1", "s2_r", "s3a_r", "s3b_r", "s3c_r", "s4_r", "s5_r", "s5_adv", "s6_r"]
STRUCTURED_R  = {"s5_r", "s5_adv", "s4_r"}   # all emit "Summary:" label


def build_all_prompts_r(article: str) -> dict:
    return {
        "s1"     : prompt_s1(article),
        "s2_r"   : prompt_s2_r(article),
        "s3a_r"  : prompt_s3a_r(article),
        "s3b_r"  : prompt_s3b_r(article),
        "s3c_r"  : prompt_s3c_r(article),
        "s4_r"   : prompt_s4_r(article),
        "s5_r"   : prompt_s5_r(article),
        "s5_adv" : prompt_s5_adv(article),
        "s6_r"   : prompt_s6_r(article),
    }
