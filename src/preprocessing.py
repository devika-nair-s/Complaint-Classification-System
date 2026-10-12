"""Lightweight text cleanup that preserves useful complaint wording.

TF-IDF performs lowercasing and tokenization. Keep negations and sentiment words
such as "not", "never", and "unhelpful": they distinguish complaint classes.
"""

import html
import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


# Drop common function words, but retain negation and complaint intensity.
STOP_WORDS = ENGLISH_STOP_WORDS.difference(
    {"no", "not", "nor", "never", "without", "against", "unable", "cannot", "too", "very"}
)

def clean_text(text):
    text = html.unescape(str(text))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    return re.sub(r"\s+", " ", text).strip()
