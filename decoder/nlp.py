"""Shared spaCy pipeline.

Everything in the decoder runs on one spaCy model: sentence splitting, stop
words, lemmas, part-of-speech tags and named entities. Loading it takes a
second or two, so it is loaded once and reused.
"""

from __future__ import annotations

from functools import lru_cache

import spacy
from spacy.language import Language

MODEL = "en_core_web_md"

# spaCy refuses texts above this length by default; course notes rarely get
# close, but a long PDF can.
MAX_CHARACTERS = 300_000


@Language.component("line_breaks_end_sentences")
def line_breaks_end_sentences(doc):
    """Headings and bullet points have no full stop, so treat every line
    break as the end of a sentence instead of gluing lines together."""
    for token in doc[:-1]:
        if "\n" in token.text_with_ws or token.is_space and "\n" in token.text:
            doc[token.i + 1].is_sent_start = True
    return doc


@lru_cache(maxsize=1)
def get_nlp() -> Language:
    nlp = spacy.load(MODEL)
    nlp.add_pipe("line_breaks_end_sentences", before="parser")
    nlp.max_length = MAX_CHARACTERS + 1
    return nlp


def is_content_word(token) -> bool:
    """A word that carries meaning: not a stop word, number-only or punctuation."""
    return token.is_alpha and not token.is_stop and len(token.text) > 1


# Nouns too general to be worth testing on their own.
GENERIC_NOUNS = {
    "example",
    "factor",
    "form",
    "kind",
    "number",
    "part",
    "place",
    "process",
    "result",
    "step",
    "thing",
    "time",
    "type",
    "use",
    "way",
    "case",
    "fact",
}


def term_span(token):
    """A noun plus the compound nouns directly in front of it, so the whole
    term 'carbon dioxide concentration' is kept together."""
    doc, start = token.doc, token.i
    while start > token.sent.start:
        previous = doc[start - 1]
        if previous.dep_ != "compound" or not previous.is_alpha:
            break
        if not start - 1 < previous.head.i <= token.i:
            break
        start -= 1
    # Keep hyphenated words whole: "three-way handshake", not "way handshake".
    while (
        start - 2 >= token.sent.start
        and doc[start - 1].text == "-"
        and not doc[start - 1].whitespace_
        and not doc[start - 2].whitespace_
        and doc[start - 2].is_alpha
    ):
        start -= 2
    return doc[start : token.i + 1]
