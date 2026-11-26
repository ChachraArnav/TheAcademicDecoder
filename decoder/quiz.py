"""Fill-in-the-blank questions built from the notes themselves.

For each sentence the generator finds candidate answers:

  * named entities spaCy recognises (people, organisations, dates, places ...)
  * key terms: nouns, joined with the nouns in front of them, so the answer is
    "carbon dioxide" rather than just "dioxide"

Each candidate gets an importance score: how often its words appear in the
notes (the same word scores the summary uses), plus a bonus for named
entities. The best candidate in a sentence is blanked out to make the
question, and the most important questions across the notes are kept.

Multiple-choice options are other candidates of the same kind from the same
notes (another date for a date, another person for a person), so the wrong
answers are plausible without anything being made up.
"""

from __future__ import annotations

import random
import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher

import numpy as np

from .nlp import GENERIC_NOUNS, get_nlp, is_content_word, term_span
from .summarize import summarize

BLANK = "_____"
ENTITY_BONUS = 0.5
SKIP_ENTITY_LABELS = {"ORDINAL", "CARDINAL"}
MIN_SENTENCE_WORDS = 6
MAX_SENTENCE_WORDS = 60
OPTIONS = 4

# Wrong answers come from the same group as the right one: a date for a date,
# a person for a person. spaCy often files science terms and acronyms under
# ORG or PRODUCT, so those share a pool with ordinary terms.
GROUPS = {
    "PERSON": "person",
    "DATE": "time",
    "TIME": "time",
    "GPE": "place",
    "LOC": "place",
    "FAC": "place",
    "PERCENT": "number",
    "MONEY": "number",
    "QUANTITY": "number",
}


def group(kind: str) -> str:
    return GROUPS.get(kind, "term")


@dataclass
class Candidate:
    text: str
    kind: str  # spaCy entity label, or "TERM"
    start: int  # character offsets within the sentence
    end: int
    importance: float
    vector: np.ndarray | None = field(default=None, repr=False, compare=False)


@dataclass
class Question:
    prompt: str
    answer: str
    kind: str
    sentence_index: int
    sentence: str
    options: list[str] = field(default_factory=list)

    @property
    def multiple_choice(self) -> bool:
        return len(self.options) >= 2


def _trim(span):
    """Drop leading 'the' / 'a' / punctuation and trailing punctuation from a span."""
    start, end = span.start, span.end
    while start < end and (span.doc[start].pos_ in ("DET", "PUNCT") or span.doc[start].is_space):
        start += 1
    # "UDP (User Datagram Protocol" -> "UDP": never leave a bracket half-open.
    for i in range(start, end):
        if span.doc[i].text == "(" and not any(t.text == ")" for t in span.doc[i:end]):
            end = i
            break
    while end > start and (span.doc[end - 1].is_punct or span.doc[end - 1].is_space):
        end -= 1
    return span.doc[start:end]


def _vector(span) -> np.ndarray | None:
    return span.vector if span.has_vector and span.vector_norm else None


def _similarity(a: Candidate, b: Candidate) -> float:
    """Cosine similarity of the two terms' word vectors (0 if either has none)."""
    if a.vector is None or b.vector is None:
        return 0.0
    return float(np.dot(a.vector, b.vector) / (np.linalg.norm(a.vector) * np.linalg.norm(b.vector)))


def _candidates(sentence, word_scores) -> list[Candidate]:
    found: dict[tuple[int, int], Candidate] = {}
    offset = sentence.start_char

    def importance(span) -> float:
        words = [t for t in span if is_content_word(t)]
        if not words:
            return 0.0
        return max(word_scores.get(t.lemma_.lower(), 0.0) for t in words)

    for ent in sentence.ents:
        if ent.label_ in SKIP_ENTITY_LABELS:
            continue
        span = _trim(ent)
        if not span or len(span) > 6:
            continue
        found[(span.start_char, span.end_char)] = Candidate(
            text=span.text,
            kind=ent.label_,
            start=span.start_char - offset,
            end=span.end_char - offset,
            importance=importance(span) + ENTITY_BONUS,
            vector=_vector(span),
        )

    for token in sentence:
        if token.pos_ not in ("NOUN", "PROPN") or not is_content_word(token):
            continue
        if len(token.text) < 4 and not token.text.isupper():
            continue  # short words are rarely worth testing, but acronyms like TCP are
        if token.dep_ == "compound":
            continue  # handled as part of the noun it modifies
        span = term_span(token)
        if len(span) == 1 and token.lemma_.lower() in GENERIC_NOUNS:
            continue
        inside_entity = any(s <= span.start_char and span.end_char <= e for (s, e) in found)
        if inside_entity:
            continue
        found[(span.start_char, span.end_char)] = Candidate(
            text=span.text,
            kind="TERM",
            start=span.start_char - offset,
            end=span.end_char - offset,
            importance=importance(span),
            vector=_vector(span),
        )

    # An entity inside a longer term ("Calvin" in "Calvin cycle") gives way to the term.
    terms = [(s, e) for (s, e), c in found.items() if c.kind == "TERM"]
    for (s, e), c in list(found.items()):
        if c.kind != "TERM" and any(ts <= s and e <= te and (ts, te) != (s, e) for ts, te in terms):
            del found[(s, e)]

    sentence_text = sentence.text
    usable = []
    for c in found.values():
        # If the answer appears twice in the sentence, blanking one gives it away.
        if len(re.findall(rf"\b{re.escape(c.text)}\b", sentence_text, flags=re.IGNORECASE)) > 1:
            continue
        usable.append(c)
    return usable


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^\w\s]", " ", text.lower())
    text = re.sub(r"^(the|a|an)\s+", "", text.strip())
    return re.sub(r"\s+", " ", text).strip()


def check_answer(given: str, expected: str) -> bool:
    """Case, punctuation, a leading 'the' and a plural 's' are ignored. A small
    typo is accepted only in longer answers (8+ letters, 88% similar), so
    'plant' is not accepted for 'planet'."""
    a, b = normalise(given), normalise(expected)
    if not a:
        return False
    if a == b or a.rstrip("s") == b.rstrip("s"):
        return True
    return len(b) >= 8 and SequenceMatcher(None, a, b).ratio() >= 0.88


def _options(
    answer: Candidate, pool: list[Candidate], prompt: str, rng: random.Random
) -> list[str]:
    key = normalise(answer.text)
    shown = normalise(prompt)
    seen = {key}
    same_kind = []
    for c in pool:
        k = normalise(c.text)
        # Skip repeats, overlaps with the answer, and anything already visible
        # in the question (it could be ruled out at a glance).
        if group(c.kind) != group(answer.kind) or k in seen or k in key or key in k or k in shown:
            continue
        seen.add(k)
        same_kind.append(c)
    # Hardest wrong answers first: closest in meaning (word vectors), then
    # similar length so the right answer doesn't stand out.
    same_kind.sort(
        key=lambda c: (
            -round(_similarity(answer, c), 2),
            abs(len(c.text.split()) - len(answer.text.split())),
        )
    )
    distractors = [c.text for c in same_kind[: OPTIONS - 1]]
    if len(distractors) < OPTIONS - 1:
        return []
    options = [answer.text, *distractors]
    rng.shuffle(options)
    return options


def make_quiz(text: str, n: int = 5, seed: int | None = None, doc=None) -> list[Question]:
    """Build up to n questions. The same notes and seed give the same quiz;
    a different seed picks from the next-best questions for variety."""
    doc = doc if doc is not None else get_nlp()(text)
    word_scores = summarize(text, doc=doc).word_scores

    best_per_sentence = []
    pool: list[Candidate] = []
    for index, sentence in enumerate(doc.sents):
        words = [t for t in sentence if not t.is_punct and not t.is_space]
        candidates = _candidates(sentence, word_scores)
        pool += candidates
        if not candidates or not MIN_SENTENCE_WORDS <= len(words) <= MAX_SENTENCE_WORDS:
            continue
        best = max(candidates, key=lambda c: (c.importance, len(c.text)))
        best_per_sentence.append((index, sentence, best))

    best_per_sentence.sort(key=lambda item: -item[2].importance)
    if seed is not None:
        rng = random.Random(seed)
        shortlist = best_per_sentence[: max(2 * n, n)]
        best_per_sentence = rng.sample(shortlist, len(shortlist))

    chosen, used_answers = [], set()
    for index, sentence, best in best_per_sentence:
        key = normalise(best.text)
        if key in used_answers:
            continue
        used_answers.add(key)
        chosen.append((index, sentence, best))
        if len(chosen) == n:
            break

    rng = random.Random(seed if seed is not None else 0)
    questions = []
    for index, sentence, best in sorted(chosen, key=lambda item: item[0]):
        s = sentence.text
        questions.append(
            Question(
                prompt=(s[: best.start] + BLANK + s[best.end :]).strip(),
                answer=best.text,
                kind=best.kind,
                sentence_index=index,
                sentence=s.strip(),
                options=_options(best, pool, s[: best.start] + s[best.end :], rng),
            )
        )
    return questions
