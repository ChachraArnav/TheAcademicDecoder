"""Extractive summary: pick the sentences that use the notes' most frequent words.

This is the classic word-frequency method (Luhn, 1958):

  1. Count how often each content word appears. Words are reduced to their
     lemma first, so "reaction" and "reactions" count as the same word.
  2. Divide every count by the highest count, so the most common word scores 1.
  3. A sentence's score is the sum of the scores of the words in it.
  4. Keep the top-scoring sentences, shown in their original order.

Nothing is generated: every summary sentence is copied from the notes.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from .nlp import GENERIC_NOUNS, get_nlp, is_content_word, term_span

# Headings and fragments are too short to be useful summary sentences.
MIN_WORDS = 5


@dataclass
class Sentence:
    index: int
    text: str
    score: float
    selected: bool = False
    # Line breaks after this sentence in the notes: 0, 1 (new line) or 2 (new paragraph).
    breaks_after: int = 0


@dataclass
class Summary:
    sentences: list[Sentence]
    word_scores: dict[str, float] = field(default_factory=dict)
    key_terms: list[tuple[str, int]] = field(default_factory=list)

    @property
    def selected(self) -> list[Sentence]:
        return [s for s in self.sentences if s.selected]

    @property
    def text(self) -> str:
        return " ".join(s.text for s in self.selected)

    def top_words(self, n: int = 10) -> list[tuple[str, float]]:
        return sorted(self.word_scores.items(), key=lambda item: -item[1])[:n]

    def terms(self, n: int = 10) -> list[str]:
        return [term for term, _ in self.key_terms[:n]]


def summarize(text: str, top_n: int = 5, doc=None) -> Summary:
    doc = doc if doc is not None else get_nlp()(text)

    counts = Counter(t.lemma_.lower() for t in doc if is_content_word(t))
    highest = max(counts.values(), default=1)
    word_scores = {word: count / highest for word, count in counts.items()}

    # Key terms: the most frequent nouns and noun compounds ("carbon dioxide"),
    # shown the way they are most often written in the notes.
    term_counts: Counter[str] = Counter()
    spellings: dict[str, Counter[str]] = {}
    for t in doc:
        if t.pos_ not in ("NOUN", "PROPN") or t.dep_ == "compound" or not is_content_word(t):
            continue
        span = term_span(t)
        too_short = len(t.text) < 4 and not t.text.isupper()
        if len(span) == 1 and (t.lemma_.lower() in GENERIC_NOUNS or too_short):
            continue
        key = "".join(w.lemma_.lower() + w.whitespace_ for w in span).strip()
        term_counts[key] += 1
        shown = "".join(
            (w.text if w.pos_ == "PROPN" or not w.text[1:].islower() else w.lemma_.lower())
            + w.whitespace_
            for w in span
        ).strip()
        spellings.setdefault(key, Counter())[shown] += 1
    ranked_terms = sorted(term_counts.items(), key=lambda item: (-item[1], -len(item[0].split())))
    key_terms = [(spellings[k].most_common(1)[0][0], n) for k, n in ranked_terms[:15]]

    sentences = []
    for i, span in enumerate(doc.sents):
        words = [t for t in span if not t.is_punct and not t.is_space]
        score = 0.0
        if len(words) >= MIN_WORDS:
            score = sum(word_scores.get(t.lemma_.lower(), 0.0) for t in span if is_content_word(t))
        trailing = span.text[len(span.text.rstrip()) :] + span[-1].whitespace_
        sentences.append(
            Sentence(
                index=i,
                text=span.text.strip(),
                score=round(score, 3),
                breaks_after=min(trailing.count("\n"), 2),
            )
        )

    ranked = sorted((s for s in sentences if s.score > 0), key=lambda s: (-s.score, s.index))
    for sentence in ranked[: max(top_n, 0)]:
        sentence.selected = True
    return Summary(sentences=sentences, word_scores=word_scores, key_terms=key_terms)
