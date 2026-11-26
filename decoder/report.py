"""Study report as Markdown: summary, key terms, and quiz results if taken."""

from __future__ import annotations

import datetime

from .quiz import Question
from .summarize import Summary


def build_report(
    summary: Summary,
    questions: list[Question],
    answers: dict[int, str] | None = None,
    results: dict[int, bool] | None = None,
    title: str = "Study notes",
) -> str:
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [f"# The Academic Decoder: {title}", f"_Generated {now}_", "", "## Summary", ""]
    lines += [f"- {s.text}" for s in summary.selected] or ["_No summary sentences found._"]

    lines += ["", "## Key terms", ""]
    lines += [", ".join(summary.terms(12)) or "_None found._"]

    lines += ["", "## Practice questions", ""]
    for i, q in enumerate(questions, 1):
        lines.append(f"{i}. {q.prompt}")
        if answers is not None and i - 1 in answers:
            mark = "correct" if results and results.get(i - 1) else "incorrect"
            lines.append(f"   - Your answer: {answers[i - 1] or '(blank)'} ({mark})")
        lines.append(f"   - Answer: **{q.answer}**")
    if results:
        score = sum(results.values())
        lines += ["", f"**Score: {score}/{len(questions)}**"]
    return "\n".join(lines) + "\n"
