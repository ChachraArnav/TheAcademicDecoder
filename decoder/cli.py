"""Terminal version: python -m decoder [notes.txt|.pdf|.docx]"""

from __future__ import annotations

import argparse
import datetime
from pathlib import Path

from .quiz import check_answer, make_quiz
from .report import build_report
from .summarize import summarize
from .text import clean, read_file

LINE = "=" * 55


def grade(percent: float) -> str:
    if percent == 100:
        return "A+ (perfect)"
    if percent >= 80:
        return "A (great work)"
    if percent >= 60:
        return "B (review the ones you missed)"
    if percent >= 40:
        return "C (more studying needed)"
    return "D (re-read the summary)"


def read_pasted() -> str:
    print("\n  Paste your notes. Press Enter on an empty line twice when done:\n")
    lines: list[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line == "" and lines and lines[-1] == "":
            break
        lines.append(line)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Summarise notes and quiz yourself on them.")
    parser.add_argument(
        "file", nargs="?", help="notes file (.txt, .md, .pdf, .docx); omit to paste"
    )
    parser.add_argument("-s", "--sentences", type=int, default=5, help="summary length")
    parser.add_argument("-q", "--questions", type=int, default=5, help="number of questions")
    parser.add_argument("--seed", type=int, default=None, help="pick a different set of questions")
    args = parser.parse_args(argv)

    print(f"\n{LINE}\n  THE ACADEMIC DECODER\n{LINE}")
    text = clean(read_file(args.file) if args.file else read_pasted())
    if not text:
        print("  No text entered. Exiting.")
        return

    summary = summarize(text, args.sentences)
    print(f"\n{LINE}\n  SUMMARY\n{LINE}")
    print(f"\n  {len(summary.sentences)} sentences -> {len(summary.selected)} selected\n")
    for sentence in summary.selected:
        print(f"  - {sentence.text}")
    print("\n  Key terms: " + ", ".join(summary.terms(10)))

    questions = make_quiz(text, args.questions, seed=args.seed)
    if not questions:
        print("\n  Not enough material to build questions from these notes.")
        return

    print(f"\n{LINE}\n  CHECK YOUR KNOWLEDGE\n{LINE}")
    answers, results = {}, {}
    for i, q in enumerate(questions):
        print(f"\nQ{i + 1}. {q.prompt}")
        if q.multiple_choice:
            for letter, option in zip("abcd", q.options, strict=False):
                print(f"     {letter}) {option}")
        given = input("  Your answer: ").strip()
        if q.multiple_choice and len(given) == 1 and given.lower() in "abcd"[: len(q.options)]:
            given = q.options["abcd".index(given.lower())]
        answers[i], results[i] = given, check_answer(given, q.answer)
        print("  Correct!" if results[i] else f"  Answer: {q.answer}")

    score = sum(results.values())
    percent = score / len(questions) * 100
    print(f"\n  SCORE: {score}/{len(questions)} ({percent:.0f}%)  Grade: {grade(percent)}")

    if input("\n  Save report? (y/n): ").strip().lower() == "y":
        name = Path(f"report_{datetime.datetime.now():%Y%m%d_%H%M%S}.md")
        name.write_text(build_report(summary, questions, answers, results), encoding="utf-8")
        print(f"  Saved to {name}")
    print("\n  Done! Good luck studying.\n")
