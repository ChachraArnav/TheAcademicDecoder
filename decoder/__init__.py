"""The Academic Decoder: summaries and practice questions from your own notes."""

from .quiz import Question, check_answer, make_quiz
from .summarize import Summary, summarize
from .text import clean, read_bytes, read_file

__all__ = [
    "Question",
    "Summary",
    "check_answer",
    "clean",
    "make_quiz",
    "read_bytes",
    "read_file",
    "summarize",
]
