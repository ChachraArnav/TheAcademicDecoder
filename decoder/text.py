"""Read notes from a file and clean them up before analysis."""

from __future__ import annotations

import io
import re
from pathlib import Path

SUPPORTED = (".txt", ".md", ".pdf", ".docx")


class UnsupportedFile(ValueError):
    pass


def read_bytes(name: str, data: bytes) -> str:
    """Extract plain text from an uploaded file's bytes."""
    suffix = Path(name).suffix.lower()
    if suffix in (".txt", ".md"):
        return data.decode("utf-8", errors="replace")
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if suffix == ".docx":
        from docx import Document

        document = Document(io.BytesIO(data))
        return "\n".join(paragraph.text for paragraph in document.paragraphs)
    raise UnsupportedFile(f"Unsupported file type '{suffix}'. Use one of: {', '.join(SUPPORTED)}")


def read_file(path: str | Path) -> str:
    path = Path(path)
    return read_bytes(path.name, path.read_bytes())


def _is_heading(line: str) -> bool:
    """Short line with no closing punctuation, e.g. 'The Calvin Cycle'."""
    return len(line.split()) <= 6 and not line.endswith((".", "!", "?", ",", ";"))


def clean(text: str) -> str:
    """Undo the usual damage from copying out of PDFs and slides.

    - words split across lines with a hyphen ("photo-\\nsynthesis")
    - single line breaks inside a paragraph
    - bullet characters and lines that are only a page number
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    lines = []
    for line in text.split("\n"):
        line = re.sub(r"^\s*[•▪◦●\-\*–]\s+", "", line).strip()
        if re.fullmatch(r"(page\s*)?\d{1,4}", line, flags=re.IGNORECASE):
            continue
        lines.append(line)
    paragraphs = re.split(r"\n\s*\n", "\n".join(lines))
    joined = []
    for paragraph in paragraphs:
        parts = [p for p in paragraph.split("\n") if p]
        if not parts:
            continue
        # A line without closing punctuation followed by more text is a wrapped
        # line, so join with a space; otherwise keep it as its own sentence.
        merged = parts[0]
        for previous, part in zip(parts, parts[1:], strict=False):
            # A next line starting in lower case always continues the sentence.
            wrapped = part[:1].islower() or (
                not previous.endswith((".", "!", "?", ":")) and not _is_heading(previous)
            )
            merged += (" " if wrapped else "\n") + part
        joined.append(merged)
    text = "\n\n".join(joined)
    return re.sub(r"[ \t]+", " ", text).strip()
