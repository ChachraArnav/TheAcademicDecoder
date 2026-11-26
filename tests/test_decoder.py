import io

import pytest

from decoder import check_answer, clean, make_quiz, read_bytes, summarize
from decoder.quiz import BLANK
from decoder.text import UnsupportedFile

NOTES = """Photosynthesis

Photosynthesis is the process by which green plants convert light energy into chemical energy.
It takes place mainly in the chloroplasts of leaf cells. Chlorophyll absorbs red and blue light.
The Calvin cycle uses ATP and NADPH to fix carbon dioxide into glucose. Melvin Calvin described
the cycle in the 1950s at the University of California, Berkeley. Light intensity, carbon dioxide
concentration and temperature limit the rate of photosynthesis. Water molecules are split in the
light reactions, releasing oxygen. RuBisCO catalyses the first step of carbon fixation.
"""


# ------------------------------------------------------------------ text --


def test_clean_joins_wrapped_lines_and_hyphenated_words():
    text = clean("The cycle uses ATP to fix\ncarbon dioxide. Photo-\nsynthesis needs light.")
    assert text == "The cycle uses ATP to fix carbon dioxide. Photosynthesis needs light."


def test_clean_keeps_headings_separate_and_drops_bullets_and_page_numbers():
    text = clean("The Calvin Cycle\n- It happens in the stroma.\n12\n")
    assert text == "The Calvin Cycle\nIt happens in the stroma."


def test_read_bytes_plain_text_and_unsupported_type():
    assert read_bytes("notes.txt", b"Hello world.") == "Hello world."
    with pytest.raises(UnsupportedFile):
        read_bytes("notes.xlsx", b"")


def test_read_bytes_docx():
    from docx import Document

    buffer = io.BytesIO()
    document = Document()
    document.add_paragraph("Mitochondria produce ATP.")
    document.save(buffer)
    assert "Mitochondria produce ATP." in read_bytes("notes.docx", buffer.getvalue())


def test_read_bytes_pdf():
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    pdf.drawString(72, 720, "Ribosomes build proteins from amino acids.")
    pdf.save()
    assert "Ribosomes build proteins" in read_bytes("notes.pdf", buffer.getvalue())


# --------------------------------------------------------------- summary --


def test_summary_copies_sentences_verbatim_in_original_order():
    text = clean(NOTES)
    summary = summarize(text, top_n=3)
    assert len(summary.selected) == 3
    for sentence in summary.selected:
        assert sentence.text in text
    indexes = [s.index for s in summary.selected]
    assert indexes == sorted(indexes)


def test_summary_picks_the_highest_scoring_sentences():
    summary = summarize(clean(NOTES), top_n=2)
    chosen = {s.score for s in summary.selected}
    others = [s.score for s in summary.sentences if not s.selected]
    assert min(chosen) >= max(others)


def test_heading_is_never_a_summary_sentence():
    summary = summarize(clean(NOTES), top_n=20)
    heading = summary.sentences[0]
    assert heading.text == "Photosynthesis" and heading.score == 0 and not heading.selected


def test_word_scores_are_normalised_and_lemmatised():
    summary = summarize("Reactions happen. The reaction is fast. A reaction needs energy here.")
    assert max(summary.word_scores.values()) == 1.0
    assert summary.word_scores["reaction"] == 1.0
    assert "reactions" not in summary.word_scores


def test_key_terms_keep_compound_nouns_together():
    terms = [t.lower() for t in summarize(clean(NOTES)).terms(15)]
    assert "carbon dioxide" in terms


def test_hyphenated_terms_stay_whole():
    terms = summarize(
        "The client completes a three-way handshake. Every three-way handshake uses SYN."
    )
    assert "three-way handshake" in terms.terms(5)


def test_empty_and_tiny_input_do_not_crash():
    assert summarize("").selected == []
    assert summarize("Hi.").selected == []
    assert make_quiz("") == []
    assert make_quiz("Hi there.") == []


# ------------------------------------------------------------------ quiz --


def test_each_question_blanks_its_answer_out_of_a_real_sentence():
    text = clean(NOTES)
    questions = make_quiz(text, n=5)
    assert 3 <= len(questions) <= 5
    for q in questions:
        assert q.prompt.count(BLANK) == 1
        assert q.prompt.replace(BLANK, q.answer) == q.sentence
        assert q.sentence in text


def test_answers_are_unique_and_never_start_with_the():
    questions = make_quiz(clean(NOTES), n=6)
    answers = [q.answer.lower() for q in questions]
    assert len(answers) == len(set(answers))
    assert not any(a.startswith(("the ", "a ", "an ")) for a in answers)


def test_multiple_choice_options_include_the_answer_once_and_come_from_the_notes():
    text = clean(NOTES)
    for q in make_quiz(text, n=6):
        if q.multiple_choice:
            assert len(q.options) == 4
            assert q.options.count(q.answer) == 1
            assert len(set(o.lower() for o in q.options)) == 4
            for option in q.options:
                assert option in text


def test_same_seed_same_quiz_and_new_seed_can_change_it():
    text = clean(NOTES)
    first = [q.prompt for q in make_quiz(text, n=3, seed=1)]
    assert first == [q.prompt for q in make_quiz(text, n=3, seed=1)]
    variants = {tuple(q.prompt for q in make_quiz(text, n=3, seed=s)) for s in range(1, 8)}
    assert len(variants) > 1


def test_whole_term_is_blanked_not_part_of_it():
    questions = make_quiz(clean(NOTES), n=10)
    answers = {q.answer.lower() for q in questions}
    assert "dioxide" not in answers  # it should be "carbon dioxide" or not asked at all


# ------------------------------------------------------------- answering --


@pytest.mark.parametrize(
    "given, expected",
    [
        ("chloroplasts", "chloroplasts"),
        ("Chloroplasts", "chloroplasts"),
        ("chloroplast", "chloroplasts"),
        ("university of california berkeley", "the University of California, Berkeley"),
        ("Melvn Calvin", "Melvin Calvin"),
        ("chlorophyl", "Chlorophyll"),
    ],
)
def test_check_answer_accepts(given, expected):
    assert check_answer(given, expected)


@pytest.mark.parametrize(
    "given, expected",
    [("", "ATP"), ("ADP", "ATP"), ("plant", "planet"), ("oxygen", "carbon dioxide")],
)
def test_check_answer_rejects(given, expected):
    assert not check_answer(given, expected)
