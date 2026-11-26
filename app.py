"""The Academic Decoder: web app.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import html
from pathlib import Path

import streamlit as st

from decoder.nlp import MAX_CHARACTERS, get_nlp
from decoder.quiz import BLANK, check_answer, group, make_quiz
from decoder.report import build_report
from decoder.summarize import summarize
from decoder.text import SUPPORTED, UnsupportedFile, clean, read_bytes

ROOT = Path(__file__).parent
SAMPLES = {
    "Biology: photosynthesis": ROOT / "samples" / "photosynthesis.txt",
    "Computer science: the transport layer": ROOT / "samples" / "networking.txt",
}
MAX_UPLOAD_MB = 10

st.set_page_config(page_title="The Academic Decoder", page_icon="📝", layout="wide")

# ---------------------------------------------------------------- styling --

st.markdown(f"<style>{(ROOT / 'assets' / 'style.css').read_text()}</style>", unsafe_allow_html=True)


def section(label: str) -> None:
    st.markdown(f'<div class="section">{label}</div>', unsafe_allow_html=True)


def esc(text: str) -> str:
    return html.escape(text, quote=False)


# --------------------------------------------------------------- analysis --


@st.cache_resource(show_spinner="Loading the language model…")
def load_model():
    return get_nlp()


@st.cache_data(show_spinner=False, max_entries=32)
def analyse(text: str, sentences: int, questions: int, seed: int):
    doc = load_model()(text)
    summary = summarize(text, sentences, doc=doc)
    quiz = make_quiz(text, questions, seed=seed or None, doc=doc)
    words = sum(1 for t in doc if t.is_alpha)
    return summary, quiz, words


def reset_quiz() -> None:
    for key in [k for k in st.session_state if str(k).startswith("answer_")]:
        del st.session_state[key]
    st.session_state.submitted = False


st.session_state.setdefault("text", "")
st.session_state.setdefault("seed", 0)
st.session_state.setdefault("submitted", False)
st.session_state.setdefault("title", "Study notes")

# ------------------------------------------------------------------- hero --

st.markdown(
    """
<div class="kicker">Study tool · classic NLP · runs locally</div>
<h1 class="hero-title">The Academic <mark>Decoder</mark></h1>
<p class="hero-sub">Paste or upload your notes. The decoder highlights the sentences that carry the
most weight, pulls out the key terms, and writes fill-in-the-blank questions from your own words,
so you can check what you actually remember.</p>
<div class="badges">
  <span class="badge"><b>spaCy</b> sentence, entity &amp; part-of-speech analysis</span>
  <span class="badge"><b>No AI service</b>, nothing is generated</span>
  <span class="badge"><b>Nothing stored</b> after you leave</span>
</div>
""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------ input --

section("1 · Your notes")
paste_tab, upload_tab, sample_tab = st.tabs(["Paste text", "Upload a file", "Try a sample"])

with paste_tab:
    pasted = st.text_area(
        "Notes",
        height=220,
        placeholder=(
            "Paste lecture notes, a reading or a textbook section. A few paragraphs works best."
        ),
        label_visibility="collapsed",
    )
    if st.button("Use this text", type="primary", disabled=not pasted.strip()):
        st.session_state.text, st.session_state.title = pasted, "Pasted notes"
        st.session_state.seed = 0
        reset_quiz()

with upload_tab:
    upload = st.file_uploader(
        f"Text, Markdown, PDF or Word file, up to {MAX_UPLOAD_MB} MB",
        type=[s.lstrip(".") for s in SUPPORTED],
    )
    if upload is not None and st.button("Decode this file", type="primary"):
        if upload.size > MAX_UPLOAD_MB * 1024 * 1024:
            st.error(f"That file is over {MAX_UPLOAD_MB} MB.")
        else:
            try:
                extracted = read_bytes(upload.name, upload.getvalue())
            except UnsupportedFile as error:
                st.error(str(error))
            except Exception:
                st.error(
                    "Couldn't read that file. If it's a scanned PDF, it has no text layer to read."
                )
            else:
                if not extracted.strip():
                    st.warning("No text found in that file. Scanned PDFs need OCR first.")
                st.session_state.text, st.session_state.title = extracted, Path(upload.name).stem
                st.session_state.seed = 0
                reset_quiz()

with sample_tab:
    choice = st.selectbox("Sample notes", list(SAMPLES), label_visibility="collapsed")
    st.caption("Short sample notes written for this demo.")
    if st.button("Load sample", type="primary"):
        st.session_state.text = SAMPLES[choice].read_text(encoding="utf-8")
        st.session_state.title = choice
        st.session_state.seed = 0
        reset_quiz()

raw = st.session_state.text
if not raw.strip():
    st.markdown(
        '<p class="hero-sub" style="font-size:.92rem">Nothing decoded yet. '
        "Paste some notes, upload a file, or load a sample to see it work.</p>",
        unsafe_allow_html=True,
    )
    st.stop()

text = clean(raw)
truncated = len(text) > MAX_CHARACTERS
if truncated:
    text = text[:MAX_CHARACTERS]
    st.warning(f"Long document: only the first {MAX_CHARACTERS:,} characters are analysed.")

with st.expander("Settings", expanded=False):
    c1, c2 = st.columns(2)
    n_sentences = c1.slider("Summary sentences", 2, 12, 5)
    n_questions = c2.slider("Practice questions", 3, 15, 6, on_change=reset_quiz)

with st.spinner("Reading your notes…"):
    summary, questions, word_count = analyse(text, n_sentences, n_questions, st.session_state.seed)

usable = [s for s in summary.sentences if s.score > 0]
if not usable:
    st.warning(
        "There isn't enough to work with here. The decoder needs full sentences, "
        "ideally a few paragraphs."
    )
    st.stop()

# ---------------------------------------------------------------- results --

section(f"2 · Decoded: {esc(st.session_state.title)}")
stats = st.columns(4)
for column, value, label in zip(
    stats,
    [
        f"{word_count:,}",
        f"{len(summary.sentences)}",
        f"{len(summary.selected)}",
        f"{len(questions)}",
    ],
    ["words", "sentences", "in summary", "questions"],
    strict=True,
):
    column.markdown(
        f'<div class="stat"><div class="v">{value}</div><div class="l">{label}</div></div>',
        unsafe_allow_html=True,
    )

summary_tab, quiz_tab, notes_tab, how_tab = st.tabs(
    ["Summary", "Quiz yourself", "Highlighted notes", "How it works"]
)

with summary_tab:
    left, right = st.columns([1.7, 1], gap="large")
    with left:
        items = "".join(
            f'<div class="summary-item"><div class="n">{i:02d}</div><p>{esc(s.text)}</p></div>'
            for i, s in enumerate(summary.selected, 1)
        )
        st.markdown(f'<div class="card">{items}</div>', unsafe_allow_html=True)
        st.caption(
            "Every sentence is copied word-for-word from your notes and shown in its "
            "original order. Nothing is reworded."
        )
    with right:
        st.markdown("**Key terms**")
        chips = "".join(
            f'<span class="chip">{esc(term)}<span>×{count}</span></span>'
            for term, count in summary.key_terms[:14]
        )
        st.markdown(f'<div class="chips">{chips}</div>', unsafe_allow_html=True)
        st.caption("The nouns and noun phrases that appear most often.")

with quiz_tab:
    if not questions:
        st.info("Couldn't find enough key terms to build questions from these notes.")
    else:
        top = st.columns([3, 1])
        top[0].caption(
            "Fill in each blank. Spelling slips in longer answers, capital letters and "
            "a leading 'the' are forgiven."
        )
        if top[1].button("New questions", icon=":material/refresh:", use_container_width=True):
            st.session_state.seed += 1
            reset_quiz()
            st.rerun()

        with st.form("quiz"):
            for i, q in enumerate(questions):
                parts = [esc(p) for p in q.prompt.split(BLANK)]
                prompt = '<span class="blank"></span>'.join(parts)
                st.markdown(
                    f'<div class="q-num">Question {i + 1} · {group(q.kind)}</div>'
                    f'<div class="q-prompt">{prompt}</div>',
                    unsafe_allow_html=True,
                )
                if q.multiple_choice:
                    st.radio(
                        "Answer",
                        q.options,
                        index=None,
                        key=f"answer_{i}",
                        horizontal=True,
                        label_visibility="collapsed",
                    )
                else:
                    st.text_input(
                        "Answer",
                        key=f"answer_{i}",
                        placeholder="Type your answer",
                        label_visibility="collapsed",
                    )
                if st.session_state.submitted:
                    given = st.session_state.get(f"answer_{i}") or ""
                    if check_answer(given, q.answer):
                        st.markdown(
                            f'<div class="verdict ok">Correct: <b>{esc(q.answer)}</b></div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        shown = esc(given) if given else "no answer"
                        st.markdown(
                            f'<div class="verdict bad">You said {shown}. The answer is '
                            f"<b>{esc(q.answer)}</b>."
                            f'<span class="src">From your notes: “{esc(q.sentence)}”</span></div>',
                            unsafe_allow_html=True,
                        )
                st.write("")
            if st.form_submit_button("Check answers", type="primary"):
                st.session_state.submitted = True
                st.rerun()

        if st.session_state.submitted:
            answers = {i: st.session_state.get(f"answer_{i}") or "" for i in range(len(questions))}
            results = {i: check_answer(answers[i], q.answer) for i, q in enumerate(questions)}
            score = sum(results.values())
            pct = score / len(questions) * 100
            verdict = (
                "Perfect."
                if pct == 100
                else "Strong."
                if pct >= 80
                else "Review the ones you missed."
                if pct >= 60
                else "Re-read the summary and try again."
            )
            c1, c2 = st.columns([1, 2.2])
            c1.markdown(
                f'<div class="score">{score}<small> / {len(questions)}</small></div>',
                unsafe_allow_html=True,
            )
            c2.markdown(f"**{pct:.0f}%**. {verdict}")
            c2.download_button(
                "Download study report",
                build_report(summary, questions, answers, results, st.session_state.title),
                file_name="study-report.md",
                mime="text/markdown",
                icon=":material/download:",
            )
        else:
            st.download_button(
                "Download summary and questions",
                build_report(summary, questions, title=st.session_state.title),
                file_name="study-notes.md",
                mime="text/markdown",
                icon=":material/download:",
            )

with notes_tab:
    st.caption(
        "Your full notes. Highlighted sentences made the summary; the small number is each "
        "sentence's score (the sum of its word scores)."
    )
    body = []
    for s in summary.sentences:
        score = (
            f'<span class="sc{" hit" if s.selected else ""}">{s.score:.1f}</span>'
            if s.score
            else ""
        )
        sentence = esc(s.text)
        body.append(f"{score}<mark>{sentence}</mark>" if s.selected else f"{score}{sentence}")
        body.append(["", "<br>", "</p><p>"][s.breaks_after] or " ")
    st.markdown(f'<div class="card notes"><p>{"".join(body)}</p></div>', unsafe_allow_html=True)

with how_tab:
    a, b = st.columns(2, gap="large")
    a.markdown(
        """
<div class="how">
<h4>Summary: word frequency</h4>
<p>The classic extractive method (Luhn, 1958). Words the notes use most often are probably what
they are about.</p>
<ol>
<li>Reduce each word to its dictionary form (<code>reactions</code> → <code>reaction</code>)
and drop filler words like <code>the</code> and <code>is</code>.</li>
<li>Count each word, then divide by the top count so the most common word scores 1.</li>
<li>Score each sentence as the sum of its word scores. Fragments under five words are skipped.</li>
<li>Keep the top sentences, in their original order.</li>
</ol>
</div>
""",
        unsafe_allow_html=True,
    )
    b.markdown(
        """
<div class="how">
<h4>Questions: blank out the key term</h4>
<p>For every sentence, spaCy finds candidate answers: named entities (people, dates,
places, organisations) and noun terms, kept whole (<code>carbon dioxide</code>, not
<code>dioxide</code>).</p>
<ol>
<li>Each candidate is scored by how often its words appear in the notes, with a bonus for named
entities.</li>
<li>The best candidate in a sentence becomes the blank; the strongest questions are kept.</li>
<li>Wrong options are other terms of the same kind from your notes, ranked by meaning with word
vectors so they are plausible.</li>
</ol>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="privacy"><b>Privacy.</b> Your notes are processed in memory on the server '
        "running this app and are not saved, logged or sent to any AI service. Close the tab and "
        "they are gone.</div>",
        unsafe_allow_html=True,
    )
    st.markdown("**Limits**")
    st.markdown(
        "- It picks sentences, it doesn't understand them: a summary can miss an idea that is "
        "stated only once.\n"
        "- spaCy's general-English model sometimes mislabels technical terms (an acronym tagged "
        "as an organisation, say), which can make an option look odd.\n"
        "- Scanned PDFs have no text layer and need OCR first."
    )

st.markdown(
    '<div class="foot">The Academic Decoder · Python, spaCy, Streamlit · '
    "built by Arnav Chachra</div>",
    unsafe_allow_html=True,
)
