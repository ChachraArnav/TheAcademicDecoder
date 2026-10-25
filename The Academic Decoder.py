#The Academic Decoder — COMP4475


import spacy
import nltk
import string
import random
import datetime
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import defaultdict

nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

nlp = spacy.load("en_core_web_sm")
STOPWORDS = set(stopwords.words("english"))
CHEAT_MODE = False  


#1.SUMMARIZER 

def summarize(text, top_n=5):
    sentences = sent_tokenize(text)

    freq = defaultdict(int)
    for word in word_tokenize(text.lower()):
        if word not in STOPWORDS and word not in string.punctuation:
            freq[word] += 1

    max_freq = max(freq.values()) if freq else 1
    for word in freq:
        freq[word] /= max_freq

    scores = defaultdict(float)
    for sent in sentences:
        for word in word_tokenize(sent.lower()):
            if word in freq:
                scores[sent] += freq[word]

    top = set(sorted(scores, key=scores.get, reverse=True)[:top_n])
    summary = " ".join(s for s in sentences if s in top)
    return summary, scores, sentences


#2.QUIZ GENERATOR
def get_targets(text):
    doc = nlp(text)
    targets = {}
    for ent in doc.ents:
        targets[ent.text] = ent.label_
    for token in doc:
        if token.pos_ == "PROPN" and token.text not in targets:
            targets[token.text] = "PROPN"
    #Also grab important nouns and technical terms
    for token in doc:
        if (token.pos_ in ("NOUN", "ADJ")
                and len(token.text) >= 5
                and token.text.lower() not in STOPWORDS
                and token.text not in targets
                and token.is_alpha):
            targets[token.text] = "TERM"
    return sorted(targets.items(), key=lambda x: len(x[0]), reverse=True)


def make_quiz(sentences, targets, n=10):
    questions = []
    used = set()
    target_words = {t[0] for t in targets}
    eligible = [s for s in sentences if any(t in s for t in target_words)]
    random.shuffle(eligible)

    for sent in eligible:
        if len(questions) >= n:
            break
        if sent in used:
            continue
        for word, label in targets:
            if word in sent:
                blank = "_" * len(word)
                questions.append({
                    "q": sent.replace(word, f"[{blank}]", 1),
                    "a": word
                })
                used.add(sent)
                break
    return questions


#3.INTERACTIVE QUIZ 

def run_quiz(questions):
    print("\n" + "="*55)
    print("  CHECK YOUR KNOWLEDGE")
    print("="*55)
    score = 0
    results = []
    for i, item in enumerate(questions, 1):
        print(f"\nQ{i}. {item['q']}")
        if CHEAT_MODE:
            print(f"  💡 Answer: {item['a']}")
        answer = input("  Your answer: ").strip()
        correct = answer.lower() == item["a"].lower()
        if correct:
            print("  ✅ Correct!")
            score += 1
        else:
            print(f"  ❌ Answer: {item['a']}")
        results.append({**item, "yours": answer, "correct": correct})

    pct = score / len(questions) * 100
    print(f"\n  SCORE: {score}/{len(questions)} ({pct:.0f}%)")
    if pct == 100: print("  Grade: A+ — Perfect!")
    elif pct >= 80: print("  Grade: A  — Great work!")
    elif pct >= 60: print("  Grade: B  — Review missed topics.")
    elif pct >= 40: print("  Grade: C  — More studying needed.")
    else:           print("  Grade: D  — Re-read the summary.")

    return results, score


#4.SAVE REPORT

def save_report(summary, scores, quiz_results, score, total):
    filename = f"report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write("THE ACADEMIC DECODER — STUDY REPORT\n")
        f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write("── SUMMARY ──\n")
        f.write(summary + "\n\n")
        f.write("── SENTENCE SCORES ──\n")
        for i, (s, sc) in enumerate(sorted(scores.items(), key=lambda x: x[1], reverse=True), 1):
            f.write(f"  S{i:02d} [{sc:.2f}] {'█' * int(sc*2)}\n")
        f.write(f"\n── QUIZ RESULTS: {score}/{total} ──\n")
        for i, r in enumerate(quiz_results, 1):
            status = "✓" if r["correct"] else "✗"
            f.write(f"  Q{i} [{status}] Answer: {r['a']} | You said: {r['yours']}\n")
    print(f"\n  💾 Saved to: {filename}")


#5.MAIN 

def main():
    print("\n" + "="*55)
    print("  THE ACADEMIC DECODER")
    print("="*55)
    print("\n  Input options:")
    print("  1. Paste notes in terminal")
    print("  2. Load from a .txt file")
    choice = input("\n  Choose (1 or 2): ").strip()

    if choice == "2":
        path = input("  File path: ").strip().strip('"')
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        print("\n  Paste your notes. Press Enter twice when done:\n")
        lines = []
        while True:
            line = input()
            if line == "" and lines and lines[-1] == "":
                break
            lines.append(line)
        text = "\n".join(lines).strip()

    if not text:
        print("  No text entered. Exiting.")
        return

    top_n = input("\n  Summary sentences (default 5): ").strip()
    top_n = int(top_n) if top_n.isdigit() else 5

    summary, scores, sentences = summarize(text, top_n)

    print("\n" + "="*55)
    print("  SUMMARY")
    print("="*55)
    print(f"\n  {len(sentences)} sentences → {len(sent_tokenize(summary))} selected\n")
    print(summary)

    print("\n── Sentence Scores ──")
    for i, (s, sc) in enumerate(sorted(scores.items(), key=lambda x: x[1], reverse=True), 1):
        print(f"  S{i:02d} [{sc:.2f}] {'█' * int(sc*2)}")

    targets = get_targets(text)
    questions = make_quiz(sentences, targets, n=5)
    quiz_results, score = run_quiz(questions)

    if input("\n  Save report? (y/n): ").strip().lower() == "y":
        save_report(summary, scores, quiz_results, score, len(questions))

    print("\n  Done! Good luck studying.\n")


if __name__ == "__main__":
    main()