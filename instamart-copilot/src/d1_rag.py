"""Part D, tasks 1-4: policy RAG assistant (chunking, TF-IDF retrieval, grounded answering, evaluation).

Importable module (chunk / Retriever / retrieve / answer) and runnable script (prints the task 1-4 report).
Reads data/policies/*.txt and data/raw/policy_eval.csv. All model calls go through mock_llm.chat().
"""
import os

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from common import POLICIES, RAW, check, start_report
import mock_llm

CHUNK_WORDS = 40
OVERLAP = 10
FALLBACK_PREFIX = "I could not find"

POLICY_SYSTEM_PROMPT = ("You are the POLICY ASSISTANT for Instamart support. Answer only from the supplied CONTEXT. "
                        "If the context does not contain the answer, say you could not find the information.")


def load_docs():
    docs = {}
    for fn in sorted(os.listdir(POLICIES)):
        if fn.endswith(".txt"):
            with open(os.path.join(POLICIES, fn), encoding="utf-8") as fh:
                docs[fn] = fh.read()
    return docs


def chunk_text(text, size=CHUNK_WORDS, overlap=OVERLAP):
    """Windows of `size` words, each starting `size - overlap` words after the previous one.
    Stops after the first chunk whose end reaches or passes the last word."""
    words, stride, chunks, start = text.split(), size - overlap, [], 0
    while True:
        end = start + size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            return chunks
        start += stride


def build_chunks(size=CHUNK_WORDS, overlap=OVERLAP):
    return [{"source": fn, "chunk_no": i, "text": t}
            for fn, doc in load_docs().items() for i, t in enumerate(chunk_text(doc, size, overlap))]


class Retriever:
    """TF-IDF over the chunk texts; queries use the same fitted vectorizer; ranking by cosine similarity."""

    def __init__(self, size=CHUNK_WORDS, overlap=OVERLAP):
        self.chunks = build_chunks(size, overlap)
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform([c["text"] for c in self.chunks])

    def retrieve(self, query, k=3):
        sims = cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
        order = sorted(range(len(sims)), key=lambda i: (-sims[i], i))[:k]
        return [{"source": self.chunks[i]["source"], "score": float(sims[i]), "text": self.chunks[i]["text"]} for i in order]


_DEFAULT = None


def default_retriever():
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Retriever()
    return _DEFAULT


def retrieve(query, k=3):
    """retrieve(query, k=3) -> list of {source, score, text} using the default 40-word / 10-overlap index."""
    return default_retriever().retrieve(query, k)


def answer(question, retriever=None):
    """Retrieve three chunks, send them to the mock POLICY ASSISTANT and return (text, sources)."""
    hits = (retriever or default_retriever()).retrieve(question, k=3)
    context = "\n".join(f"[{h['source']}] {h['text']}" for h in hits)
    messages = [{"role": "system", "content": POLICY_SYSTEM_PROMPT},
                {"role": "user", "content": f"CONTEXT:\n{context}\nQUESTION:\n{question}"}]
    reply = mock_llm.chat(messages).choices[0].message.content
    return reply, [h["source"] for h in hits]


def evaluate(retriever, verbose=True):
    ev = pd.read_csv(os.path.join(RAW, "policy_eval.csv"), keep_default_na=False)   # blank expected_fact stays ""
    ans = ev[ev["expected_doc"] != "NONE"]
    unans = ev[ev["expected_doc"] == "NONE"]
    hit_doc = hit_fact = fallback = 0
    failures = []
    for _, r in ans.iterrows():
        text, sources = answer(r["question"], retriever)
        d, f = r["expected_doc"] in sources, r["expected_fact"] in text
        hit_doc += d
        hit_fact += f
        if not (d and f):
            failures.append((r["question"], r["expected_fact"], text))
        if verbose:
            print(f"    {'ok ' if d and f else 'FAIL'} doc_in_top3={d} fact_in_answer={f} | {r['question']}")
    for _, r in unans.iterrows():
        text, _ = answer(r["question"], retriever)
        fb = text.startswith(FALLBACK_PREFIX)
        fallback += fb
        if verbose:
            print(f"    {'ok ' if fb else 'FAIL'} fallback={fb} | {r['question']} -> {text}")
    return {"answerable": len(ans), "doc_hits": int(hit_doc), "fact_hits": int(hit_fact),
            "unanswerable": len(unans), "fallbacks": int(fallback), "failures": failures}


if __name__ == "__main__":
    start_report("d1_rag")
    print("=== TASK 1: chunking (40 words, overlap 10) ===")
    r = default_retriever()
    per_doc = pd.Series([c["source"] for c in r.chunks]).value_counts().sort_index()
    print(f"  total chunks: {len(r.chunks)}; per document: {per_doc.to_dict()}")
    check("total chunks", len(r.chunks), 15)
    check("per document", per_doc.to_dict(), {"delivery_policy.txt": 3, "escalation_policy.txt": 3, "payment_policy.txt": 3,
                                              "privacy_policy.txt": 2, "refund_policy.txt": 4})
    c0 = [c for c in r.chunks if c["source"] == "refund_policy.txt"]
    print(f"  refund_policy.txt chunk 0 words 0-39, chunk 1 starts at word 30: "
          f"{c0[1]['text'].split()[:3]} == {c0[0]['text'].split()[30:33]}")

    print("\n=== TASK 2: TF-IDF retrieval ===")
    print(f"  vocabulary size: {len(r.vectorizer.vocabulary_)}")
    for h in retrieve("refund policy late delivery"):
        print(f"  {h['source']:<22} score {h['score']:.4f}  {h['text'][:70]}...")
    top3 = [h["source"] for h in retrieve("refund policy late delivery")]
    check("top three chunks all from refund_policy.txt", top3, ["refund_policy.txt"] * 3)

    print("\n=== TASK 3: one grounded answer ===")
    q = "What delivery time makes a completed order eligible for a full refund?"
    text, sources = answer(q)
    print(f"  Q: {q}\n  A: {text}\n  sources: {sources}")

    print("\n=== TASK 4: evaluation with overlap 10 (40-word chunks) ===")
    ev10 = evaluate(r)
    print(f"  answerable: expected doc in top three {ev10['doc_hits']}/{ev10['answerable']}, "
          f"expected_fact in answer {ev10['fact_hits']}/{ev10['answerable']}; "
          f"unanswerable: fallback {ev10['fallbacks']}/{ev10['unanswerable']}")
    check("doc hits", ev10["doc_hits"], 12)
    check("fact hits", ev10["fact_hits"], 12)
    check("fallbacks", ev10["fallbacks"], 2)

    print("\n=== TASK 4: evaluation with ZERO overlap (40-word chunks, stride 40) ===")
    r0 = Retriever(overlap=0)
    print(f"  total chunks with zero overlap: {len(r0.chunks)}")
    ev0 = evaluate(r0)
    print(f"  answerable: expected doc in top three {ev0['doc_hits']}/{ev0['answerable']}, "
          f"expected_fact in answer {ev0['fact_hits']}/{ev0['answerable']}; "
          f"unanswerable: fallback {ev0['fallbacks']}/{ev0['unanswerable']}")
    assert ev0["fact_hits"] < ev0["answerable"] or ev0["doc_hits"] < ev0["answerable"], "expected at least one failure"
    for q, fact, a in ev0["failures"]:
        print(f"  BREAKS: {q!r}\n     expected fact {fact!r}; answer was: {a!r}")
        for fn, d in load_docs().items():
            if fact and fact in d:
                words = d.split()
                before = len(d[:d.index(fact)].split())
                print(f"     the fact starts at word {before} of {fn}; zero-overlap chunk boundaries fall at words "
                      f"{list(range(40, len(words), 40))}, so the sentence is cut in two")
