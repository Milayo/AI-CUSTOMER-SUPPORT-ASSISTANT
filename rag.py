
import os
import glob
import re
import numpy as np

from config import (
    client, EMBED_MODEL, CHAT_MODEL,
    CHUNK_SIZE, OVERLAP_SENTENCES, TOP_K, SCORE_THRESHOLD,
    DOCS_FOLDER, COMPANY_NAME,
)


# ---------- 1. LOAD ----------
def load_documents(folder=DOCS_FOLDER):
    """Read every .txt file in the folder. Returns a list of (filename, text)."""
    docs = []
    for path in sorted(glob.glob(os.path.join(folder, "*.txt"))):
        with open(path, "r", encoding="utf-8") as f:
            docs.append((os.path.basename(path), f.read()))
    return docs


HEADING_RE = re.compile(r"^[A-Z][A-Z0-9 ,'&/()\-:]{3,}\??$")


def _is_heading(line):
    return bool(HEADING_RE.match(line.strip()))


def _split_units(text):
    """Break text into small units, remembering which heading each sits under.
    Splits on blank lines -> lines -> sentences, so bullet lists split properly."""
    units = []          # list of (heading, text)
    heading = ""
    for block in re.split(r"\n\s*\n", text.strip()):        # paragraphs
        for raw_line in block.split("\n"):                  # lines (bullets!)
            line = raw_line.strip()
            if not line:
                continue
            if _is_heading(line):
                heading = line
                continue
            for sentence in re.split(r"(?<=[.!?])\s+", line):   # sentences
                sentence = sentence.strip()
                if sentence:
                    units.append((heading, sentence))
    return units


def chunk_text(text, max_chars=CHUNK_SIZE, overlap_sentences=OVERLAP_SENTENCES):
    """Group units into chunks that stay under one heading and under max_chars.
    Each chunk is prefixed with its heading so it is self-describing."""
    units = _split_units(text)
    chunks = []
    current, current_heading, current_len = [], None, 0

    def flush():
        if current:
            body = " ".join(current)
            chunks.append(f"{current_heading}\n{body}" if current_heading else body)

    for h, sentence in units:
        heading_changed = (current_heading is not None and h != current_heading)
        too_long = current_len + len(sentence) > max_chars
        if current and (heading_changed or too_long):
            flush()
            # carry overlap only when we're still in the same section
            keep = current[-overlap_sentences:] if (overlap_sentences and not heading_changed) else []
            current, current_len = list(keep), sum(len(x) for x in keep)
        current_heading = h
        current.append(sentence)
        current_len += len(sentence)
    flush()

    return [c for c in chunks if c.strip()]


# ---------- 3. EMBED + BUILD INDEX ----------
def embed(texts):
    """Turn a list of strings into a list of embedding vectors."""
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in resp.data]


def build_index(folder=DOCS_FOLDER):
    """Load, chunk, and embed all documents. Returns a list of chunk records."""
    records = []
    for filename, text in load_documents(folder):
        for chunk in chunk_text(text):
            records.append({"source": filename, "text": chunk})
    if not records:
        return records
    vectors = embed([r["text"] for r in records])
    for r, v in zip(records, vectors):
        r["vector"] = v
    return records



def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def retrieve(question, index, top_k=TOP_K):
    """Return the top_k most relevant chunks as (score, record) pairs."""
    q_vec = embed([question])[0]
    scored = [(cosine_similarity(q_vec, r["vector"]), r) for r in index]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]



def generate_answer(question, results):
    """Given already-retrieved (score, record) results, produce a cited answer."""
    if not results or results[0][0] < SCORE_THRESHOLD:
        return (f"I'm sorry, I don't have that information in our help documents. "
                f"I'd be happy to connect you with a human agent from {COMPANY_NAME}.")

    context = "\n\n".join(
        f"[{i}] (from {r['source']})\n{r['text']}"
        for i, (score, r) in enumerate(results, 1)
    )
    system_prompt = (
        f"You are a friendly customer-support assistant for {COMPANY_NAME}. "
        "Answer the customer's question using ONLY the help-doc context provided below. "
        "If the answer is not contained in the context, say you don't have that "
        "information and offer to connect them with a human agent. "
        "Do not use outside knowledge. Be concise and clear. "
        "Cite the source number(s) you used, like [1] or [2]."
    )
    user_prompt = f"Help-doc context:\n{context}\n\nCustomer question: {question}"

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content



def answer_question(question, index):
    results = retrieve(question, index)
    return generate_answer(question, results)