
import os
import glob
import re
import numpy as np

from config import (
    client, EMBED_MODEL, CHAT_MODEL,
    CHUNK_SIZE, OVERLAP_SENTENCES, TOP_K, SCORE_THRESHOLD,
    DOCS_FOLDER, COMPANY_NAME,
)


def load_documents(folder=DOCS_FOLDER):
   
    docs = []
    for path in sorted(glob.glob(os.path.join(folder, "*.txt"))):
        with open(path, "r", encoding="utf-8") as f:
            docs.append((os.path.basename(path), f.read()))
    return docs



def chunk_text(text, max_chars=CHUNK_SIZE, overlap_sentences=OVERLAP_SENTENCES):

    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks, current, current_len = [], [], 0
    for sentence in sentences:
        if current_len + len(sentence) > max_chars and current:
            chunks.append(" ".join(current))
            current = current[-overlap_sentences:] if overlap_sentences else []
            current_len = sum(len(s) for s in current)
        current.append(sentence)
        current_len += len(sentence)
    if current:
        chunks.append(" ".join(current))
    return [c for c in chunks if c.strip()]     # drop any empty chunks



def embed(texts):
    """Turn a list of strings into a list of embedding vectors."""
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in resp.data]


def build_index(folder=DOCS_FOLDER):
    """Load, chunk, and embed all documents. Returns a list of chunk records:
    each record is {"source": filename, "text": chunk, "vector": embedding}."""
    records = []
    for filename, text in load_documents(folder):
        for chunk in chunk_text(text):
            records.append({"source": filename, "text": chunk})

    if not records:
        return records

    vectors = embed([r["text"] for r in records])   # embed all chunks in one batch
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



def answer_question(question, index):
    """Retrieve relevant context and generate a grounded, cited support answer."""
    results = retrieve(question, index)


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