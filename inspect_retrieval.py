from rag import build_index, retrieve
from config import DOCS_FOLDER, TOP_K, SCORE_THRESHOLD, CHUNK_SIZE


def main():
    print("Building index...")
    index = build_index()
    if not index:
        print(f"No documents found in '{DOCS_FOLDER}'. Add .txt files there.")
        return

    print(f"Indexed {len(index)} chunks.")
    print(f"Current settings -> CHUNK_SIZE={CHUNK_SIZE}, TOP_K={TOP_K}, THRESHOLD={SCORE_THRESHOLD}\n")
    print("Type a question to see what gets retrieved (or 'quit').\n")

    while True:
        question = input("Question: ")
        if question.lower() in ("quit", "exit"):
            break
        if not question.strip():
            continue

        results = retrieve(question, index)
        top_score = results[0][0] if results else 0.0
        verdict = "OK" if top_score >= SCORE_THRESHOLD else "BELOW THRESHOLD -> would escalate"
        print(f"\nTop score: {top_score:.3f}   ({verdict})")

        for i, (score, r) in enumerate(results, 1):
            preview = r["text"][:130].replace("\n", " ")
            print(f"  [{i}] {score:.3f}  ({r['source']})  {preview}...")
        print()


if __name__ == "__main__":
    main()