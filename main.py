
from rag import build_index, answer_question
from config import DOCS_FOLDER, COMPANY_NAME


def main():
    print(f"Loading {COMPANY_NAME}'s help documents...")
    index = build_index()

    if not index:
        print(f"No documents found. Put .txt files in the '{DOCS_FOLDER}' folder and try again.")
        return

    print(f"Ready! Indexed {len(index)} chunks. Ask a question (or type 'quit').\n")

    while True:
        question = input("Customer: ")
        if question.lower() in ("quit", "exit"):
            break
        if not question.strip():
            continue
        print("\nAssistant:", answer_question(question, index), "\n")


if __name__ == "__main__":
    main()