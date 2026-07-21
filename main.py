from rag import build_index
from agent import run_agent
from config import DOCS_FOLDER, COMPANY_NAME

SHOW_TOOL_CALLS = True    


def main():
    print(f"Loading {COMPANY_NAME}'s help documents...")
    index = build_index()

    if not index:
        print(f"No documents found. Put .txt files in the '{DOCS_FOLDER}' folder and try again.")
        return

    print(f"Ready! Indexed {len(index)} chunks. Ask a question (or type 'quit').\n")

    history = []         

    while True:
        question = input("Customer: ")
        if question.lower() in ("quit", "exit"):
            break
        if not question.strip():
            continue

        reply, history = run_agent(question, index, history=history, verbose=SHOW_TOOL_CALLS)
        print(f"\nAssistant: {reply}\n")


if __name__ == "__main__":
    main()