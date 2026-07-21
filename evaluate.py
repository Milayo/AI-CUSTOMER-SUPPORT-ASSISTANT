from pydantic import BaseModel, Field

from config import client, CHAT_MODEL, CHUNK_SIZE, TOP_K, SCORE_THRESHOLD
from rag import build_index, retrieve, generate_answer
from test_set import TEST_SET



class Judgement(BaseModel):
    declined: bool = Field(description="True if the assistant declined to answer or offered to escalate to a human, rather than giving a substantive answer.")
    correct: bool = Field(description="True if the assistant's answer is factually consistent with the reference answer. If the assistant declined, set this to False.")
    reason: str = Field(description="One short sentence explaining the judgement.")


def judge(question, reference, actual):
    """Use an LLM to grade an answer. Returns a Judgement object."""
    system = (
        "You grade a customer-support assistant's answers. "
        "Judge meaning, not wording. Be strict about factual accuracy."
    )
    user = (
        f"Customer question: {question}\n\n"
        f"Reference answer: {reference}\n\n"
        f"Assistant's answer: {actual}"
    )
    completion = client.chat.completions.parse(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format=Judgement,
    )
    return completion.choices[0].message.parsed


# ---------- deterministic refusal detection ----------
REFUSAL_MARKERS = [
    "don't have that information",
    "do not have that information",
    "connect you with a human",
    "human agent",
    "i don't know",
    "support ticket",
]


def looks_like_refusal(text):
    """Deterministic refusal check - more reliable than asking the judge."""
    t = text.lower()
    return any(m in t for m in REFUSAL_MARKERS)


def run_evaluation(test_set=TEST_SET):
    print("Building index...")
    index = build_index()
    if not index:
        print("No documents found. Add .txt files to the docs folder.")
        return

    print(f"Indexed {len(index)} chunks.")
    print(f"Settings: CHUNK_SIZE={CHUNK_SIZE}, TOP_K={TOP_K}, THRESHOLD={SCORE_THRESHOLD}")
    print(f"Running {len(test_set)} test cases...\n")

    answerable = [c for c in test_set if c["should_answer"]]
    unanswerable = [c for c in test_set if not c["should_answer"]]

    retrieval_hits = 0
    answers_correct = 0
    refusals_correct = 0
    failures = []

    for case in test_set:
        question = case["question"]
        results = retrieve(question, index)
        retrieved_text = " ".join(r["text"] for _, r in results).lower()
        top_score = results[0][0] if results else 0.0

        actual = generate_answer(question, results)
        verdict = judge(question, case["expected_answer"], actual)

        # deterministic check first, judge as backup
        declined = looks_like_refusal(actual) or verdict.declined

        if case["should_answer"]:
            needle = (case["context_must_contain"] or "").lower()
            got_context = needle in retrieved_text
            retrieval_hits += got_context

            ok = verdict.correct and not declined
            answers_correct += ok

            mark = "PASS" if ok else "FAIL"
            print(f"[{mark}] {question}")
            print(f"       top_score={top_score:.3f}  context_found={got_context}  declined={declined}")
            if not ok:
                failures.append((question, "answer", verdict.reason, got_context, top_score))
        else:
            ok = declined
            refusals_correct += ok

            mark = "PASS" if ok else "FAIL"
            print(f"[{mark}] {question}   (should decline)")
            print(f"       top_score={top_score:.3f}  declined={declined}")
            if not ok:
                failures.append((question, "refusal", verdict.reason, None, top_score))

    # ---------- report ----------
    n_ans = len(answerable)
    n_unans = len(unanswerable)

    print("\n" + "=" * 55)
    print("RESULTS")
    print("=" * 55)
    print(f"Context recall (retrieval):  {retrieval_hits}/{n_ans}   {retrieval_hits/n_ans:.0%}")
    print(f"Answer correctness:          {answers_correct}/{n_ans}   {answers_correct/n_ans:.0%}")
    print(f"Refusal accuracy:            {refusals_correct}/{n_unans}   {refusals_correct/n_unans:.0%}")
    overall = (answers_correct + refusals_correct) / len(test_set)
    print(f"Overall:                     {overall:.0%}")

    if failures:
        print("\nFAILURES (investigate these):")
        for q, kind, reason, got_context, score in failures:
            if kind == "answer" and got_context is False:
                diagnosis = "RETRIEVAL problem - needed fact was not retrieved"
            elif kind == "answer":
                diagnosis = "GENERATION problem - context was there, answer still wrong"
            else:
                diagnosis = "GROUNDING problem - answered something it should have declined"
            print(f"  - {q}")
            print(f"      {diagnosis} (top_score={score:.3f})")
            print(f"      judge: {reason}")
    print()


if __name__ == "__main__":
    run_evaluation()