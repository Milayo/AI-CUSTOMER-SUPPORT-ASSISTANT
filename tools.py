import json
import os
from datetime import datetime

from rag import retrieve
from config import SCORE_THRESHOLD, COMPANY_NAME

TICKETS_FILE = "tickets.json"


# ---------- TOOL 1: search the help docs ----------
def search_help_docs(query: str, index) -> str:
    """Retrieve relevant help-doc passages. Returns them as text for the model."""
    results = retrieve(query, index)

    # Drop weak matches so we don't feed noise into the prompt.
    results = [(s, r) for s, r in results if s >= SCORE_THRESHOLD * 0.8]

    if not results or results[0][0] < SCORE_THRESHOLD:
        return "NO_RELEVANT_DOCS: nothing in the help documents covers this question."

    return "\n\n".join(
        f"[{i}] (from {r['source']}, relevance {score:.2f})\n{r['text']}"
        for i, (score, r) in enumerate(results, 1)
    )


# ---------- TOOL 2: escalate to a human ----------
def create_support_ticket(summary: str, customer_question: str, customer_email: str = "") -> str:
    """Create a support ticket for a human agent to follow up on.
    In a real system this would call a helpdesk API (Zendesk, Freshdesk, etc.).
    Here we append it to a local JSON file so the demo is self-contained."""
    tickets = []
    if os.path.exists(TICKETS_FILE):
        try:
            with open(TICKETS_FILE, "r", encoding="utf-8") as f:
                tickets = json.load(f)
        except (json.JSONDecodeError, ValueError):
            tickets = []

    ticket = {
        "id": f"TICKET-{1000 + len(tickets) + 1}",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "summary": summary,
        "customer_question": customer_question,
        "customer_email": customer_email or "not provided",
        "status": "open",
    }
    tickets.append(ticket)

    with open(TICKETS_FILE, "w", encoding="utf-8") as f:
        json.dump(tickets, f, indent=2)

    return (f"Ticket {ticket['id']} created successfully. "
            f"A {COMPANY_NAME} agent will follow up within 1-2 business days.")


# ---------- SCHEMAS: how the tools are described to the model ----------
# NOTE: this nested {"type": "function", "function": {...}} shape is the
# Chat Completions format. The Responses API uses a flatter shape.
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_help_docs",
            "description": (
                "Search the company's help documentation for information needed to answer "
                "a customer question. Use this FIRST for any question about shipping, returns, "
                "refunds, payments, orders, tracking, or account issues."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query, phrased as a clear standalone question.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_support_ticket",
            "description": (
                "Create a support ticket so a human agent can follow up. Use this ONLY when "
                "the help documents do not contain the answer, or the customer explicitly asks "
                "for a human. Do not use it for questions the documentation already answers."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "A short summary of what the customer needs help with.",
                    },
                    "customer_question": {
                        "type": "string",
                        "description": "The customer's original question, verbatim.",
                    },
                    "customer_email": {
                        "type": "string",
                        "description": "The customer's email address, if they provided one.",
                    },
                },
                "required": ["summary", "customer_question"],
            },
        },
    },
]