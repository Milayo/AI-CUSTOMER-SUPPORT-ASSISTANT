import json

from config import client, CHAT_MODEL, COMPANY_NAME
from tools import TOOL_SCHEMAS, search_help_docs, create_support_ticket

MAX_STEPS = 5       

SYSTEM_PROMPT = f"""You are a friendly customer-support assistant for {COMPANY_NAME}.

How to handle every customer message:
1. For any question about shipping, returns, refunds, payments, orders, tracking,
   or accounts, call search_help_docs FIRST.
2. Answer using ONLY the information the search returns. Never use outside
   knowledge and never guess. Cite the source numbers you used, like [1].
3. If the search returns NO_RELEVANT_DOCS, or the returned text does not actually
   contain the answer, tell the customer you don't have that information and call
   create_support_ticket so a human can help.
4. If the customer explicitly asks for a human, call create_support_ticket.
5. Be concise, warm, and clear.
"""


def run_agent(user_message, index, history=None, verbose=True, on_tool_call=None):
    """Run the agent for one customer message. Returns (reply, updated_history).

    on_tool_call: optional callback(name, args, result) fired after each tool
    runs. The CLI leaves it as None; the Streamlit app uses it to display
    what the assistant did."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    for step in range(MAX_STEPS):
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
        )
        msg = response.choices[0].message

        # No tool requested -> the model has its final answer.
        if not msg.tool_calls:
            messages.append({"role": "assistant", "content": msg.content})
            new_history = [m for m in messages[1:]]      # drop the system prompt
            return msg.content, new_history

        # The model requested one or more tools. Record the request...
        messages.append(msg)

        # ...then execute each one and feed the result back.
        for tool_call in msg.tool_calls:
            name = tool_call.function.name
            try:
                args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                result = "Error: could not parse tool arguments."
            else:
                if verbose:
                    print(f"   [tool] {name}({args})")
                if name == "search_help_docs":
                    result = search_help_docs(args["query"], index)
                elif name == "create_support_ticket":
                    result = create_support_ticket(
                        summary=args.get("summary", ""),
                        customer_question=args.get("customer_question", user_message),
                        customer_email=args.get("customer_email", ""),
                    )
                else:
                    result = f"Error: unknown tool '{name}'."

            if on_tool_call:
                on_tool_call(name, args if isinstance(args, dict) else {}, str(result))

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result),
            })

    return ("I'm having trouble processing that request. Let me connect you with a human agent.",
            history or [])