import streamlit as st

from rag import build_index
from agent import run_agent
from config import COMPANY_NAME, DOCS_FOLDER

EXAMPLE_QUESTIONS = [
    "Where's my package?",
    "Can I return a swimsuit?",
    "How long does delivery to Nigeria take?",
]

st.set_page_config(page_title=f"{COMPANY_NAME} Support", page_icon="💬")
st.title(f"💬 {COMPANY_NAME} Support Assistant")
st.caption(
    "Ask about shipping, returns, refunds, payments, or your account. "
    "Answers come only from our help documentation, with sources shown. "
    "If I can't help, I'll create a support ticket for a human agent."
)



@st.cache_resource(show_spinner="Loading help documents...")
def get_index():
    return build_index()


index = get_index()

if not index:
    st.error(f"No documents found. Add .txt files to the '{DOCS_FOLDER}' folder.")
    st.stop()

with st.sidebar:
    st.subheader("About")
    st.write(
        "This is ACME's assistant: it searches "
        "the help docs, answers only from what it finds, and escalates to a "
        "human support agent when the docs don't cover the question."
    )
    if st.button("Clear conversation"):
        st.session_state.display_messages = []
        st.session_state.agent_history = []
        st.rerun()



if "display_messages" not in st.session_state:
    st.session_state.display_messages = []    
if "agent_history" not in st.session_state:
    st.session_state.agent_history = []        


for msg in st.session_state.display_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("trace"):
            with st.expander("What I did"):
                for name, args, result in msg["trace"]:
                    st.markdown(f"**Tool called:** `{name}`")
                    st.caption(f"Input: {args}")
                    st.code(result[:1500], language=None)



prefill = None
if not st.session_state.display_messages:
    st.write("**Try one of these:**")
    cols = st.columns(len(EXAMPLE_QUESTIONS))
    for col, example in zip(cols, EXAMPLE_QUESTIONS):
        if col.button(example, use_container_width=True):
            prefill = example



question = st.chat_input("Ask a question...") or prefill

if question:
    st.session_state.display_messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    trace = []

    def record_tool(name, args, result):
        trace.append((name, args, result))

    with st.chat_message("assistant"):
        with st.spinner("Searching the help docs..."):
            try:
                reply, st.session_state.agent_history = run_agent(
                    question,
                    index,
                    history=st.session_state.agent_history,
                    verbose=False,
                    on_tool_call=record_tool,
                )
            except Exception as e:
                reply = "Sorry, something went wrong on my end. Please try again."
                st.error(f"Error: {e}")

        st.markdown(reply)

        if trace:
            with st.expander("What I did"):
                for name, args, result in trace:
                    st.markdown(f"**Tool called:** `{name}`")
                    st.caption(f"Input: {args}")
                    st.code(result[:1500], language=None)

    st.session_state.display_messages.append(
        {"role": "assistant", "content": reply, "trace": trace}
    )