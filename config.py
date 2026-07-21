import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()                       # local: read .env

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:                     # deployed: fall back to Streamlit secrets
    try:
        import streamlit as st
        api_key = st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass

if not api_key:
    raise RuntimeError(
        "No OPENAI_API_KEY found. Set it in a .env file locally, "
        "or in Streamlit secrets when deployed."
    )

client = OpenAI(api_key=api_key)

# --- Models ---
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-5.5"              # use a cheaper mini model to reduce cost

# --- Retrieval settings ---
CHUNK_SIZE = 400
OVERLAP_SENTENCES = 1
TOP_K = 5
SCORE_THRESHOLD = 0.40

# --- Data + identity ---
DOCS_FOLDER = "docs"
COMPANY_NAME = "Acme Store"