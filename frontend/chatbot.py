# chatbot.py
import os
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
load_dotenv(find_dotenv(), override=False)
# Try env var first, then (optionally) Streamlit secrets
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    try:
        import streamlit as st
        api_key = st.secrets.get("OPENAI_API_KEY")
    except Exception:
        pass

if not api_key:
    raise RuntimeError(
        "OPENAI_API_KEY not found. Create a .env in the repo root or frontend/ "
        "with OPENAI_API_KEY=sk-..., or set it in Streamlit secrets."
    )

client = OpenAI(api_key=api_key)

SYSTEM_PROMPT = """
You are Deloitte Skills Finder, an internal AI assistant designed to help managers quickly and accurately
find the best-suited employees for projects by using the latest One-Pager documents.

Your primary goal is to provide precise, professional, and concise answers based solely on the retrieved documents.

GUIDELINES:
- Accuracy first: base factual answers on retrieved documents.
- Professional tone; concise; use bullet points for lists.
- Formatting: label Name, Skills, Experience, Clients clearly.
- Missing data: say what's missing. Never invent.

MEMORY POLICY:
- You MAY use chat_history to resolve pronouns and follow-ups (e.g., “they”, “this candidate”, “the previous list”).
- You MAY answer meta questions about the conversation itself (e.g., “what was my first question?”, “repeat my last message”)
  using chat_history even if documents don’t contain that information.
- For all employee facts (skills, experience, certifications, clients), still rely on the retrieved documents only.

Answer clearly and directly.
""".strip()

def chat_once(user_question: str, context: str = "") -> str:
    """
    Stateless, single-turn chat call.
    If you later add RAG, pass the retrieved JSON/text as `context`.
    """
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,  # keep deterministic for accuracy
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context from documents:\n{context}\n\nUser question:\n{user_question}\n\nAnswer:"
            }
        ],
    )
    return resp.choices[0].message.content
