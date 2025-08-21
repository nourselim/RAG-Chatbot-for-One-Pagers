import streamlit as st
import time
from datetime import datetime
from pathlib import Path
import sys
import os
import re

# Add the rag folder to the path so we can import from it
rag_path = Path(__file__).parent.parent / "rag"
sys.path.append(str(rag_path))

from faiss_service import FaissCandidateSearch
from embed_only import OUT_DIR, EMB_NPY, META_JSONL

# Add OpenAI client for history-aware rewrite
from openai import OpenAI

# Page configuration
st.set_page_config(
    page_title="DeBotte AI - Employee Skills Finder", 
    page_icon="🤖", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Load environment variables
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=False)

# Check for OpenAI API key
if not os.getenv("OPENAI_API_KEY"):
    st.error("❌ OPENAI_API_KEY not found. Please set it in your .env file.")
    st.stop()

# Initialize OpenAI client (for query rewriting)
_oa_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize FAISS service
@st.cache_resource(show_spinner=False)
def load_faiss_service():
    """Load the FAISS service for candidate search."""
    try:
        if not (OUT_DIR / "faiss_index.bin").exists():
            st.warning("⚠️ FAISS index not found. Please run the RAG pipeline first.")
            st.info("💡 Run this command in the rag/ folder: `python main.py auto`")
            return None
        
        service = FaissCandidateSearch(OUT_DIR)
        service.load_index()
        return service
    except Exception as e:
        st.error(f"❌ Error loading FAISS service: {str(e)}")
        return None

faiss_service = load_faiss_service()

# ----------------- History-aware helpers -----------------
def _recent_candidates_from_history(messages):
    """Return the most recent non-empty assistant candidates list from history, if present."""
    for m in reversed(messages or []):
        if (
            isinstance(m, dict)
            and m.get("role") == "assistant"
            and m.get("candidates")
            and isinstance(m.get("candidates"), list)
            and len(m.get("candidates")) > 0
        ):
            return m.get("candidates")
    return []

def _resolved_names_from_history(messages) -> list[str]:
    """Return a list of names that were explicitly discussed in assistant follow-ups (most recent first)."""
    names: list[str] = []
    seen = set()
    for m in reversed(messages or []):
        if isinstance(m, dict) and m.get("role") == "assistant":
            for nm in m.get("resolved_names", []) or []:
                if nm and nm not in seen:
                    seen.add(nm)
                    names.append(nm)
    return names

def _extract_target_names_from_prompt(prompt: str, messages):
    """Resolve references (names, pronouns, ordinals) to candidate names using LLM with history context.

    Returns a list of zero or more names exactly as they appear in the recent candidate list.
    Falls back to simple name substring matching if the model does not return valid names.
    """
    recent_cands = _recent_candidates_from_history(messages)
    cand_names = [c.get("name", "").strip() for c in recent_cands if c.get("name")]
    prompt_text = (prompt or "").strip()

    if cand_names and prompt_text:
        sys_prompt = (
            "You are resolving references in a chat about employees. Given the user's latest message and "
            "the list of recently mentioned candidate names, return a comma-separated list of the names "
            "from that list that the user is referring to (0-3 names). Consider pronouns, ordinals "
            "(first/second/third), and phrases like 'another one'. Prefer names that have NOT been already discussed "
            "if the user implies 'another'. Return ONLY the names that appear "
            "in the provided list, exactly as written, comma-separated. Return empty if none."
            "if you are not sure, ask the user to clarify"
            "if the user is asking about a specific employee, return the name of the employee"
            "if the user is asking about a specific skill, return the name of the skill"
            "if the user is asking about a specific cloud platform, return the name of the cloud platform"
            "if the user writes an incomplete question, ask the user to clarify"
        )
        try:
            resp = _oa_client.chat.completions.create(
                model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "system", "content": "Candidates: " + ", ".join(cand_names)},
                    {"role": "system", "content": "Already discussed: " + ", ".join(_resolved_names_from_history(messages))},
                    {"role": "user", "content": prompt_text},
                ],
                temperature=0,
                max_tokens=50,
            )
            raw = (resp.choices[0].message.content or "").strip()
            if raw:
                # Parse comma/line separated
                parts = [p.strip() for p in re.split(r"[,\n]", raw) if p.strip()]
                resolved = [p for p in parts if p in cand_names]
                if resolved:
                    return resolved
        except Exception:
            pass

    # Fallback: lightweight substring token matching within recent candidates
    out = []
    prompt_low = prompt_text.lower()
    for nm in cand_names:
        nm_low = nm.lower()
        if nm_low and (nm_low in prompt_low or any(tok and len(tok) >= 3 and re.search(rf"\b{re.escape(tok)}\b", prompt_low) for tok in nm_low.split())):
            out.append(nm)
    # Deduplicate
    seen = set(); deduped = []
    for n in out:
        if n not in seen:
            seen.add(n); deduped.append(n)
    if deduped:
        return deduped

    # Last resort: try resolving names against all known employees (if loaded)
    try:
        if faiss_service and getattr(faiss_service, "metas", None):
            all_names = []
            seen = set()
            for m in faiss_service.metas:
                nm = (m.get("employee_name") or "").strip()
                if nm and nm.lower() not in seen:
                    seen.add(nm.lower())
                    all_names.append(nm)
            if all_names:
                sys_prompt = (
                    "From the user's message, select 0-3 names that appear in the provided candidate list. "
                    "Return ONLY the names exactly as written, comma-separated."
                )
                resp = _oa_client.chat.completions.create(
                    model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "system", "content": "Candidates: " + ", ".join(all_names)},
                        {"role": "user", "content": prompt_text},
                    ],
                    temperature=0,
                    max_tokens=60,
                )
                raw = (resp.choices[0].message.content or "").strip()
                parts = [p.strip() for p in re.split(r"[,\n]", raw) if p.strip()]
                resolved = [p for p in parts if p in all_names]
                # Deduplicate while preserving order
                seen = set(); final = []
                for n in resolved:
                    if n not in seen:
                        seen.add(n); final.append(n)
                return final
    except Exception:
        pass

    return []


def _extract_topk_from_prompt(user_text: str) -> int | None:
    """Ask the model to extract a desired top-K count (1..50) from the user's request.

    Returns None if no explicit count is found.
    """
    text = (user_text or "").strip()
    if not text:
        return None
    sys_prompt = (
        "If the user requests a number of results (e.g., 'top 4 candidates'), return that integer only. "
        "Otherwise return the word NONE. Valid range: 1..50."
    )
    try:
        resp = _oa_client.chat.completions.create(
            model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": text},
            ],
            temperature=0,
            max_tokens=4,
        )
        val = (resp.choices[0].message.content or "").strip().upper()
        if val == "NONE":
            return None
        try:
            k = int(re.sub(r"[^0-9]", "", val))
            if 1 <= k <= 50:
                return k
        except Exception:
            return None
    except Exception:
        return None
    return None

def _rewrite_query_with_history(messages, current_question: str) -> str:
    """Use OpenAI to rewrite the user's question into a standalone search query using recent history."""
    # Gather recent context (limit to last 8 turns for brevity)
    recent_history = messages[-8:] if messages else []
    recent_cands = _recent_candidates_from_history(recent_history)
    cand_names_str = ", ".join([c.get("name", "") for c in recent_cands if c.get("name")])

    sys_prompt = (
        "Rewrite the user's latest message into a concise standalone search query about employees, skills, "
        "experience, certifications or clients. Include any relevant names and constraints from the recent "
        "conversation. If pronouns are used (they, them, first one), resolve them to the most likely name(s). "
        "Keep it under 25 words. Return ONLY the rewritten query."
    )

    messages_payload = [{"role": "system", "content": sys_prompt}]
    if cand_names_str:
        messages_payload.append({"role": "system", "content": f"Recent candidates mentioned: {cand_names_str}"})
    # Add trimmed history
    for m in recent_history:
        role = m.get("role")
        content = m.get("content", "")
        if role in ("user", "assistant") and content:
            messages_payload.append({"role": role, "content": content[:2000]})
    # Add the current question as the final user message
    messages_payload.append({"role": "user", "content": current_question})

    try:
        resp = _oa_client.chat.completions.create(
            model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            messages=messages_payload,
            temperature=0,
            max_tokens=100,
        )
        rewritten = (resp.choices[0].message.content or "").strip()
        # Fallback sanity
        if not rewritten:
            return current_question
        return rewritten
    except Exception:
        return current_question

# ----------------- Synthesis helper -----------------
def _synthesize_employee_narrative(employee_name: str, chunks: list[dict], user_question: str) -> str:
    """Use the chat model to produce a friendly, concise paragraph about the employee based on chunks."""
    if not chunks:
        return f"I couldn't find additional details for {employee_name}."

    # Prepare a compact context string
    selected = []
    for c in chunks[:6]:
        ct = c.get("chunk_type", "unknown")
        tx = (c.get("text", "") or "").strip()
        if tx:
            selected.append(f"[{ct}] {tx}")
    context = "\n\n".join(selected)

    sys_prompt = (
        "You are a helpful assistant. Given context snippets about an employee, answer the user's request "
        "in a friendly, concise paragraph using complete sentences. Prefer prose over bullet points. "
        "Only use facts present in the provided context. If a detail is not present, say so briefly."
        "If the user's query is an incomplete sentence, ask for clarity."
        "If the question is unclear, ask for clarity."
    )
    messages_payload = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": f"Context about {employee_name}:\n{context}\n\nUser request: {user_question}"},
    ]

    try:
        resp = _oa_client.chat.completions.create(
            model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            messages=messages_payload,
            temperature=0.2,
            max_tokens=350,
        )
        out = (resp.choices[0].message.content or "").strip()
        return out or f"Here are the available details about {employee_name}."
    except Exception:
        # Fallback: join snippets
        return f"Here are the available details about {employee_name}:\n\n" + context

def _synthesize_comparison(name_a: str, chunks_a: list[dict], name_b: str, chunks_b: list[dict], user_question: str) -> str:
    """Produce a concise, friendly comparison between two employees using provided chunks only."""
    def mk_ctx(name: str, chs: list[dict]) -> str:
        lines = []
        for c in chs[:6]:
            ct = c.get("chunk_type", "unknown")
            tx = (c.get("text", "") or "").strip()
            if tx:
                lines.append(f"[{ct}] {tx}")
        return "\n\n".join(lines)

    ctx_a = mk_ctx(name_a, chunks_a)
    ctx_b = mk_ctx(name_b, chunks_b)

    sys_prompt = (
        "You are a helpful assistant. Compare two employees strictly based on the provided contexts. "
        "Write a short, natural paragraph that answers the user's request (e.g., focus on Kubernetes if asked). "
        "Mention strengths or relevant evidence for each and note any missing information. Keep it crisp and helpful."
        "If the user's query is an incomplete sentence, ask for clarity."
        "If the question is unclear, ask for clarity."
    )
    messages_payload = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": f"User request: {user_question}\n\nEmployee A: {name_a}\n{ctx_a}\n\nEmployee B: {name_b}\n{ctx_b}"},
    ]

    try:
        resp = _oa_client.chat.completions.create(
            model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            messages=messages_payload,
            temperature=0.2,
            max_tokens=350,
        )
        out = (resp.choices[0].message.content or "").strip()
        return out or f"Here’s a brief comparison between {name_a} and {name_b} based on the available details."
    except Exception:
        return f"Here’s a brief comparison between {name_a} and {name_b} based on the available details."

def _last_candidate_names(messages, limit: int = 2) -> list[str]:
    """Get up to `limit` names from the most recent assistant candidates list."""
    cands = _recent_candidates_from_history(messages)
    names = []
    for c in cands:
        nm = c.get("name")
        if nm and nm not in names:
            names.append(nm)
        if len(names) >= limit:
            break
    return names

# ----------------- CSS Styling -----------------
st.markdown("""
<style>
    /* Main container styling */
    .main { padding: 0; }
    .chat-container { max-width: 800px; margin: 0 auto; padding: 20px; }
    .user-message { background-color: #40414f; color: white; padding: 15px; border-radius: 10px; margin: 10px 0; margin-left: 50px; margin-right: 0; }
    .assistant-message { background-color: #444654; color: white; padding: 15px; border-radius: 10px; margin: 10px 0; margin-right: 50px; margin-left: 0; }
    .stTextInput > div > div > input { background-color: #40414f; color: white; border: 1px solid #565869; border-radius: 10px; padding: 15px; font-size: 16px; }
    .css-1d391kg { background-color: black; }
    .stButton > button { background-color: #26890d; color: white; border: none; border-radius: 5px; padding: 10px 20px; font-size: 14px; cursor: pointer; }
    section[data-testid="stSidebar"] { background-color: #000000 !important; }
    section[data-testid="stSidebar"] .css-1d391kg { background-color: #000000 !important; }
    section[data-testid="stSidebar"] * { color: white !important; }
    .stButton > button:hover { background-color: #1f6d0b; }
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: #2d2d2d; }
    ::-webkit-scrollbar-thumb { background: #888; border-radius: 4px; }
    .chat-history { background-color: #202123; padding: 10px; border-radius: 5px; margin-bottom: 10px; }
    .model-selector { background-color: #202123; color: white; border: 1px solid #565869; border-radius: 5px; padding: 5px; }
    .candidate-result { background-color: #2d2d2d; border: 1px solid #565869; border-radius: 8px; padding: 15px; margin: 10px 0; }
    .candidate-name { font-size: 18px; font-weight: bold; color: #10a37f; margin-bottom: 5px; }
    .candidate-details { color: #cccccc; font-size: 14px; margin-bottom: 10px; }
    .candidate-text { color: #ffffff; font-size: 13px; line-height: 1.4; }
</style>
""", unsafe_allow_html=True)

# ----------------- State Management -----------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {}
if "current_session" not in st.session_state:
    st.session_state.current_session = "default"

# ----------------- Sidebar -----------------
with st.sidebar:
    st.title("🤖 DeBotte AI")

# Just shrink the first title on the page
    st.markdown("""
    <style>
    h1:first-of-type {
      font-size: 2.rem !important;   /* try 1.75rem–2.25rem to taste */
      line-height: 1.2 !important;
      margin: 0 0 .5rem 0 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("**Employee Skills Finder**")
    st.markdown(
    "<div style='height:2px; background:#ffffff; margin:0.75rem 0;'></div>",
    unsafe_allow_html=True
)
    
    # New Chat Button
    if st.button("New Chat", use_container_width=True):
        new_session_id = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        st.session_state.chat_sessions[new_session_id] = []
        st.session_state.current_session = new_session_id
        st.session_state.messages = []
        st.rerun()
    
    st.markdown(
    "<div style='height:2px; background:#ffffff; margin:0.75rem 0;'></div>",
    unsafe_allow_html=True
)
    
    # Recent Chats
    st.subheader("Recent Chats")
    for session_id in list(st.session_state.chat_sessions.keys()):
        if st.button(f"💬 Chat {session_id[-6:]}", key=session_id, use_container_width=True):
            st.session_state.current_session = session_id
            st.session_state.messages = st.session_state.chat_sessions[session_id]
            st.rerun()
    
    # Clear All Chats
    if st.button("Clear All Chats", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_sessions = {}
        st.session_state.current_session = "default"
        st.rerun()
    
#     st.markdown(
#     "<div style='height:2px; background:#ffffff; margin:0.75rem 0;'></div>",
#     unsafe_allow_html=True
# )
    
    # # System Info
    # st.subheader("System Status")
    # if faiss_service:
    #     st.success("✅ FAISS Index Loaded")
    #     st.info(f"📊 Ready to search")
    # else:
    #     st.error("❌ FAISS Index Not Available")
    #     st.info("💡 Run: `cd rag && python main.py auto`")
    
    st.markdown(
    "<div style='height:2px; background:#ffffff; margin:0.75rem 0;'></div>",
    unsafe_allow_html=True
)
    st.markdown("**Instructions:**")
    st.markdown(
        """
        1. Ask about employee skills, experience, or certifications
        2. Use natural language queries
        3. Example: "Find employees with AWS certification"
        4. Example: "Who has experience with SAP?"
        """
    )

# ----------------- Main Chat Interface -----------------
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Display chat messages (render message content only; no candidate cards)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ----------------- Intent classification helpers -----------------
def _classify_intent_llm(text: str) -> str:
    """Use the chat model to classify the user's intent robustly (supports non-English greetings)."""
    cleaned = (text or "").strip()
    if not cleaned:
        return "empty"

    sys_prompt = (
        "Classify the user's message into exactly one of these labels: "
        "greeting, compare, followup_detail, search. "
        "Return only the label. Use 'greeting' for salutations/small talk without an info request; "
        "'compare' if they ask to compare two people; 'followup_detail' if they ask for more details "
        "about a previously mentioned candidate; otherwise 'search'."
        "If the user's query is an incomplete sentence, ask for clarity."
        "If the question is unclear, ask for clarity."
    )
    try:
        resp = _oa_client.chat.completions.create(
            model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": cleaned},
            ],
            temperature=0,
            max_tokens=5,
        )
        label = (resp.choices[0].message.content or "").strip().lower()
        if label in ("greeting", "compare", "followup_detail", "search"):
            return label
        return "search"
    except Exception:
        return "search"


def _classify_intent(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return "empty"
    # Delegate intent recognition fully to the model (supports multilingual and flexible phrasing)
    return _classify_intent_llm(t)


def _generate_greeting_response(user_text: str) -> str:
    """Generate a short greeting in the user's language with a brief capability hint."""
    sys_prompt = (
        "You are a brief, friendly assistant for an employee skills finder chatbot. "
        "Respond to the user's greeting in the same language. In one or two short sentences, "
        "say you can help find employees by skills, experience, certifications, or clients, and give "
        "one concise example query. Keep it under 30 words."
        "If the user's query is an incomplete sentence, ask for clarity."
        "If the question is unclear, ask for clarity."
    )
    try:
        resp = _oa_client.chat.completions.create(
            model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_text},
            ],
            temperature=0.3,
            max_tokens=80,
        )
        out = (resp.choices[0].message.content or "").strip()
        if out:
            return out
    except Exception:
        pass
    return (
        "Hello! I’m your Deloitte Skills Finder. Ask me about skills, experience, certifications, or clients "
        "(e.g., ‘Who has Kubernetes experience?’)."
    )


def _generate_clarifying_question(messages, user_text: str) -> str:
    """Ask one concise clarifying question when the reference is ambiguous.

    Uses the recent candidate names to frame the question naturally and avoids hard-coded wording.
    """
    recent_cands = _recent_candidates_from_history(messages)
    name_list = ", ".join([c.get("name", "") for c in recent_cands if c.get("name")])
    sys_prompt = (
        "You are assisting with an employee skills finder. The user's last message is ambiguous. "
        "Ask ONE short clarifying question to identify which person or topic they mean. "
        "If a list of recent candidate names is provided, include 2–4 of them as options in the question. "
        "Keep it under 20 words."
        "If the user's query is an incomplete sentence, ask for clarity."
        "If the question is unclear, ask for clarity."
    )
    user_payload = f"User said: {user_text}\nRecent candidates: {name_list}" if name_list else f"User said: {user_text}"
    try:
        resp = _oa_client.chat.completions.create(
            model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_payload},
            ],
            temperature=0.2,
            max_tokens=60,
        )
        q = (resp.choices[0].message.content or "").strip()
        if q:
            return q
    except Exception:
        pass
    return "Which person do you mean?"


def _assess_query_readiness(messages, user_text: str, rewritten_text: str) -> tuple[bool, str]:
    """Decide if the message is clear enough to run retrieval, else produce one clarifying question.

    Returns (needs_clarification, clarifying_question).
    """
    recent_cands = _recent_candidates_from_history(messages)
    name_list = ", ".join([c.get("name", "") for c in recent_cands if c.get("name")])
    sys_prompt = (
        "You help decide if a user's message is ready for an employee search. "
        "If the request is under-specified, unrelated to employees, or incomplete, reply with 'CLARIFY: <single short question>'. "
        "If it's clear enough to search/compare, reply 'READY'. Use up to one short question."
    )
    payload = [
        {"role": "system", "content": sys_prompt},
        {"role": "system", "content": f"Recent candidates: {name_list}"},
        {"role": "user", "content": f"User: {user_text}\nRewritten: {rewritten_text}"},
    ]
    try:
        resp = _oa_client.chat.completions.create(
            model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            messages=payload,
            temperature=0,
            max_tokens=60,
        )
        out = (resp.choices[0].message.content or "").strip()
        low = out.lower()
        if low.startswith("ready"):
            return (False, "")
        if low.startswith("clarify:"):
            q = out.split(":", 1)[1].strip()
            return (True, q or _generate_clarifying_question(messages, user_text))
    except Exception:
        pass
    return (False, "")

# Chat input
if prompt := st.chat_input("Ask about employee skills, experience, or certifications..."):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        if faiss_service is None:
            response = "❌ FAISS service is not available. Please run the RAG pipeline first."
            candidates = []
        else:
            with st.spinner("🔍 Thinking..."):
                try:
                    intent = _classify_intent(prompt)
                    # Greetings / smalltalk → friendly response, no retrieval
                    if intent == "greeting":
                        response = _generate_greeting_response(prompt)
                        candidates = []
                    else:
                        resolved_names_for_msg = []
                        # 1) Rewrite the query using chat history
                        rewritten = _rewrite_query_with_history(st.session_state.messages, prompt)
                        # 2) Resolve any explicit candidate references from last turn
                        target_names = _extract_target_names_from_prompt(prompt, st.session_state.messages)

                        lowq = prompt.lower()
                        desired_types = []
                        if any(k in lowq for k in ["skill", "skills", "business skills", "technology skills", "industry"]):
                            desired_types.append("skills")
                        if "cert" in lowq or "certification" in lowq:
                            desired_types.append("certifications")
                        if "summary" in lowq:
                            desired_types.append("summary")
                        if "experience" in lowq or "role" in lowq or "company" in lowq:
                            desired_types.append("experience")
                        if "client" in lowq:
                            desired_types.append("clients")

                        # Follow-up detail → narrative for a specific candidate
                        if intent == "followup_detail":
                            # Prefer explicit reference; otherwise try resolving from text against FAISS metadata
                            if target_names:
                                name = target_names[0]
                            else:
                                name = faiss_service.canonical_employee_name_from_text(prompt)
                            if not name:
                                # Final fallback: use the most recent candidate shown previously
                                last_names = _last_candidate_names(st.session_state.messages, limit=1)
                                name = last_names[0] if last_names else None
                        
                        if intent == "followup_detail" and name:
                            # Get top chunks for this employee, then synthesize into a short paragraph
                            chunks = faiss_service.search_employee_details(name, query=rewritten, top_k=10)
                            if not chunks:
                                response = f"I couldn’t find additional details for {name}. Try asking about their skills, certifications, or experience."
                                candidates = []
                            else:
                                # Compose a friendly, chat-like summary using the model
                                response = _synthesize_employee_narrative(name, chunks, prompt)
                                # Also include the raw chunks as candidate cards for reference
                                candidates = []
                                for c in chunks[:3]:
                                    candidates.append({
                                        'name': name,
                                        'title': c.get('title', ''),
                                        'email': c.get('email', ''),
                                        'text': c.get('text', ''),
                                        'confidence': '',
                                        'score': c.get('similarity', 0.0),
                                    })
                                resolved_names_for_msg = [name]
                        elif intent == "followup_detail" and not name:
                            response = _generate_clarifying_question(st.session_state.messages, prompt)
                            candidates = []
                        elif intent == "compare":
                            # Compare first and second candidates (or explicitly referenced) on the topic
                            names = target_names if target_names else _last_candidate_names(st.session_state.messages, limit=2)
                            if len(names) < 2:
                                response = "Please specify two candidates to compare (e.g., ‘compare Nour Selim and Daniel Ebeid on Kubernetes’)."
                                candidates = []
                            else:
                                a, b = names[0], names[1]
                                chunks_a = faiss_service.search_employee_details(a, query=rewritten, top_k=8)
                                chunks_b = faiss_service.search_employee_details(b, query=rewritten, top_k=8)
                                response = _synthesize_comparison(a, chunks_a, b, chunks_b, prompt)
                                # Show a couple of supporting chunks for each person
                                candidates = []
                                for c in chunks_a[:2]:
                                    candidates.append({'name': a, 'title': c.get('title',''), 'email': c.get('email',''), 'text': c.get('text',''), 'confidence': '', 'score': c.get('similarity',0.0)})
                                for c in chunks_b[:2]:
                                    candidates.append({'name': b, 'title': c.get('title',''), 'email': c.get('email',''), 'text': c.get('text',''), 'confidence': '', 'score': c.get('similarity',0.0)})
                                resolved_names_for_msg = [a, b]
                        else:
                            # Before we search, ensure the request is clear enough
                            need_clarify, question = _assess_query_readiness(st.session_state.messages, prompt, rewritten)
                            if need_clarify:
                                response = question
                                candidates = []
                            else:
                                # Default search intent → show ranked candidates
                                requested_k = _extract_topk_from_prompt(prompt) or 5
                                ranked_candidates = faiss_service.search_filtered(
                                    rewritten,
                                    target_names=target_names or None,
                                    chunk_types=desired_types or None,
                                    top_k=max(50, requested_k * 10),
                                    pool_size=requested_k,
                                )
                    
                            if not ranked_candidates:
                                # Ask an LLM-generated clarifying question instead of a generic failure
                                response = _generate_clarifying_question(st.session_state.messages, prompt)
                                candidates = []
                            else:
                                # Present a single concise header and omit confidence numbers
                                response = f"🔍 {len(ranked_candidates)} candidates match your query:\n\n"
                                candidates = []
                                # Optionally generate a one-line relevance snippet using the LLM
                                def make_snippet(name: str, text: str) -> str:
                                    sys_prompt = (
                                        "Given the user's query and a chunk of candidate info, "
                                        "write ONE short line explaining why this candidate is relevant. "
                                        "Max 18 words."
                                        "If the user's query is an incomplete sentence, ask for clarity."
                                        "If the question is unclear, ask for clarity."
                                    )
                                    try:
                                        r = _oa_client.chat.completions.create(
                                            model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
                                            messages=[
                                                {"role": "system", "content": sys_prompt},
                                                {"role": "user", "content": f"Query: {prompt}\nCandidate: {name}\nInfo: {text}"},
                                            ],
                                            temperature=0.3,
                                            max_tokens=50,
                                        )
                                        out = (r.choices[0].message.content or "").strip()
                                        return out
                                    except Exception:
                                        return ""

                                for i, (emp_id, (final_score, cos_score, meta)) in enumerate(ranked_candidates, 1):
                                    name = meta.get("employee_name") or "Name Not Available"
                                    title = meta.get("title") or ""
                                    email = meta.get("email") or ""
                                    snippet = make_snippet(name, meta.get('text', ''))

                                    response += f"**{i}. {name}**"
                                    if title:
                                        response += f" - {title}"
                                    if email:
                                        response += f" ({email})"
                                    response += f"\n"
                                    if snippet:
                                        response += f"{snippet}\n\n"
                                    else:
                                        response += "\n"

                                    candidates.append({
                                        'name': name,
                                        'title': title,
                                        'email': email,
                                        'text': meta.get('text', ''),
                                        'confidence': '',
                                        'score': cos_score
                                    })
                        
                except Exception as e:
                    response = f"❌ Error during search: {str(e)}"
                    candidates = []
        
        # Display response once (no typing effect to prevent double-render/format glitches)
        message_placeholder.markdown(response)

    # Add assistant message to chat history
    msg_record = {"role": "assistant", "content": response}
    if candidates:
        msg_record["candidates"] = candidates
    # Attach resolved names if present in local scope
    try:
        if resolved_names_for_msg:
            msg_record["resolved_names"] = resolved_names_for_msg
    except NameError:
        pass
    st.session_state.messages.append(msg_record)
    
    # Update session storage
    if st.session_state.current_session not in st.session_state.chat_sessions:
        st.session_state.chat_sessions[st.session_state.current_session] = []
    st.session_state.chat_sessions[st.session_state.current_session] = st.session_state.messages.copy()

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #888; font-size: 12px;'>
       DeBotte AI - Employee Skills Finder | Built by Innov8 with ❤️
    </div>
    """,
    unsafe_allow_html=True
)