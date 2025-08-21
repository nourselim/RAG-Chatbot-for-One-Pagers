
import streamlit as st
import time
from datetime import datetime
from pathlib import Path
import sys
import os

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
    """Return the most recent assistant candidates list from history, if present."""
    for m in reversed(messages or []):
        if m.get("role") == "assistant" and isinstance(m, dict) and "candidates" in m:
            return m.get("candidates") or []
    return []

def _extract_target_names_from_prompt(prompt: str, messages):
    """Heuristic: resolve pronouns/ordinals to candidate names from last assistant response."""
    prompt_low = (prompt or "").lower()
    recent_cands = _recent_candidates_from_history(messages)
    cand_names = [c.get("name", "").strip() for c in recent_cands if c.get("name")]

    referenced = []
    # Name mentions
    for nm in cand_names:
        if nm and nm.lower() in prompt_low:
            referenced.append(nm)
    # Ordinals
    ordinal_map = {
        "first": 0, "1st": 0, "one": 0, "top": 0,
        "second": 1, "2nd": 1, "two": 1,
        "third": 2, "3rd": 2, "three": 2,
    }
    for key, idx in ordinal_map.items():
        if key in prompt_low and 0 <= idx < len(cand_names):
            referenced.append(cand_names[idx])
    # Pronouns fallback
    if not referenced and any(p in prompt_low for p in ["they", "them", "their", "him", "her", "that person", "the candidate", "this candidate"]):
        if cand_names:
            referenced.append(cand_names[0])

    # Deduplicate, preserve order
    seen = set(); out = []
    for nm in referenced:
        if nm not in seen:
            seen.add(nm); out.append(nm)
    return out

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
    st.markdown("**Employee Skills Finder**")
    st.markdown("---")
    
    # New Chat Button
    if st.button("➕ New Chat", use_container_width=True):
        new_session_id = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        st.session_state.chat_sessions[new_session_id] = []
        st.session_state.current_session = new_session_id
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    
    # Recent Chats
    st.subheader("Recent Chats")
    for session_id in list(st.session_state.chat_sessions.keys()):
        if st.button(f"💬 Chat {session_id[-6:]}", key=session_id, use_container_width=True):
            st.session_state.current_session = session_id
            st.session_state.messages = st.session_state.chat_sessions[session_id]
            st.rerun()
    
    # Clear All Chats
    if st.button("🗑️ Clear All Chats", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_sessions = {}
        st.session_state.current_session = "default"
        st.rerun()
    
    st.markdown("---")
    
    # System Info
    st.subheader("System Status")
    if faiss_service:
        st.success("✅ FAISS Index Loaded")
        st.info(f"📊 Ready to search")
    else:
        st.error("❌ FAISS Index Not Available")
        st.info("💡 Run: `cd rag && python main.py auto`")
    
    st.markdown("---")
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

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and "candidates" in message:
            # Display candidate results
            st.markdown(message["content"])
            for candidate in message["candidates"]:
                with st.container():
                    st.markdown(f"""
                    <div class="candidate-result">
                        <div class="candidate-name">{candidate['name']}</div>
                        <div class="candidate-details">
                            {candidate.get('title', '')} {candidate.get('email', '')}
                        </div>
                        <div class="candidate-text">{candidate['text']}</div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.markdown(message["content"])

# ----------------- Intent classification helpers -----------------
def _classify_intent(text: str) -> str:
    t = (text or "").strip().lower()
    if not t:
        return "empty"
    greetings = ["hi", "hello", "hey", "good morning", "good evening", "good afternoon"]
    if any(t.startswith(g) or t == g for g in greetings):
        return "greeting"
    if any(k in t for k in ["compare", "difference between", "vs "]):
        return "compare"
    if any(k in t for k in ["tell me more", "more about", "details about", "expand on", "elaborate", "what about the", "first one", "second one", "third one"]):
        return "followup_detail"
    # otherwise assume search intent
    return "search"

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
                        response = (
                            "Hello! I’m your Deloitte Skills Finder. Ask me about skills, experience, "
                            "certifications, or clients. For example: ‘Who has Kubernetes experience?’"
                        )
                        candidates = []
                    else:
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
                        if intent == "followup_detail" and target_names:
                            name = target_names[0]
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
                        else:
                            # Default search intent → show ranked candidates
                            ranked_candidates = faiss_service.search_filtered(
                                rewritten,
                                target_names=target_names or None,
                                chunk_types=desired_types or None,
                                top_k=50,
                                pool_size=5,
                            )
                    
                            if not ranked_candidates:
                                response = "❌ No suitable candidates found for your query."
                                candidates = []
                            else:
                                response = f"🔍 Found {len(ranked_candidates)} candidates matching your query:\n\n"
                                candidates = []
                                for i, (emp_id, (final_score, cos_score, meta)) in enumerate(ranked_candidates, 1):
                                    name = meta.get("employee_name") or "Name Not Available"
                                    title = meta.get("title") or ""
                                    email = meta.get("email") or ""
                                    chunk_type = meta.get("chunk_type", "unknown").title()

                                    confidence = "High" if cos_score >= 0.60 else ("Medium" if cos_score >= 0.40 else "Low")

                                    response += f"**{i}. {name}**"
                                    if title:
                                        response += f" - {title}"
                                    if email:
                                        response += f" ({email})"
                                    response += f"\n"
                                    response += f"📊 Confidence: {confidence} ({cos_score:.2f})\n"
                                    response += f"📝 Source: {chunk_type}\n\n"

                                    candidates.append({
                                        'name': name,
                                        'title': title,
                                        'email': email,
                                        'text': meta.get('text', ''),
                                        'confidence': confidence,
                                        'score': cos_score
                                    })
                        
                except Exception as e:
                    response = f"❌ Error during search: {str(e)}"
                    candidates = []
        
        # Display response with typing effect
        partial = ""
        for token in response.split():
            partial = (partial + " " + token).strip()
            message_placeholder.markdown(partial + "▌")
            time.sleep(0.01)
        message_placeholder.markdown(partial)

    # Add assistant message to chat history
    st.session_state.messages.append({
        "role": "assistant", 
        "content": response,
        "candidates": candidates
    })
    
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