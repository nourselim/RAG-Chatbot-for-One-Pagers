import time
from datetime import datetime
import time
from pathlib import Path
import sys
import os, requests, uuid, streamlit as st

rag_path = Path(__file__).parent.parent / "rag"
sys.path.append(str(rag_path))

from faiss_service import FaissCandidateSearch
from embed_only import OUT_DIR, EMB_NPY, META_JSONL

# Page configuration
st.set_page_config(
    page_title="DeBotte AI - Employee Skills Finder", 
    page_icon="🤖", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
API_BASE = f"{BACKEND_URL}/sessions"

def api_list_sessions(limit=50):
    """Fetch recent chat sessions from the backend."""
    try:
        r = requests.get(f"{API_BASE}", params={"limit": limit}, timeout=10)
        r.raise_for_status()
        return r.json()  # Expecting: [{session_id, started_at, updated_at, last_role, last_message}]
    except Exception as e:
        print("Error fetching sessions:", e)
        return []

# one session per tab (also supports ?session=<uuid> in the URL)
# --- URL query params helpers for Streamlit <=1.28 ---
def get_query_param(name: str):
    # returns str | None
    params = st.experimental_get_query_params()  # dict[str, list[str]]
    vals = params.get(name)
    if vals:
        return vals[0]
    return None

def set_query_params(**kwargs):
    # kwargs: key="value"
    st.experimental_set_query_params(**kwargs)
# -----------------------------------------------------

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
API_BASE = f"{BACKEND_URL}/sessions"

# one session per tab; respect ?session=<uuid> in URL
if "session_id" not in st.session_state:
    existing = get_query_param("session")
    if existing:
        st.session_state.session_id = existing
    else:
        st.session_state.session_id = str(uuid.uuid4())
        set_query_params(session=st.session_state.session_id)

def api_post_message(role: str, message: str):
    url = f"{API_BASE}/{st.session_state.session_id}/messages"
    r = requests.post(url, json={"role": role, "message": message}, timeout=10)
    r.raise_for_status()
    return r.json()

def api_get_messages(order="asc", limit=200):
    url = f"{API_BASE}/{st.session_state.session_id}/messages"
    r = requests.get(url, params={"order": order, "limit": limit}, timeout=10)
    r.raise_for_status()
    return r.json()

# Load environment variables
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=False)

# Check for OpenAI API key
if not os.getenv("OPENAI_API_KEY"):
    st.error("❌ OPENAI_API_KEY not found. Please set it in your .env file.")
    st.stop()

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

# ----------------- CSS Styling -----------------
st.markdown("""
<style>
    /* Main container styling */
    .main {
        padding: 0;
    }
    
    /* Chat container */
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 20px;
    }
    
    /* Message styling */
    .user-message {
        background-color: #40414f;
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        margin-left: 50px;
        margin-right: 0;
    }
    
    .assistant-message {
        background-color: #444654;
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        margin-right: 50px;
        margin-left: 0;
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        background-color: #40414f;
        color: white;
        border: 1px solid #565869;
        border-radius: 10px;
        padding: 15px;
        font-size: 16px;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: black;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #26890d;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 14px;
        cursor: pointer;
    }
    /* Sidebar full container */
    section[data-testid="stSidebar"] {
        background-color: #000000 !important;
    }

    /* Sidebar content area */
    section[data-testid="stSidebar"] .css-1d391kg {
        background-color: #000000 !important;
    }

    /* Make sidebar text visible */
    section[data-testid="stSidebar"] * {
        color: white !important;
    }
    
    .stButton > button:hover {
        background-color: #1f6d0b;
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #2d2d2d;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 4px;
    }
    
    /* Chat history styling */
    .chat-history {
        background-color: #202123;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    
    /* Model selector styling */
    .model-selector {
        background-color: #202123;
        color: white;
        border: 1px solid #565869;
        border-radius: 5px;
        padding: 5px;
    }
            
    /* Candidate result styling */
    .candidate-result {
        background-color: #2d2d2d;
        border: 1px solid #565869;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    
    .candidate-name {
        font-size: 18px;
        font-weight: bold;
        color: #10a37f;
        margin-bottom: 5px;
    }
    
    .candidate-details {
        color: #cccccc;
        font-size: 14px;
        margin-bottom: 10px;
    }
    
    .candidate-text {
        color: #ffffff;
        font-size: 13px;
        line-height: 1.4;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- State Management -----------------
with st.sidebar:
    st.title("🤖 DeBotte AI")
    st.markdown("**Employee Skills Finder**")
    st.markdown("---")

    # New Chat
    if st.button("➕ New Chat", use_container_width=True):
        new_id = str(uuid.uuid4())
        st.session_state.session_id = new_id
        st.session_state.messages = []
        set_query_params(session=new_id)
        st.rerun()

    st.markdown("---")

    # Recent Chats (DB-backed — shows after first message is saved)
    st.subheader("Recent Chats")
    sessions = api_list_sessions()
    if not sessions:
        st.caption("No recent chats yet. Start a chat and send a message.")
    else:
        seen = set()
        for i, s in enumerate(sessions):
            sid = s["session_id"]
            if sid in seen:
                continue
            seen.add(sid)

            preview = (s.get("last_message") or "").splitlines()[0][:40]
            label = f"💬 {sid[:8]}…{sid[-4:]} — {preview}"
            if st.button(label, key=f"sid_{sid}_{i}", use_container_width=True):
                set_query_params(session=sid)
                st.session_state.session_id = sid
                if "messages_loaded" in st.session_state:
                    del st.session_state["messages_loaded"]
                st.rerun()

    st.markdown("---")

    # System Info
    st.subheader("System Status")
    if faiss_service:
        st.success("✅ FAISS Index Loaded")
        st.info("📊 Ready to search")
    else:
        st.error("❌ FAISS Index Not Available")
        st.info("💡 Run: `cd rag && python main.py auto`")

    st.markdown("---")
    st.markdown("**Instructions:**")
    st.markdown("""
    1. Ask about employee skills, experience, or certifications
    2. Use natural language queries
    3. Example: "Find employees with AWS certification"
    4. Example: "Who has experience with SAP?"
    """)


# ----------------- Main Chat Interface -----------------
st.markdown('<div class="chat-container">', unsafe_allow_html=True)


if "messages_loaded" not in st.session_state:
    try:
        past = api_get_messages(order="asc")
        st.session_state.messages = [{"role": m["role"], "content": m["message"]} for m in past]
    except Exception as e:
        st.sidebar.warning(f"Couldn’t load history: {e}")
    st.session_state.messages_loaded = True



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

# Chat input
if prompt := st.chat_input("Ask about employee skills, experience, or certifications..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # >>> ADD: persist USER message
    try:
        api_post_message("user", prompt)
    except Exception as e:
        st.warning(f"Failed to save user message: {e}")
    # <<<

    # Generate response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        if faiss_service is None:
            response = "❌ FAISS service is not available. Please run the RAG pipeline first."
            candidates = []
        else:
            with st.spinner("🔍 Searching for candidates..."):
                try:
                    ranked_candidates = faiss_service.search(prompt, top_k=50, pool_size=5)
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
                            if title: response += f" - {title}"
                            if email: response += f" ({email})"
                            response += f"\n📊 Confidence: {confidence} ({cos_score:.2f})\n📝 Source: {chunk_type}\n\n"
                            candidates.append({
                                'name': name,
                                'title': title,
                                'email': email,
                                'text': meta.get('text', '')[:200] + "..." if len(meta.get('text', '')) > 200 else meta.get('text', ''),
                                'confidence': confidence,
                                'score': cos_score
                            })
                        response += "💡 **Tip:** Ask follow-up questions about specific candidates or skills!"
                except Exception as e:
                    response = f"❌ Error during search: {str(e)}"
                    candidates = []

        # typing effect (unchanged)
        partial = ""
        for token in response.split():
            partial = (partial + " " + token).strip()
            message_placeholder.markdown(partial + "▌")
            time.sleep(0.01)
        message_placeholder.markdown(partial)

    # Add assistant message to chat history (unchanged)
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "candidates": candidates
    })

    # >>> ADD: persist ASSISTANT message
    try:
        api_post_message("bot", response)
    except Exception as e:
        st.warning(f"Failed to save assistant message: {e}")
    # <<<

    # (your session storage UI bookkeeping can remain or be removed)


    # Generate response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        if faiss_service is None:
            response = "❌ FAISS service is not available. Please run the RAG pipeline first."
            candidates = []
        else:
            with st.spinner("🔍 Searching for candidates..."):
                try:
                    # Search for candidates
                    ranked_candidates = faiss_service.search(prompt, top_k=50, pool_size=5)
                    
                    if not ranked_candidates:
                        response = "❌ No suitable candidates found for your query."
                        candidates = []
                    else:
                        # Format the response
                        response = f"🔍 Found {len(ranked_candidates)} candidates matching your query:\n\n"
                        candidates = []
                        
                        for i, (emp_id, (final_score, cos_score, meta)) in enumerate(ranked_candidates, 1):
                            name = meta.get("employee_name") or "Name Not Available"
                            title = meta.get("title") or ""
                            email = meta.get("email") or ""
                            chunk_type = meta.get("chunk_type", "unknown").title()
                            
                            # Calculate confidence
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
                                'text': meta.get('text', '')[:200] + "..." if len(meta.get('text', '')) > 200 else meta.get('text', ''),
                                'confidence': confidence,
                                'score': cos_score
                            })
                        
                        response += "💡 **Tip:** Ask follow-up questions about specific candidates or skills!"

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