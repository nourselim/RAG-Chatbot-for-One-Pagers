
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
        max-width: 900px;
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
        background-color: #202123;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #10a37f;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 14px;
        cursor: pointer;
    }
    
    .stButton > button:hover {
        background-color: #0d8f6f;
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
    st.markdown("""
    1. Ask about employee skills, experience, or certifications
    2. Use natural language queries
    3. Example: "Find employees with AWS certification"
    4. Example: "Who has experience with SAP?"
    """)

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