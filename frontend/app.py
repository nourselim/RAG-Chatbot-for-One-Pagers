import streamlit as st
import time
import random
from datetime import datetime
from openai import OpenAI
from chatbot import chat_once
# Page configuration
st.set_page_config(
    page_title="DeBotte AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for ChatGPT-like styling
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
    
    /* Model selector styling */
    .model-selector {
        background-color: #202123;
        color: white;
        border: 1px solid #565869;
        border-radius: 5px;
        padding: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {}
if "current_session" not in st.session_state:
    st.session_state.current_session = "default"
# if "model" not in st.session_state:
#     st.session_state.model = "GPT-3.5"

# Sidebar
with st.sidebar:
    st.title("🤖 DeBotte AI")
    st.markdown("---")
    
    # # Model selection
    # st.subheader("Model")
    # model = st.selectbox(
    #     "Choose your model:",
    #     ["GPT-3.5", "GPT-4", "GPT-4 Turbo", "Claude-3", "Llama-2"],
    #     key="model_selector"
    # )
    # st.session_state.model = model
    
    # st.markdown("---")
    
    # New chat button
    if st.button("➕ New Chat", use_container_width=True):
        new_session_id = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        st.session_state.chat_sessions[new_session_id] = []
        st.session_state.current_session = new_session_id
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    
    # Chat history
    st.subheader("Recent Chats")
    
    # Display chat sessions
    for session_id in list(st.session_state.chat_sessions.keys()):
        if st.button(f"💬 Chat {session_id[-6:]}", key=session_id, use_container_width=True):
            st.session_state.current_session = session_id
            st.session_state.messages = st.session_state.chat_sessions[session_id]
            st.rerun()
    
    # Clear all chats
    if st.button("🗑️ Clear All Chats", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_sessions = {}
        st.session_state.current_session = "default"
        st.rerun()

# Main chat interface
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("DeBotte AI Assistant..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # # Generate AI response (simulated)
    # with st.chat_message("assistant"):
    #     message_placeholder = st.empty()
    #     full_response = ""


    # Generate AI response (OpenAI) with typing effect
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Thinking..."):
            # If/when you add RAG, pass your retrieved text/JSON as context here
            context = ""
            full_response = chat_once(prompt, context=context)

        # Typing animation (word-by-word)
        partial = ""
        for token in full_response.split():
            partial = (partial + " " + token).strip()
            message_placeholder.markdown(partial + "▌")
            time.sleep(0.015)
        message_placeholder.markdown(partial)

    # Add assistant message to history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    
    # Update current session
    if st.session_state.current_session not in st.session_state.chat_sessions:
        st.session_state.chat_sessions[st.session_state.current_session] = []
    st.session_state.chat_sessions[st.session_state.current_session] = st.session_state.messages.copy()

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #888; font-size: 12px;'>
       DeBotte AI | Built by Innov8 with ❤️
    </div>
    """,
    unsafe_allow_html=True
)
