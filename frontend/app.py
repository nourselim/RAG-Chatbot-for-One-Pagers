import streamlit as st
import time
import random
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="BeBotte AI",
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
    
    # Generate AI response (simulated)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Simulate typing effect
        responses = [
            f"Based on your question about '{prompt[:30]}...', here's what I think:",
            f"That's an interesting point about '{prompt[:20]}...'. Let me provide some insights:",
            f"I understand you're asking about '{prompt[:25]}...'. Here's my response:",
            f"Great question! Regarding '{prompt[:20]}...', here's what I know:",
        ]
        
        # Choose a random response starter
        response_start = random.choice(responses)
        full_response = response_start + "\n\n"
        
        # Add some dynamic content based on the prompt
        if "hello" in prompt.lower() or "hi" in prompt.lower():
            full_response += "Hello! I'm your AI assistant. I'm here to help you with any questions or tasks you have. How can I assist you today?"
        elif "code" in prompt.lower() or "programming" in prompt.lower():
            full_response += "I'd be happy to help with your programming question! Could you provide more specific details about what you're trying to accomplish? For example, what language are you using and what specific problem are you facing?"
        elif "explain" in prompt.lower():
            full_response += "I'd be glad to explain that! Let me break it down for you step by step. The key concepts here are..."
        elif "help" in prompt.lower():
            full_response += "I'm here to help! What specific assistance do you need? I can help with a wide range of topics including general knowledge, programming, writing, analysis, and more."
        else:
            full_response += "This is a simulated response. In a real implementation, this would be generated by an AI model based on your input. The response would be contextually relevant and provide helpful information."
        
        # Simulate typing
        for chunk in full_response.split():
            time.sleep(0.05)
            message_placeholder.markdown(full_response[:len(full_response.split(chunk)[0] + " " + chunk)] + "▌")
        
        message_placeholder.markdown(full_response)
    
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
