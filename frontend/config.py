"""
Configuration settings for the AI Chat Assistant
"""

import os
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()

# Application settings
APP_NAME = "AI Chat Assistant"
APP_VERSION = "1.0.0"
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

# Streamlit settings
STREAMLIT_CONFIG = {
    "theme.base": "dark",
    "theme.primaryColor": "#10a37f",
    "theme.backgroundColor": "#202123",
    "theme.secondaryBackgroundColor": "#343541",
    "theme.textColor": "#ffffff",
}

# # AI Model settings
# AVAILABLE_MODELS = [
#     "GPT-3.5",
#     "GPT-4", 
#     "GPT-4 Turbo",
#     "Claude-3",
#     "Llama-2"
# ]

# Chat settings
MAX_CHAT_HISTORY = 100
MAX_MESSAGE_LENGTH = 4000
TYPING_SPEED = 0.05  # seconds between characters for typing effect
