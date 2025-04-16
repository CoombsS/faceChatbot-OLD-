import requests
import streamlit as st
from dotenv import load_dotenv
import os

# Load environment variables from the folder above this script
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

# Get the API key
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OpenAI API key is not set in environment variables.")

# Optional: Set page configuration
st.set_page_config(layout="wide", page_title="Chatbot Frontend Testing")

# Apply custom styling
st.markdown("""
    <style>
        /* Overall app styling */
        .stApp {
            background-image: url("https://images.unsplash.com/photo-1506744038136-46273834b3fb");
            background-size: cover;
            background-position: center;
            color: white;
            padding: 20px;
        }
        /* Scrollable chat container */
        .chat-container {
            height: 70vh;
            overflow-y: auto;
            background: rgba(0, 0, 0, 0.4);
            padding: 15px;
            border-radius: 15px;
            margin-bottom: 20px;
        }
        /* Styling for user messages */
        .user-bubble {
            background-color: #4CAF50;
            color: white;
            padding: 12px;
            border-radius: 20px;
            margin: 10px 0;
            max-width: 65%;
            float: right;
            clear: both;
            box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.3);
        }
        /* Styling for bot messages */
        .bot-bubble {
            background-color: #333;
            color: white;
            padding: 12px;
            border-radius: 20px;
            margin: 10px 0;
            max-width: 65%;
            float: left;
            clear: both;
            box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.3);
        }
        /* Input styling */
        input[type="text"] {
            background-color: #ffffff20;
            color: white;
            padding: 10px;
            border-radius: 10px;
            border: 1px solid #fff;
            width: 100%;
            box-sizing: border-box;
        }
        /* Button styling */
        .stButton button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border-radius: 10px;
            font-size: 1.1em;
            border: none;
            cursor: pointer;
        }
        .stButton button:hover {
            background-color: #45a049;
        }
    </style>
""", unsafe_allow_html=True)

st.title("Chatbot Frontend Testing")
st.header("Let's Talk")

# Initialize chat history if not already defined
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Build the chat container's HTML content
chat_html = '<div class="chat-container">'
for sender, message in st.session_state.chat_history:
    if sender == "user":
        chat_html += f'<div class="user-bubble"><strong>You:</strong> {message}</div>'
    else:
        chat_html += f'<div class="bot-bubble"><strong>Bot:</strong> {message}</div>'
chat_html += '</div>'

# Render the chat messages as one Markdown component
st.markdown(chat_html, unsafe_allow_html=True)

# Input area for the user message
user_question = st.text_input("Your message:", key="question", label_visibility="collapsed")
send_pressed = st.button("Send")

if (send_pressed or user_question) and user_question.strip() != "":
    # Add the user message to chat history
    st.session_state.chat_history.append(("user", user_question))
    
    with st.spinner("Thinking..."):
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai_api_key}"
        }
        data = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "user", "content": user_question},
            ]
        }
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            answer_text = response.json()["choices"][0]["message"]["content"]
            st.session_state.chat_history.append(("bot", answer_text))
        else:
            st.session_state.chat_history.append(("bot", "Failed to get a response from OpenAI API."))
    
    # Clear the text input and refresh the app
    st.session_state.question = ""
    st.experimental_rerun()
