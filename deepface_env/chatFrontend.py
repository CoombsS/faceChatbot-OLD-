import os
import requests
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

# Get the key  
openai_api_key = os.getenv("OPENAI_API_KEY")
print(f"Loaded API Key: {openai_api_key}")  # Debugging

if not openai_api_key:
    raise ValueError("OpenAI API key is not set in environment variables.")

# --- page setup ---
st.set_page_config(layout="wide", page_title="Chatbot Frontend Testing")
st.title("Chatbot Frontend Testing")

st.markdown("""
    <style>
        .stApp {
            background-image: url("https://images.unsplash.com/photo-1506744038136-46273834b3fb");
            background-size: cover;
            background-position: center;
            color: white;
            padding: 20px;
        }
        .chat-container {
            height: 40vh;
            overflow-y: auto;
            background: rgba(0, 0, 0, 0.4);
            padding: 15px;
            border-radius: 15px;
            margin-bottom: 20px;
        }
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
        .timestamp {
            font-size: 0.8em;
            color: #ccc;
            margin-top: 2px;
            display: block;
        }
        input[type="text"] {
            background-color: #ffffff20;
            color: white;
            padding: 10px;
            border-radius: 10px;
            border: 1px solid #fff;
            width: 100%;
            box-sizing: border-box;
        }
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

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Formatting (sender, message, timestamp) for chat history
st.session_state.chat_history = [
    entry for entry in st.session_state.chat_history
    if isinstance(entry, tuple) and len(entry) == 3
]

# --- chat bubbles ---
chat_html = '<div class="chat-container">'
for sender, message, timestamp in reversed(st.session_state.chat_history):
    if sender == "user":
        chat_html += f'''
            <div class="user-bubble">
                <strong>You:</strong> {message}
                <span class="timestamp">{timestamp}</span>
            </div>'''
    else:
        chat_html += f'''
            <div class="bot-bubble">
                <strong>Bot:</strong> {message}
                <span class="timestamp">{timestamp}</span>
            </div>'''
chat_html += '</div>'
st.markdown(chat_html, unsafe_allow_html=True)

# --- Auto-scroll ---
st.markdown("""
    <script>
        const chatContainer = parent.document.querySelector('.chat-container');
        if (chatContainer) {
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
    </script>
""", unsafe_allow_html=True)

# --- Input field ---
input_key = "question_input"
placeholder = st.empty()

# --- user input and bot replies ---
def handle_input():
    user_input = st.session_state.get(input_key, "").strip()
    if user_input:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.chat_history.append(("user", user_input, now))

        with st.spinner("Thinking..."):
            conversation = [
                {"role": "user" if sender == "user" else "assistant", "content": msg}
                for sender, msg, _ in st.session_state.chat_history
            ]
            conversation.append({"role": "user", "content": user_input})

            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {openai_api_key}"
                },
                json={
                    "model": "gpt-3.5-turbo",
                    "messages": conversation
                }
            )

            if response.status_code == 200:
                bot_reply = response.json()["choices"][0]["message"]["content"]
            else:
                bot_reply = response.json().get("error", {}).get("message", "Failed to get response.")

            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.chat_history.append(("bot", bot_reply, now))

        st.session_state[input_key] = ""

# --- Button Row: Clear Chat (left), Send (center), Exit App (right) ---
left_col, middle_col, right_col = st.columns([1, 1, 1]) #WHY WONT YOU WORK?

with left_col:
    if st.button("Clear Chat"):
        st.session_state.chat_history = []

with middle_col:
    if st.button("Send", key="send_button"):
        handle_input()

with right_col:
    if st.button("End Chat"):
        st.success("Chat has ended. Please close the tab or refresh to restart.")
        st.stop()

# --- Chat Input ---
user_input = placeholder.text_input(
    "Your message:",
    key=input_key,
    label_visibility="collapsed",
    on_change=handle_input
)
