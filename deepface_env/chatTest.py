# chat.py

import json
import cv2
import os
import deepface.DeepFace as DF
import streamlit as st
from datetime import datetime

# Session setup
username = st.session_state.get("username", None)
if username:
    st.write(f"Welcome back, {username}!")

# Directories
temp_faces = "temp_faces"
learned_faces = "learned_faces"
os.makedirs(temp_faces, exist_ok=True)
os.makedirs(learned_faces, exist_ok=True)

# Face data file
recognized_faces_file = "recognized_faces.json"

def load_recognized_faces():
    if os.path.exists(recognized_faces_file):
        with open(recognized_faces_file, "r") as f:
            return json.load(f)
    return {}

def save_recognized_faces(data):
    with open(recognized_faces_file, "w") as f:
        json.dump(data, f, indent=4)

def detect():
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Error: Unable to access the webcam.")
        return None

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    detected_face = None  

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Error: Unable to capture image.")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) > 0:
            x, y, w, h = faces[0]
            detected_face = frame[y:y+h, x:x+w]
            break  

        cv2.waitKey(1)  

    cam.release()
    cv2.destroyAllWindows()

    if detected_face is not None:
        cv2.imshow("Captured Face", detected_face)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return recognize(detected_face)  
    else:
        print("No face detected.")
        return None 

def recognize(detected_face):
    temp_face_path = os.path.join(temp_faces, "temp_face.jpg")
    cv2.imwrite(temp_face_path, detected_face)

    recognized_faces = load_recognized_faces()
    
    if os.listdir(learned_faces):
        try:
            results = DF.find(img_path=temp_face_path, db_path=learned_faces, model_name="Facenet", enforce_detection=False)
            if results and not results[0].empty:
                identity_path = results[0]['identity'][0]
                person_name = os.path.basename(os.path.dirname(identity_path))
                print(f"Recognized as {person_name}")
                save_to_existing(person_name, temp_face_path)
                recognized_faces[person_name] = recognized_faces.get(person_name, 0) + 1
                save_recognized_faces(recognized_faces)
                return person_name
            else:
                print("No match found. Storing as a new face.")
                return store_new_face(temp_face_path, recognized_faces)
        except Exception as e:
            print(f"Error during recognition: {e}")
            return None
    else:
        print("No known faces. Storing first face.")
        return store_new_face(temp_face_path, recognized_faces)

def store_new_face(face_path, recognized_faces):
    print("Storing new face...")
    while True:
        try:
            person_name = input("Enter the person's name: ").strip()
            if person_name:
                break
            print("Name cannot be empty. Please enter a valid name.")
        except Exception as e:
            print(f"Error while entering name: {e}")

    person_folder = os.path.join(learned_faces, person_name)
    os.makedirs(person_folder, exist_ok=True)

    jpg_count = len([f for f in os.listdir(person_folder) if f.endswith(".jpg")])
    new_face_path = os.path.join(person_folder, f"face_{jpg_count}.jpg")

    os.replace(face_path, new_face_path)
    print(f"Saved new face as {new_face_path}")
    
    recognized_faces[person_name] = recognized_faces.get(person_name, 0) + 1
    save_recognized_faces(recognized_faces)
    return person_name

def save_to_existing(person_name, face_path):
    person_folder = os.path.join(learned_faces, person_name)
    jpg_count = len([f for f in os.listdir(person_folder) if f.endswith(".jpg")])
    new_face_path = os.path.join(person_folder, f"face_{jpg_count}.jpg")
    os.replace(face_path, new_face_path)
    print(f"Saved additional image for {person_name} as {new_face_path}")

# Streamlit Config
st.set_page_config(page_title="Therapeutic Chatbot", page_icon="🧑‍⚕️")

# Styling
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

# Chat session
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat history display
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

# Input
st.markdown('<div class="input-box">', unsafe_allow_html=True)
user_input = st.text_input("Your message:", key="user_input", label_visibility="collapsed")
st.markdown('</div>', unsafe_allow_html=True)

# Face recognition trigger
if st.button("Recognize Face and Start Chat", key="recognize_face", help="Click to start face recognition and chat"):
    st.write("Starting face detection and recognition...")
    detected_person = detect()
    if detected_person:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.chat_history.append(("bot", f"Hello, {detected_person}! How can I assist you today?", now))
    else:
        st.write("No person detected.")

# Handle input
if user_input:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.chat_history.append(("user", user_input, now))

    # Replace this with actual AI/chatbot logic
    response = "This is a response from the bot."
    st.session_state.chat_history.append(("bot", response, now))

# Chat control
if st.button("Clear Chat", key="clear_chat"):
    st.session_state.chat_history = []

if st.button("End Chat", key="end_chat"):
    st.write("Chat has ended. Please close the tab or refresh to restart.")
    st.stop()
