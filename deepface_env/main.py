import json
import pandas as pd
import numpy as np
import deepface.DeepFace as DF
import cv2
import os

# Initialize camera
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("There was an error opening the webcam")
    exit()
    
# Ensure necessary directories exist
temp_faces = "temp_faces"
os.makedirs(temp_faces, exist_ok=True)

learned_faces = "learned_faces"
os.makedirs(learned_faces, exist_ok=True)

# Path for storing recognized faces data
recognized_faces_file = "recognized_faces.json"
RECOGNITION_THRESHOLD = 0.4  # Lower values mean stricter matching (0.0-1.0)

# Load recognized faces data
def load_recognized_faces():
    if os.path.exists(recognized_faces_file):
        with open(recognized_faces_file, "r") as f:
            return json.load(f)
    return {}

# Save recognized faces data
def save_recognized_faces(data):
    with open(recognized_faces_file, "w") as f:
        json.dump(data, f, indent=4)

def detect():
    cam = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    detectedFace = None  

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to capture image")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) > 0:
            x, y, w, h = faces[0]
            detectedFace = frame[y:y+h, x:x+w]
            break  

    cam.release()
    cv2.destroyAllWindows()

    if detectedFace is not None:
        cv2.imshow("Captured Face", detectedFace)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return recognize(detectedFace)  
    else:
        return None 

def recognize(detectedFace):
    temp_face_path = os.path.join(temp_faces, "temp_face.jpg")
    cv2.imwrite(temp_face_path, detectedFace)

    recognized_faces = load_recognized_faces()

    person_dirs = [os.path.join(learned_faces, d) for d in os.listdir(learned_faces) if os.path.isdir(os.path.join(learned_faces, d))]

    if person_dirs:
        try:
            results = DF.find(img_path=temp_face_path, db_path=learned_faces, model_name="VGG-Face", enforce_detection=False)

            if results and len(results) > 0 and not results[0].empty:
                print("DeepFace Output Structure:", results[0].columns)  # Debugging step
                
                # Ensure the similarity metric exists before accessing it
                similarity_metric = "VGG-Face_cosine"
                if similarity_metric not in results[0].columns:
                    print(f"Error: '{similarity_metric}' key not found in DeepFace results.")
                    return None

                identity_path = results[0]['identity'][0]
                person_name = identity_path.split(os.sep)[-2]
                confidence_score = results[0][similarity_metric][0]

                if confidence_score < RECOGNITION_THRESHOLD:
                    print(f"Recognized as {person_name} with confidence {confidence_score:.2f}")
                    save_to_existing(person_name, temp_face_path)
                    recognized_faces[person_name] = recognized_faces.get(person_name, 0) + 1
                    save_recognized_faces(recognized_faces)
                    return person_name
                else:
                    print("Face not recognized confidently. Storing as a new entry.")
                    return store_new_face(temp_face_path, recognized_faces)
            else:
                print("No match found. Storing as a new face.")
                return store_new_face(temp_face_path, recognized_faces)
        except Exception as e:
            print("Error during recognition:", e)
            return None
    else:
        print("No known faces. Storing first face.")
        return store_new_face(temp_face_path, recognized_faces)


# Save additional images to an existing person's folder
def save_to_existing(person_name, face_path):
    person_folder = os.path.join(learned_faces, person_name)
    new_face_path = os.path.join(person_folder, f"face_{len(os.listdir(person_folder))}.jpg")

    os.rename(face_path, new_face_path)
    print(f"Saved additional image for {person_name} as {new_face_path}")

# Start detection
detected_person = detect()
if detected_person:
    print(f"Detected and recognized as: {detected_person}")
