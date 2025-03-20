import json
import cv2
import os
import deepface.DeepFace as DF

# Ensure necessary directories exist
temp_faces = "temp_faces"
learned_faces = "learned_faces"
os.makedirs(temp_faces, exist_ok=True)
os.makedirs(learned_faces, exist_ok=True)

# Path for storing recognized faces data
recognized_faces_file = "recognized_faces.json"

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

# Detect a face
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

# Recognize a detected face
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

# Store new face
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

# Save additional images to an existing person's folder
def save_to_existing(person_name, face_path):
    person_folder = os.path.join(learned_faces, person_name)
    jpg_count = len([f for f in os.listdir(person_folder) if f.endswith(".jpg")])
    new_face_path = os.path.join(person_folder, f"face_{jpg_count}.jpg")

    os.replace(face_path, new_face_path)
    print(f"Saved additional image for {person_name} as {new_face_path}")

# Start detection
detected_person = detect()
if detected_person:
    print(f"Detected and recognized as: {detected_person}")
else:
    print("No person detected.")
