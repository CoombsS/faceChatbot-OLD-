import cv2
import numpy as np

def compare():
    #Read image
    imgColor = cv2.imread("deepface_env/Faces/img1.jpg")
    
    #Check if image was loaded
    if imgColor is None:
        print("Error: Image not found or unable to load.")
        return

    # Display the image
    cv2.imshow("Image Window", imgColor)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

import cv2

def capture_face():
    cam = cv2.VideoCapture(0)
    
    # Load the Haar cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    detectedFace = None  # Variable to store the detected face

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to capture image")
            break

        # Convert frame to grayscale for better face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            # Capture the first detected face
            x, y, w, h = faces[0]
            detectedFace = frame[y:y+h, x:x+w]  # Crop the face
            break  

    cam.release()
    cv2.destroyAllWindows()
    
        # Check if a face was detected
    if detectedFace is not None:
        cv2.imshow("Captured Face", detectedFace)
        cv2.waitKey(0)  # Wait for a key press
        cv2.destroyAllWindows()
        return detectedFace  # Return the captured face
    else:
        return None 
    




if __name__ == "__main__":
    capture_face()
