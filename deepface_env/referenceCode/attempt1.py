import cv2
import os
import deepface.DeepFace as DF

# Initialize webcam
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Error: Could not open webcam")
    exit()

# Ensure face database directory exists
Face_DB = "temp_faces"
os.makedirs(Face_DB, exist_ok=True)

frameCount = 0  # Keep track of images
differentFaceCount = 0  # Initialize face count

def captureAndCompare():
    global frameCount, differentFaceCount
    ret, frame = cam.read()
    if not ret or frame is None:
        print("Error: Could not capture frame")
        return

    try:
        # Detect faces first
        faces = DF.extract_faces(img_path=frame, enforce_detection=False)
        if not faces:
            print("No face detected")
            return
        
        # Save frame as image only if a face is detected
        filename = os.path.join(Face_DB, f"captured_face_{frameCount}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Saved: {filename}")

        # Check for similar faces
        result = DF.find(img_path=filename, db_path=Face_DB, model_name="VGG-Face", enforce_detection=False)

        if result and len(result) > 0 and not result[0].empty:
            print("Similar face found:", result[0]["identity"].tolist())
            differentFaceCount -= 1
        else:
            print("No similar face found. Face stored in database")
            differentFaceCount += 1
    except Exception as e:
        print("Error:", e)
    
    frameCount += 1

# Loop for continuous face detection
while True:
    captureAndCompare()
    print("Different faces detected:", differentFaceCount)
    
    if differentFaceCount >= 25:
        print("Face has left")
        break
    
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

cam.release()
cv2.destroyAllWindows()
