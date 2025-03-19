import cv2
import deepface as DF


def extractFace():
    face = DF.extract_faces(img_path=frame, enforce_detection=False)
        if not faces:
            print("No face detected")
        return face

#cam initialization
cam = cv2.VideoCapture(0)

#capture video
while True:
    ret, frame = cam.read()
    if not ret:
        print("Failed to capture image")
        break
    cv2.imshow("Webcam Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()