import json
import pandas as pd  
from deepface import DeepFace

#declaring variables
#verifiedChk = verification1['verified']


# Face verification
verification1 = DeepFace.verify(img1_path="deepface_env/Faces/img1.jpg", img2_path="deepface_env/Faces/img2.jpg")
print(json.dumps(verification1, indent=4))
print("Are the people the same in img1, img2? ", verification1['verified'])

verification2 = DeepFace.verify(img1_path="deepface_env/Faces/img2.jpg", img2_path="deepface_env/Faces/img3.jpg")
print("Are the people the same in images 2,3? ", verification2['verified'])

meTest = DeepFace.verify(img1_path ="deepface_env/Faces/Skyler.jpg", img2_path="deepface_env/Faces/JDtest1.jpg")
print("metest results: ",meTest['verified'])

# Face recognition 
recognition_results = DeepFace.find(img_path="deepface_env/Faces/img3.jpg", db_path="deepface_env/Face_DB", enforce_detection=False)

# Cleaning up the output
print("\nFace recognition results:")
if recognition_results:
    recognition_df = recognition_results[0]
    if not recognition_df.empty:
        print("Recognized!")
        print(recognition_df[['identity', 'distance']].to_string(index=False))
    else:
        print("No matches found.")
else:
    print("No matches found.")



