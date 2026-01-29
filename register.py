import cv2
import os
import face_recognition
import pickle
import numpy as np

ENCODINGS_PATH = "encodings/faces.pkl"

name = input("Enter user name: ").strip()

# Load existing encodings if available
known_encodings = []
known_names = []

if os.path.exists(ENCODINGS_PATH):
    with open(ENCODINGS_PATH, "rb") as f:
        known_encodings, known_names = pickle.load(f)

user_dir = os.path.join("data", name)
os.makedirs(user_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0
duplicate_found = False

print("Press 'c' to capture image, 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, locations)

    # Check for duplicate face
    for encoding in encodings:
        if known_encodings:
            distances = face_recognition.face_distance(known_encodings, encoding)
            if np.min(distances) < 0.45:
                match_index = np.argmin(distances)
                print(f"\n⚠ Face already registered as: {known_names[match_index]}")
                duplicate_found = True
                break

    cv2.imshow("Register Face", frame)
    key = cv2.waitKey(1)

    if duplicate_found:
        break

    if key == ord('c') and encodings:
        img_path = os.path.join(user_dir, f"{count}.jpg")
        cv2.imwrite(img_path, frame)
        print(f"Captured image {count}")
        count += 1

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if duplicate_found:
    print("Registration aborted to prevent duplicate user.")
    if os.path.exists(user_dir) and len(os.listdir(user_dir)) == 0:
        os.rmdir(user_dir)
else:
    print("Registration completed successfully.")
