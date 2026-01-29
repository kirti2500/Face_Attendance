import face_recognition
import os
import pickle

DATA_DIR = "data"
ENCODE_PATH = "encodings/faces.pkl"

known_encodings = []
known_names = []

for person_name in os.listdir(DATA_DIR):
    person_path = os.path.join(DATA_DIR, person_name)

    if not os.path.isdir(person_path):
        continue

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)

        image = face_recognition.load_image_file(img_path)
        encodings = face_recognition.face_encodings(image)

        if len(encodings) > 0:
            known_encodings.append(encodings[0])
            known_names.append(person_name)

print("Saving encodings...")
with open(ENCODE_PATH, "wb") as f:
    pickle.dump((known_encodings, known_names), f)

print("Encoding complete.")
