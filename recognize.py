import cv2
import face_recognition
import pickle
from utils import mark_attendance
import numpy as np

with open("encodings/faces.pkl", "rb") as f:
    known_encodings, known_names = pickle.load(f)

cap = cv2.VideoCapture(0)

print("Press 'i' for Punch In, 'o' for Punch Out, 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, locations)

    for face_encoding, face_location in zip(encodings, locations):
        distances = face_recognition.face_distance(known_encodings, face_encoding)
        min_dist = np.min(distances)

        if min_dist < 0.5:
            index = np.argmin(distances)
            name = known_names[index]
        else:
            name = "Unknown"

        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
        cv2.putText(frame, name, (left, top-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    cv2.imshow("Face Attendance", frame)
    key = cv2.waitKey(1)

    if key == ord('i') and name != "Unknown":
        mark_attendance(name, "Punch In")
        print(f"{name} punched in")

    elif key == ord('o') and name != "Unknown":
        mark_attendance(name, "Punch Out")
        print(f"{name} punched out")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
def is_live_face(prev_frame, current_frame, threshold=5000):
    diff = cv2.absdiff(prev_frame, current_frame)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
    motion_score = np.sum(thresh)
    return motion_score > threshold

prev_frame = None

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    if prev_frame is not None:
        if not is_live_face(prev_frame, frame):
            cv2.putText(frame, "Spoof Detected", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Attendance", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            prev_frame = frame
            continue

    prev_frame = frame

