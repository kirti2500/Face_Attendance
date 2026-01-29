import tkinter as tk
from tkinter import simpledialog, messagebox
import cv2
import face_recognition
import numpy as np
import os
import pickle
from datetime import datetime
import shutil
import dlib
from scipy.spatial import distance
# ====== Config ======
ADMIN_PASSWORD = "admin123"
DATA_DIR = "data"
ENCODINGS_FILE = "encodings/encodings.pkl"
ATTENDANCE_FILE = "attendance/attendance.csv"

os.makedirs("data", exist_ok=True)
os.makedirs("encodings", exist_ok=True)
os.makedirs("attendance", exist_ok=True)

# ====== Liveness Models ======
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # download this file


# ====== Utility Functions ======
def admin_auth(root):
    pwd = simpledialog.askstring("Admin Login", "Enter admin password:", show="*", parent=root)
    if pwd != ADMIN_PASSWORD:
        messagebox.showerror("Access Denied", "Incorrect password", parent=root)
        return False
    return True

def load_encodings():
    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, "rb") as f:
            return pickle.load(f)
    return {"encodings": [], "names": []}

def save_encodings(data):
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump(data, f)

def mark_attendance(name):
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")
    status = "IN/OUT"
    with open(ATTENDANCE_FILE, "a", newline="") as f:
        import csv
        writer = csv.writer(f)
        writer.writerow([name, date, time, status])

def delete_user(user_id):
    if not admin_auth(root):
        return
    user_path = os.path.join(DATA_DIR, user_id)
    if os.path.exists(user_path):
        shutil.rmtree(user_path)
    data = load_encodings()
    new_enc = []
    new_names = []
    for enc, name in zip(data["encodings"], data["names"]):
        if name != user_id:
            new_enc.append(enc)
            new_names.append(name)
    data["encodings"] = new_enc
    data["names"] = new_names
    save_encodings(data)
    messagebox.showinfo("Deleted", f"User {user_id} deleted successfully")

def rename_user(old_name, new_name):
    if not admin_auth(root):
        return
    old_path = os.path.join(DATA_DIR, old_name)
    new_path = os.path.join(DATA_DIR, new_name)
    if os.path.exists(old_path):
        os.rename(old_path, new_path)
    data = load_encodings()
    data["names"] = [new_name if name == old_name else name for name in data["names"]]
    save_encodings(data)
    messagebox.showinfo("Renamed", f"{old_name} renamed to {new_name}")

# ====== Spoof Detection ======
def is_live_face(prev_frame, current_frame, threshold=1200):
    diff = cv2.absdiff(prev_frame, current_frame)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
    motion_score = np.sum(thresh)
    return motion_score > threshold

# ====== Face Recognition & Attendance ======
def start_recognition():
    data = load_encodings()
    known_encodings = data["encodings"]
    known_names = data["names"]

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    prev_frame = None
    blink_counter=0


    while True:
        ret, frame = cap.read()
        blink_counter = detect_blink(frame, blink_counter)

        if not ret:
            continue
        small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        spoof_warning = False
        if prev_frame is not None:
            if not is_live_face(prev_frame, frame):
                spoof_warning = True
        prev_frame = frame.copy()

        # ====== Recognition ======
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding, loc in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.55)
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            name = "Unknown"
            if len(face_distances) > 0:
                best_match = np.argmin(face_distances)
                if matches[best_match]:
                   if blink_counter < 1:
                       name = "Blink Required"
                   else:
                       name = known_names[best_match]
                       mark_attendance(name)


            top, right, bottom, left = [v*4 for v in loc]
            cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
            cv2.putText(frame, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0),2)

        # ====== Spoof warning overlay ======
        if spoof_warning:
            cv2.putText(frame, "Spoof Detected!", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Attendance", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ====== UI ======
root = tk.Tk()
root.title("Face Attendance System")
root.geometry("400x350")

tk.Label(root, text="Face Attendance System", font=("Arial", 16)).pack(pady=20)
tk.Button(root, text="Start Attendance", width=25, command=start_recognition).pack(pady=10)

def admin_panel_ui():
    if not admin_auth(root):
        return
    panel = tk.Toplevel(root)
    panel.title("Admin Panel")
    panel.geometry("300x250")
    tk.Label(panel, text="Admin Panel", font=("Arial", 14)).pack(pady=10)
    tk.Button(panel, text="Delete User", width=20, command=lambda: delete_user(simpledialog.askstring("Delete User", "Enter user ID:", parent=panel))).pack(pady=10)
    tk.Button(panel, text="Rename User", width=20, command=lambda: rename_user(simpledialog.askstring("Old Name", "Enter old name:", parent=panel), simpledialog.askstring("New Name", "Enter new name:", parent=panel))).pack(pady=10)

tk.Button(root, text="Admin Panel", width=25, command=admin_panel_ui).pack(pady=10)

root.mainloop()
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def detect_blink(frame, blink_counter):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        shape = predictor(gray, face)
        coords = np.array([[p.x, p.y] for p in shape.parts()])

        left_eye = coords[36:42]
        right_eye = coords[42:48]

        ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0

        if ear < 0.21:
            blink_counter += 1

    return blink_counter
