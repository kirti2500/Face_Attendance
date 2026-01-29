import pandas as pd
from datetime import datetime
import os
import numpy as np

def mark_attendance(name, action):
    file_exists = os.path.isfile("attendance.csv")

    now = datetime.now()
    row = {
        "Name": name,
        "Date": now.strftime("%Y-%m-%d"),
        "Time": now.strftime("%H:%M:%S"),
        "Action": action
    }

    df = pd.DataFrame([row])

    if file_exists:
        df.to_csv("attendance.csv", mode='a', header=False, index=False)
    else:
        df.to_csv("attendance.csv", index=False)

def last_action(name):
    if not os.path.exists("attendance.csv"):
        return None

    df = pd.read_csv("attendance.csv")
    user_df = df[df["Name"] == name]

    if user_df.empty:
        return None

    return user_df.iloc[-1]["Action"]

def is_live_face(prev_loc, curr_loc, threshold=5):
    if prev_loc is None:
        return True

    pt, pr, pb, pl = prev_loc
    ct, cr, cb, cl = curr_loc

    movement = abs(pt - ct) + abs(pl - cl)
    return movement > threshold

# utils.py (update admin_auth)
from tkinter import simpledialog, messagebox

def admin_auth(root):
    pwd = simpledialog.askstring("Admin Login", "Enter admin password:", show="*",
                                 parent=root)
    if pwd != ADMIN_PASSWORD:
        messagebox.showerror("Access Denied", "Incorrect password", parent=root)
        return False
    return True
