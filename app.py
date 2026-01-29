import tkinter as tk
import subprocess
import sys

PYTHON_EXEC = sys.executable  

def run_script(script):
    subprocess.Popen([PYTHON_EXEC, script])

root = tk.Tk()
root.title("Face Attendance System")
root.geometry("300x300")

tk.Label(root, text="Face Attendance System",
         font=("Arial", 14)).pack(pady=15)

tk.Button(root, text="Register User",
          width=25, command=lambda: run_script("register.py")).pack(pady=5)

tk.Button(root, text="Encode Faces",
          width=25, command=lambda: run_script("encode_faces.py")).pack(pady=5)

tk.Button(root, text="Punch In / Out",
          width=25, command=lambda: run_script("recognize.py")).pack(pady=5)

tk.Button(root, text="Exit",
          width=25, command=root.quit).pack(pady=15)

root.mainloop()
