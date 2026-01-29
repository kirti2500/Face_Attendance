# Face Attendance System

A Python-based face recognition attendance system that uses **computer vision** to automatically mark attendance of registered users. This project includes user registration, face recognition, and attendance logging.

---

## **Table of Contents**
- [Overview](#overview)  
- [Model and Approach](#model-and-approach)  
- [Training Process](#training-process)  
- [Accuracy Expectations](#accuracy-expectations)  
- [Known Failure Cases](#known-failure-cases)  
- [Installation](#installation)  
- [Usage](#usage)  
- [GitHub Setup](#github-setup)  

---

## **Overview**
This system uses facial recognition to identify registered users and record their attendance automatically. Key features:
- Register new users with multiple face images
- Encode faces for recognition
- Real-time face recognition using a webcam
- Attendance recorded in a CSV file
- Basic GUI using Tkinter

---

## **Model and Approach**
- **Library Used:** [face_recognition](https://github.com/ageitgey/face_recognition) (built on dlib’s face recognition model)
- **Model Approach:**
  - Detect faces in images using **HOG + SVM** or CNN-based detector
  - Extract **128-dimensional face embeddings**
  - Compare embeddings with known faces using **Euclidean distance**
  - If distance < threshold (default 0.6), face is recognized

**Why this approach:**  
This method is fast, works in real-time, and is robust to small changes in lighting or orientation.

---

## **Training Process**
1. **Collect images:** Take multiple images of each person (recommended 30–50 per person)
2. **Encode faces:** Run `encode_faces.py` to convert face images into embeddings
3. **Save encodings:** Encodings stored in `encodings/faces.pkl`  
   These embeddings are used during recognition; no heavy training required, only encoding

---

## **Accuracy Expectations**
- Works well for **frontal faces under good lighting**
- Recognition accuracy: ~90–92% for clear images
- Attendance logging is reliable if face is properly detected
- Real-time performance: 5–15 FPS depending on CPU/GPU

---

## **Known Failure Cases**
- Identical twins or very similar faces
- Recognition may fail if only **1–2 images** are registered per person
- Can be spoofed with photos or videos (consider anti-spoofing measures for production)

---

## **Installation**
1. Clone the repository:
```bash
git clone https://github.com/<your-username>/face_attendance.git
cd face_attendance
