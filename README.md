# Face-detection-and-tracking-
AI-powered Computer Vision projects using Python, OpenCV, Dlib, and Mediapipe. Includes face detection, 68-landmark extraction, emotion recognition, multi-object tracking, and hand tracking. Great for learning AI, ML, Deep Learning, Robotics, and IoT.

📌 Project Overview

This repository is a complete collection of real-time Computer Vision programs built using Python, OpenCV, Dlib, Mediapipe, and NumPy.

It includes face detection, 68-point facial landmarks, rule-based emotion recognition, multi-object tracking, and hand tracking.

This project is perfect for students, AI/ML beginners, robotic enthusiasts, and developers building intelligent vision systems.

🚀 Key Features
🧠 Face Detection

Real-time face detection using Haar Cascade

Crops & resizes faces for dataset creation

Ideal for ML preprocessing

👁 68 Facial Landmarks + Emotion Recognition

Uses Dlib’s 68-point predictor

Calculates:

EAR (Eye Aspect Ratio)

MAR (Mouth Aspect Ratio)

Eyebrow distance

Mouth width ratio

Detects emotions:
Happy, Sad, Angry, Neutral, Surprised, Sleepy

🎯 Multi-Object Tracking

Custom Centroid Tracker

Background subtraction using MOG2

Assigns unique IDs to objects

Works like real CCTV surveillance tracking

✋ Hand Tracking (Mediapipe)

Real-time hand detection

Can be used for gesture control, virtual mouse, robotics

🛠️ Tech Stack

Python

OpenCV

Dlib

Mediapipe

NumPy

📂 Project Structure
📁 Computer-Vision-AI
│── face.py                     # Face detection + dataset creation
│── face_68-landmark.py         # Facial landmarks + emotion recognition
│── mulit_faces_detection.py    # Multi-object tracking
│── HandTrackingMain.py         # Hand tracking (Mediapipe)
│── haarcascade_frontalface_default.xml
│── shape_predictor_68_face_landmarks.dat

📦 Installation
1️⃣ Install required libraries
pip install opencv-python dlib mediapipe numpy

2️⃣ Add the Dlib model

Place shape_predictor_68_face_landmarks.dat in the root folder.

▶️ How to Run
Face Detection:
python face.py

Facial Landmarks + Emotion Prediction:
python face_68-landmark.py

Multi-Face/Object Tracking:
python mulit_faces_detection.py

Hand Tracking:
python HandTrackingMain.py
