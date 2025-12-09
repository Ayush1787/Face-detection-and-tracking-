import cv2 # type: ignore
import time
import mediapipe as mp # type: ignore

cap = cv2.VideoCapture(0)

while True:
    success , img = cap.read()
    cv2.imshow("Image", img)
    cv2.waitKey(1) 