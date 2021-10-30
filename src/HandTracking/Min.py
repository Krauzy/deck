import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

media_hands = mp.solutions.hands
hands = media_hands.Hands()

while True:
    success, img = cap.read()
    print(img)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
