import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

media_hands = mp.solutions.hands
hands = media_hands.Hands()
media_draw = mp.solutions.drawing_utils

p_time = 0
c_time = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            for id, lm in enumerate(hand_lms.landmark):
                # print(id, lm)
                height, width, c = img.shape
                cx, cy = int(lm.x * width), int(lm.y * height)
                print(id, cx, cy)
                cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

            media_draw.draw_landmarks(img, hand_lms, media_hands.HAND_CONNECTIONS)

    c_time = time.time()
    fps = 1 / (c_time - p_time)
    p_time = c_time

    cv2.putText(
        img=img,
        text=str(int(fps)),
        org=(10, 70),
        fontFace=cv2.FONT_HERSHEY_PLAIN,
        fontScale=3,
        color=(255, 0, 255),
        thickness=3
    )

    cv2.imshow("Image", img)
    cv2.waitKey(1)
