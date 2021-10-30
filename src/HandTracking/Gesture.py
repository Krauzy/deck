import cv2
import mediapipe as mp
import numpy as np
import time
from Module import HandDetector
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

width_cam, height_cam = 1280, 720

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width_cam)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height_cam)

detector = HandDetector(detection_confidence=0.4)


devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None
)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volume_range = volume.GetVolumeRange()
# volume.SetMasterVolumeLevel(-10, None)

min_volume, max_volume, inc_volume = volume_range


p_time = 0

while cap.isOpened():
    success, img = cap.read()

    img = detector.find_hands(img, draw=True)
    lm_list = detector.find_position(img, draw=False)
    # print(lm_list)
    # vol_per = 0
    if len(lm_list) != 0:

        _, x1, y1 = lm_list[4]
        _, x2, y2 = lm_list[8]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        cv2.circle(img, (x1, y1), 15, (255, 0, 0), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (255, 0, 0), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

        length = math.hypot(x2 - x1, y2 - y1)
        # print(length)

        vol = np.interp(length, [15, 250], [min_volume, max_volume])
        # vol_bar = np.interp(length, [15, 250], [400, 150])
        # vol_per = np.interp(length, [15, 250], [0, 100])

        volume.SetMasterVolumeLevel(vol, None)
        # print(vol)

        if length < 15:
            cv2.circle(img, (cx, cy), 15, (0, 0, 255), cv2.FILLED)

    # cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0))
    # cv2.rectangle(img, (50, int(vol_per)), (85, 400), (0, 255, 0), cv2.FILLED)
    # cv2.putText(img, f'{int(vol_per)}%', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)

    c_time = time.time()
    fps = 1 / (c_time - p_time)
    p_time = c_time

    cv2.putText(img, f'FPS: {int(fps)}', (40, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
