import cv2
import mediapipe as mp
import time


class HandDetector:
    def __init__(self,
                 mode=False,
                 max_hands=2,
                 detection_confidence=0.5,
                 tracking_confidence=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        self.results = None

        self.media_hands = mp.solutions.hands
        self.hands = self.media_hands.Hands(
            self.mode,
            self.max_hands,
            self.detection_confidence,
            self.tracking_confidence
        )
        self.media_draw = mp.solutions.drawing_utils

    def find_hands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for hand_lms in self.results.multi_hand_landmarks:
                if draw:
                    # for id, lm in enumerate(hand_lms.landmark):
                    #     # print(id, lm)
                    #     height, width, c = img.shape
                    #     cx, cy = int(lm.x * width), int(lm.y * height)
                    #     print(id, cx, cy)
                    #     cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
                    self.media_draw.draw_landmarks(
                        img,
                        hand_lms,
                        self.media_hands.HAND_CONNECTIONS
                    )
        return img

    def find_position(self, img, hand_index=0, draw=True) -> list:
        lm_list = []
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[hand_index]
            for id, lm in enumerate(hand.landmark):
                height, width, c = img.shape
                cx, cy = int(lm.x * width), int(lm.y * height)
                lm_list.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
        return lm_list


def main():
    p_time = 0
    c_time = 0

    cap = cv2.VideoCapture(0)

    detector = HandDetector()

    while True:
        success, img = cap.read()

        img = detector.find_hands(img)

        lm_list = detector.find_position(img)

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


if __name__ == "__main__":
    main()
