import time
import cv2
import mediapipe as mp
import math
import numpy as np

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

class FingerDrawer:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.font_face = cv2.FONT_HERSHEY_SIMPLEX
        self.line_type = cv2.LINE_AA
        self.width, self.height = 1080, 720
        self.canvas = np.zeros((self.height, self.width, 4), dtype='uint8')
        self.dots = []
        self.color = (0, 0, 255, 255)
        self.init_color_palette()

    def init_color_palette(self):
        # 在畫面上方放入紅色、綠色和藍色正方形
        cv2.rectangle(self.canvas, (20, 20), (60, 60), (0, 0, 255, 255), -1)
        cv2.rectangle(self.canvas, (80, 20), (120, 60), (0, 255, 0, 255), -1)
        cv2.rectangle(self.canvas, (140, 20), (180, 60), (255, 0, 0, 255), -1)

    # 根據兩點的座標，計算角度
    def vector_2d_angle(self, v1, v2):
        v1_x, v1_y = v1
        v2_x, v2_y = v2
        try:
            angle = math.degrees(math.acos((v1_x * v2_x + v1_y * v2_y) / (((v1_x ** 2 + v1_y ** 2) ** 0.5) * ((v2_x ** 2 + v2_y ** 2) ** 0.5))))
        except:
            angle = 180
        return angle

    # 根據傳入的 21 個節點座標，得到該手指的角度
    def hand_angle(self, hand_):
        angle_list = []
        # thumb 大拇指角度
        angle_list.append(self.vector_2d_angle(
            ((int(hand_[0][0]) - int(hand_[2][0])), (int(hand_[0][1]) - int(hand_[2][1]))),
            ((int(hand_[3][0]) - int(hand_[4][0])), (int(hand_[3][1]) - int(hand_[4][1])))))
        # index 食指角度
        angle_list.append(self.vector_2d_angle(
            ((int(hand_[0][0]) - int(hand_[6][0])), (int(hand_[0][1]) - int(hand_[6][1]))),
            ((int(hand_[7][0]) - int(hand_[8][0])), (int(hand_[7][1]) - int(hand_[8][1])))))
        # middle 中指角度
        angle_list.append(self.vector_2d_angle(
            ((int(hand_[0][0]) - int(hand_[10][0])), (int(hand_[0][1]) - int(hand_[10][1]))),
            ((int(hand_[11][0]) - int(hand_[12][0])), (int(hand_[11][1]) - int(hand_[12][1])))))
        # ring 無名指角度
        angle_list.append(self.vector_2d_angle(
            ((int(hand_[0][0]) - int(hand_[14][0])), (int(hand_[0][1]) - int(hand_[14][1]))),
            ((int(hand_[15][0]) - int(hand_[16][0])), (int(hand_[15][1]) - int(hand_[16][1])))))
        # pink 小拇指角度
        angle_list.append(self.vector_2d_angle(
            ((int(hand_[0][0]) - int(hand_[18][0])), (int(hand_[0][1]) - int(hand_[18][1]))),
            ((int(hand_[19][0]) - int(hand_[20][0])), (int(hand_[19][1]) - int(hand_[20][1])))))
        return angle_list

    # 根據手指角度的串列內容，返回對應的手勢名稱
    def hand_pos(self, finger_angle):
        f1, f2, f3, f4, f5 = finger_angle
        # 小於 50 表示手指伸直，大於等於 50 表示手指捲縮
        if f1 >= 50 and f2 < 50 and f3 >= 50 and f4 >= 50 and f5 >= 50:
            return '1'
        else:
            return ''

    def process_landmarks(self, results):
        if results.hand_landmarks:
            for hand_landmarks in results.hand_landmarks:
                finger_points = [(int(lm.x * self.width), int(lm.y * self.height)) for lm in hand_landmarks]
                if finger_points:
                    finger_angle = self.hand_angle(finger_points)  # 計算手指角度，回傳長度為 5 的串列
                    text = self.hand_pos(finger_angle)  # 取得手勢所回傳的內容
                    if text == '1':
                        fx, fy = finger_points[8]  # 如果手勢為 1，記錄食指末端的座標
                        if 20 <= fy <= 60:
                            if 20 <= fx <= 60:
                                self.color = (0, 0, 255, 255)  # 如果食指末端碰到紅色，顏色改成紅色
                            elif 80 <= fx <= 120:
                                self.color = (0, 255, 0, 255)  # 如果食指末端碰到綠色，顏色改成綠色
                            elif 140 <= fx <= 180:
                                self.color = (255, 0, 0, 255)  # 如果食指末端碰到藍色，顏色改成藍色
                        else:
                            self.dots.append([fx, fy])  # 記錄食指座標
                            if len(self.dots) > 1:
                                start_point = tuple(self.dots[-2])
                                end_point = tuple(self.dots[-1])
                                cv2.line(self.canvas, start_point, end_point, self.color, 5)  # 在黑色畫布上畫圖
                    else:
                        self.dots = []  # 如果換成別的手勢，清空 dots

    def update_frame(self, frame, canvas):
        frame_BGRA = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)  # 畫圖的影像轉換成 BGRA 色彩

        # 將影像和黑色畫布合成
        for j in range(self.width):
            frame_BGRA[:, j, 0] = frame_BGRA[:, j, 0] * (1 - canvas[:, j, 3] / 255) + canvas[:, j, 0] * (canvas[:, j, 3] / 255)
            frame_BGRA[:, j, 1] = frame_BGRA[:, j, 1] * (1 - canvas[:, j, 3] / 255) + canvas[:, j, 1] * (canvas[:, j, 3] / 255)
            frame_BGRA[:, j, 2] = frame_BGRA[:, j, 2] * (1 - canvas[:, j, 3] / 255) + canvas[:, j, 2] * (canvas[:, j, 3] / 255)

        return frame_BGRA

    def run(self):
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path='./models/hand_landmarker.task'),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        with HandLandmarker.create_from_options(options) as landmarker:
            if not self.cap.isOpened():
                print("Cannot open camera")
                return

            while True:
                ret, frame = self.cap.read()

                if not ret:
                    print("Cannot receive frame")
                    break

                frame = cv2.resize(frame, (self.width, self.height))  # 縮小尺寸，加快處理效率
                frame = cv2.flip(frame, 1)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                hand_landmarker_result = landmarker.detect_for_video(mp_image, int(time.time() * 1000))

                self.process_landmarks(hand_landmarker_result)

                frame = self.update_frame(frame, self.canvas)

                cv2.imshow('WYDIWYG', frame)

                keyboard = cv2.waitKey(5)
                if keyboard == ord('q'):
                    break
                # 按下 r 重置畫面
                if keyboard == ord('r'):
                    self.canvas = np.zeros((self.height, self.width, 4), dtype='uint8')
                    self.init_color_palette()

        self.cap.release()
        cv2.destroyAllWindows()
