import time
import cv2
import mediapipe as mp
import math
import numpy as np
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmark

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

class FingerDrawer:
    def __init__(self, width=1080, height=720, fourcc='MJPG'):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.width, self.height = width, height
        self.font_face = cv2.FONT_HERSHEY_SIMPLEX
        self.line_type = cv2.LINE_AA
        self.canvas = np.zeros((self.height, self.width, 4), dtype='uint8')
        self.dots = []
        self.color = (0, 0, 255, 255) # 畫筆預設為紅色
        self.init_video_capture(width, height, fourcc)
        # self.init_color_palette()
        self.eraser_mode = False

    def init_video_capture(self, width, height, fourcc):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, 30)

        self.cap = cap

    def init_color_palette(self):
        # 在畫面上方放入紅色、綠色和藍色正方形色塊，用來選擇畫筆顏色
        # 目前暫時不使用，如要使用需要加上在手指進入色塊時改變 self.color 的邏輯
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
        wrist = hand_[HandLandmark.WRIST]

        # thumb 大拇指角度
        thumb_mcp = hand_[HandLandmark.THUMB_MCP]
        thumb_ip = hand_[HandLandmark.THUMB_IP]
        thumb_tip = hand_[HandLandmark.THUMB_TIP]
        angle_list.append(self.vector_2d_angle(
            ((wrist[0] - thumb_mcp[0]), (wrist[1] - thumb_mcp[1])),
            ((thumb_ip[0] - thumb_tip[0]), (thumb_ip[1] - thumb_tip[1]))
        ))

        # index 食指角度
        index_pip = hand_[HandLandmark.INDEX_FINGER_PIP]
        index_dip = hand_[HandLandmark.INDEX_FINGER_DIP]
        index_tip = hand_[HandLandmark.INDEX_FINGER_TIP]
        angle_list.append(self.vector_2d_angle(
            ((wrist[0] - index_pip[0]), (wrist[1] - index_pip[1])),
            ((index_dip[0] - index_tip[0]), (index_dip[1] - index_tip[1]))
        ))

        # middle 中指角度
        middle_pip = hand_[HandLandmark.MIDDLE_FINGER_PIP]
        middle_dip = hand_[HandLandmark.MIDDLE_FINGER_DIP]
        middle_tip = hand_[HandLandmark.MIDDLE_FINGER_TIP]
        angle_list.append(self.vector_2d_angle(
            ((wrist[0] - middle_pip[0]), (wrist[1] - middle_pip[1])),
            ((middle_dip[0] - middle_tip[0]), (middle_dip[1] - middle_tip[1]))
        ))

        # ring 無名指角度
        ring_pip = hand_[HandLandmark.RING_FINGER_PIP]
        ring_dip = hand_[HandLandmark.RING_FINGER_DIP]
        ring_tip = hand_[HandLandmark.RING_FINGER_TIP]
        angle_list.append(self.vector_2d_angle(
            ((wrist[0] - ring_pip[0]), (wrist[1] - ring_pip[1])),
            ((ring_dip[0] - ring_tip[0]), (ring_dip[1] - ring_tip[1]))
        ))

        # pinky 小拇指角度
        pinky_pip = hand_[HandLandmark.PINKY_PIP]
        pinky_dip = hand_[HandLandmark.PINKY_DIP]
        pinky_tip = hand_[HandLandmark.PINKY_TIP]
        angle_list.append(self.vector_2d_angle(
            ((wrist[0] - pinky_pip[0]), (wrist[1] - pinky_pip[1])),
            ((pinky_dip[0] - pinky_tip[0]), (pinky_dip[1] - pinky_tip[1]))
        ))

        return angle_list

    # 根據手指角度的串列內容，返回對應的手勢名稱
    def hand_pos(self, finger_angle):
        f1, f2, f3, f4, f5 = finger_angle
        # 小於 50 表示手指伸直，大於等於 50 表示手指捲縮
        if f1 >= 50 and f2 < 50 and f3 >= 50 and f4 >= 50 and f5 >= 50:
            return 'draw'
        elif f1 < 50 and f2 >= 50 and f3 >= 50 and f4 >= 50 and f5 >= 50:
            return 'eraser'
        else:
            return ''

    def process_landmarks(self, results):
        if not results.hand_landmarks:
            return

        finger_points = [(int(lm.x * self.width), int(lm.y * self.height)) for lm in results.hand_landmarks[0]]

        if not finger_points:
            return

        finger_angle = self.hand_angle(finger_points)  # 計算手指角度，回傳長度為 5 的串列
        text = self.hand_pos(finger_angle)  # 取得手勢所回傳的內容

        if text == 'draw':
            fx, fy = finger_points[8]  # 如果手勢為 1，記錄食指末端的座標
            self.dots.append([fx, fy])  # 記錄食指座標
            if len(self.dots) > 1:
                start_point = tuple(self.dots[-2])
                end_point = tuple(self.dots[-1])
                cv2.line(self.canvas, start_point, end_point, self.color, 5)  # 在黑色畫布上畫圖

        elif text == 'eraser':
            fx, fy = finger_points[4]  # 如果手勢為 'eraser'，記錄大拇指末端的座標
            eraser_color = (0, 0, 0, 0)
            self.dots.append([fx, fy])
            if len(self.dots) > 1:
                start_point = tuple(self.dots[-2])
                end_point = tuple(self.dots[-1])
                cv2.line(self.canvas, start_point, end_point, eraser_color, 20)

        else:
            self.dots = []  # 如果換成別的手勢，清空 dots

    def draw_landmarks(self, frame, results):
        if not results.hand_landmarks:
            return frame

        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in results.hand_landmarks[0]
        ])

        mp.solutions.drawing_utils.draw_landmarks(
            frame,
            hand_landmarks_proto,
            mp.solutions.hands.HAND_CONNECTIONS,
        )

        return frame

    def update_frame(self, frame, canvas):
        frame_BGRA = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)  # 畫圖的影像轉換成 BGRA 色彩

        # 將影像和黑色畫布合成
        for j in range(self.width):
            frame_BGRA[:, j, 0] = frame_BGRA[:, j, 0] * (1 - canvas[:, j, 3] / 255) + canvas[:, j, 0] * (canvas[:, j, 3] / 255)
            frame_BGRA[:, j, 1] = frame_BGRA[:, j, 1] * (1 - canvas[:, j, 3] / 255) + canvas[:, j, 1] * (canvas[:, j, 3] / 255)
            frame_BGRA[:, j, 2] = frame_BGRA[:, j, 2] * (1 - canvas[:, j, 3] / 255) + canvas[:, j, 2] * (canvas[:, j, 3] / 255)

        return frame_BGRA

    def show_start_screen(self):
        start_screen = np.zeros((self.height, self.width, 3), dtype='uint8')
        start_text = [
            " _       ____  __    ____     ____ _       ____  __   ______",
            "| |     / /\ \/ /   / __ \   /  _/| |     / /\ \/ /  / ____/",
            "| | /| / /  \  /   / / / /   / /  | | /| / /  \  /  / / __",
            "| |/ |/ /   / /   / /_/ /  _/ /   | |/ |/ /   / /  / /_/ /",
            "|_/|__/   /_/   /_____/  /___/   |__/|__/   /_/   \____/"
        ]
        y0, dy = self.height // 2 - 100, 30
        for i, line in enumerate(start_text):
            y = y0 + i * dy
            cv2.putText(start_screen, line, (50, y), self.font_face, 0.7, (255, 255, 255), 2, self.line_type)
        cv2.putText(start_screen, 'Press any key to start drawing', (50, y0 + len(start_text) * dy + 20), self.font_face, 1, (255, 255, 255), 2, self.line_type)
        cv2.imshow('WYDIWYG', start_screen)
        cv2.waitKey(0)  # 等待按下任何按鍵

    def run(self):
        self.show_start_screen()

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path='./models/hand_landmarker.task'),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        sketch_image = None

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

                # frame = self.draw_landmarks(frame, hand_landmarker_result) # debug
                frame = self.update_frame(frame, self.canvas)

                cv2.imshow('WYDIWYG', frame)

                keyboard = cv2.waitKey(5)

                # 按下 q 退出畫面
                if keyboard == ord('q'):
                    break

                # 按下 s 完成草稿
                if keyboard == ord('s'):
                    sketch_image = cv2.cvtColor(self.canvas, cv2.COLOR_BGRA2BGR)
                    sketch_image[np.any(sketch_image[:, :, :3] != [0, 0, 0], axis=-1)] = [255, 255, 255] # 將圖片非黑色的部分轉為白色
                    sketch_image = cv2.bitwise_not(sketch_image) # 將圖片轉為白底黑線
                    break

                # 按下 r 重置畫面
                if keyboard == ord('r'):
                    self.canvas = np.zeros((self.height, self.width, 4), dtype='uint8')
                    # self.init_color_palette()
                    self.eraser_mode = False

        self.cap.release()
        cv2.destroyAllWindows()

        return sketch_image
