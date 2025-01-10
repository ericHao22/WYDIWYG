import queue
import threading
import cv2
import numpy as np
import os
from datetime import datetime
from gradio_client import Client, handle_file

class SmartPainter:
    def __init__(self):
        self.width, self.height = 768, 512 # 此為 hysts/ControlNet-v1-1 模型可接受的最大輸入解析度
        self.result_dir = self.init_result_dir()

    def init_result_dir(self):
        result_dir = os.path.join('.', 'results', datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(result_dir, exist_ok=True)
        return result_dir

    def paint(self, sketch_image: np.ndarray):
        stop_event = threading.Event()
        result_queue = queue.Queue()
        painting_thread = threading.Thread(target=self.generate_painting, args=(sketch_image, stop_event, result_queue,))
        painting_thread.start()

        self.show_processing_screen(stop_event)

        painting_thread.join()
        painting_image = result_queue.get()

        cv2.imshow('WYDIWYG', painting_image)
        cv2.waitKey(0)

        return painting_image

    def generate_painting(self, sketch_image: np.ndarray, stop_event, result_queue):
        sketch_path = self.save_sketch_image(sketch_image)
        sketch_prompt = self.get_sketch_prompt(sketch_path)
        painting_image = self.paint_sketch(sketch_path, sketch_prompt)

        result_queue.put(painting_image)
        stop_event.set()

    def show_processing_screen(self, stop_event):
        base_text = 'Painting'
        font_face = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 2
        text_size = cv2.getTextSize(base_text, font_face, font_scale, thickness)[0]
        text_x = (self.width - text_size[0]) // 2
        text_y = (self.height + text_size[1]) // 2
        dot_count = 1

        while not stop_event.is_set():
            img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            full_text = f'{base_text}{"." * dot_count}'

            cv2.putText(img, full_text, (text_x, text_y), font_face, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            cv2.imshow('WYDIWYG', img)
            cv2.waitKey(500)

            dot_count = (dot_count % 3) + 1

    def save_sketch_image(self, sketch_image: np.ndarray):
        sketch_path = os.path.join(self.result_dir, 'sketch.png')
        cv2.imwrite(sketch_path, sketch_image)
        return sketch_path

    def get_sketch_prompt(self, sketch_path: str):
        client = Client("pharmapsychotic/CLIP-Interrogator")
        result = client.predict(
            sketch_path, # image: filepath or URL to image
            'ViT-L (best for Stable Diffusion 1.*)', # clip_model
            'best',	# mode
            fn_index=3
        )
        prompt = result[0]

        print(f'sketch prompt: {prompt}')

        return prompt

    def paint_sketch(self, sketch_path: str, sketch_prompt: str):
        client = Client("hysts/ControlNet-v1-1")
        result = client.predict(
            image=handle_file(sketch_path),
            prompt=sketch_prompt,
            additional_prompt="best quality, highly detailed, intricate, artistic, realistic textures",
            negative_prompt="blurry, low resolution, distorted, cartoonish, minimalistic, extra digit, fewer digits, cropped, worst quality, low quality",
            num_images=1,
            image_resolution=self.width,
            preprocess_resolution=self.height,
            num_steps=20,
            guidance_scale=9,
            seed=0,
            preprocessor_name="None",
            api_name="/scribble"
        )
        painting_path = os.path.join(self.result_dir, 'painting.webp')

        os.rename(result[1]['image'], painting_path)

        print(f'painting image: {painting_path}')

        return cv2.imread(painting_path)
