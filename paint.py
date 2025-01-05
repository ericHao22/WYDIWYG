import cv2
import numpy as np
import os
from datetime import datetime
from gradio_client import Client, handle_file

class SmartPainter:
    def __init__(self):
        self.result_dir = self.init_result_dir()

    def init_result_dir(self):
        result_dir = os.path.join('.', 'results', datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(result_dir, exist_ok=True)
        return result_dir

    def paint(self, sketch_image: np.ndarray):
        sketch_path = self.save_sketch_image(sketch_image)
        sketch_prompt = self.get_sketch_prompt(sketch_path)
        painting_image = self.paint_sketch(sketch_path, sketch_prompt)

        return painting_image

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
            image_resolution=768,
            preprocess_resolution=512,
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
