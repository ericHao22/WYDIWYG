from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import torch

# 載入模型文件夾
model_path = r"C:\Users\tsce8\WYDIWYG\stable_diffusion_img2img" # 替換為你的模型文件夾路徑
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_path)
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")  # 使用 GPU 或 CPU

# 載入輸入圖像
input_image_path = "output_canvas.png"  # 确保路径与保存的画布一致 # 替換為你的輸入圖像路徑
init_image = Image.open(input_image_path).convert("RGB").resize((512, 512))

# 設置生成參數
prompt = "A sculpture which if made of stone"
negative_prompt = "blurry, low quality"
strength = 0.75  # 控制原圖與生成圖的相似程度（0~1）
guidance_scale = 7.5  # 控制生成與提示詞的匹配程度

# 進行圖生圖推理
output = pipe(prompt=prompt, image=init_image, strength=strength, guidance_scale=guidance_scale, num_inference_steps=20)

# # 保存生成結果
# output_image = output.images[0]
# output_image.save("output_image.png")
# print("圖生圖任務完成，生成圖像已保存為 output_image.png")

# 新代码生成多张图像
for i in range(5):
    output = pipe(prompt=prompt, image=init_image, strength=strength, guidance_scale=guidance_scale, num_inference_steps=20)
    output_image = output.images[0]
    output_image.save(f"output_image_{i}.png")
print("生成的图像已保存！")