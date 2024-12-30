import base64
from mimetypes import guess_type
from gradio_client import Client
import cv2

def local_image_to_data_url(image_path):
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'  # Default MIME type if none is found

    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}", mime_type

input_image_path = './input.png'
client = Client("pharmapsychotic/CLIP-Interrogator")
result = client.predict(
				input_image_path,	# str (filepath or URL to image)
				"ViT-L (best for Stable Diffusion 1.*)",	# str (Option from: ['ViT-L (best for Stable Diffusion 1.*)'])\
				"best",	# str in 'Mode' Radio component
				fn_index=3
)
input_image_prompt = result[0]

print(f'prompt: {input_image_prompt}')

base64_data, mime_type = local_image_to_data_url(input_image_path)
client = Client("hysts/ControlNet-v1-1")
result = client.predict(
		image={"url": base64_data, "mime_type": mime_type},
		prompt=input_image_prompt,
		additional_prompt="best quality, extremely detailed",
		negative_prompt="longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality",
		num_images=1,
		image_resolution=768,
		preprocess_resolution=512,
		num_steps=20,
		guidance_scale=9,
		seed=0,
		preprocessor_name="None",
		api_name="/scribble"
)

output_image_path = result[1]['image']
output_image = cv2.imread(output_image_path)

cv2.imshow('Final Result', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
