import gradio as gradio
import numpy as np
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

processor=BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')
model=BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base')

def caption_image(input_image: np.ndarray):
    raw_image=Image.fromarray(input_image).convert('RGB')
    inputs=processor(images=raw_image, return_tensors='pt')
    outputs=model.generate(**inputs)
    caption=processor.decode(outputs[0], skip_special_tokens=True)
    return caption

iface=gr.Interface(
    fn=caption_image,
    inputs=gr.Image(),
    outputs="text",
    title="Image Captioning Interface"
)

iface.launch()