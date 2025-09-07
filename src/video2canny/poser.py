import gradio as gr
from controlnet_aux.processor import Processor

processor = Processor("openpose_face")

def pose(raw_image):
    processed_image = processor(raw_image)
    return processed_image

app = gr.Interface(
    fn=pose,
    inputs=["image"],
    outputs=["image"],
)

app.launch()