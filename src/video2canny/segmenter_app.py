import gradio as gr
from segmenter import generate_mask

def process(src_img, point_x, point_y):
    output = generate_mask(raw_image=src_img, x=point_x, y=point_y, invert=True)
    return output

app = gr.Interface(
    fn=process,
    inputs=["image", "number", "number"],
    outputs=["image"],
)

app.launch()
