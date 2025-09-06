import gradio as gr
from segmenter import generate_mask

def process_click(image, evt: gr.SelectData):
    if evt.index:
        x, y = evt.index
        return (x, y)
    return (0, 0)

def process(src_img, point_x, point_y, invert):
    output = generate_mask(raw_image=src_img, x=point_x, y=point_y, invert=invert)
    return output

def use_output(output):
    return output

with gr.Blocks() as app:
    src_img = gr.Image(type="pil")
    point_x = gr.Number(label="Point X")
    point_y = gr.Number(label="Point Y")
    invert = gr.Checkbox(label="Invert")
    output = gr.Image(type="pil")
    src_img.select(process_click, [src_img], [point_x, point_y])

    btn1 = gr.Button("Process")
    btn2 = gr.Button("Use")
    btn1.click(process, inputs=[src_img, point_x, point_y, invert], outputs=[output])
    btn2.click(use_output, inputs=[output], outputs=[src_img])

app.launch()
