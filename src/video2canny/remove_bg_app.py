import gradio
import remove_bg

# create a gradio app that uses remove_bg to remove background.
# it takes an input image file
# and returns an output image

app = gradio.Interface(
    fn=remove_bg._rb,
    inputs="image",
    outputs="image",
)

app.launch()