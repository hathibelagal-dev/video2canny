from transformers import Sam2Processor, Sam2Model
import torch
import numpy as np
import os
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

def generate_mask(image_path, invert = False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Sam2Model.from_pretrained("facebook/sam2-hiera-large").to(device)
    processor = Sam2Processor.from_pretrained("facebook/sam2-hiera-large")
    raw_image = Image.open(image_path)

    input_points = [[[[512, 512]]]]
    input_labels = [[[1]]]

    inputs = processor(images=[raw_image], input_points=input_points, input_labels=input_labels, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]

    if len(masks) > 0:
        print(f"Found {len(masks)} masks")
        mask = masks[0]
        if invert:
            mask = ~mask
        img_array = np.array(
            mask.cpu().numpy(), dtype=np.uint8) * 255
        print(img_array.shape)
        output_path = "mask_" + os.path.basename(image_path)
        Image.fromarray(img_array[2], mode='L').save(output_path)
        print(f"Mask saved to {output_path}")
    else:
        print("No masks found")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python mask_generator.py <image_path> [invert]")
        sys.exit(1)
    else:
        image_path = sys.argv[1]
        invert = False
        if len(sys.argv) == 3:
            if sys.argv[2] == "invert":
                invert = True
        generate_mask(image_path, invert)

