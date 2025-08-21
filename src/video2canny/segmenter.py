from transformers import pipeline
import torch
import numpy as np
import os
from PIL import Image

def generate_mask(image_path, invert = False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = pipeline(
        "mask-generation",
        model="facebook/sam2-hiera-large",
        device=device
    )
    outputs = generator(image_path,  points_per_batch=16)
    masks = outputs["masks"]

    if len(masks) > 0:
        mask = masks[0]
        if invert:
            mask = ~mask
        img_array = np.array(
            mask.cpu().numpy(), dtype=np.uint8) * 255
        output_path = "mask_" + os.path.basename(image_path)
        Image.fromarray(img_array, mode='L').save(output_path)
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

