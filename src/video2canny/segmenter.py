from transformers import pipeline
import torch
import sys
import numpy as np
import os
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
generator = pipeline(
    "mask-generation", model="facebook/sam2-hiera-large", device=device)

image_path = sys.argv[1]
outputs = generator(image_path,  points_per_batch=16)
masks = outputs["masks"]

if len(masks) > 0:
    main_mask = masks[0]
    img_array = np.array(outputs["masks"][0], dtype=np.uint8) * 255
    output_path = "mask_" + os.path.basename(image_path)
    Image.fromarray(img_array, mode='L').save(output_path)
    print(f"Mask saved to {output_path}")

else:
    print("No masks found")
