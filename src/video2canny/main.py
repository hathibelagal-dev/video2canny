import cv2
from PIL import Image
import numpy as np
import os
import sys

def canny_video(input_path, output_path, low_threshold=100, high_threshold=200):
    ext = os.path.splitext(input_path)[1].lower()

    frames = []
    fps = 30  # default for GIFs unless overwritten

    if ext == ".gif":
        # Handle GIF using Pillow
        gif = Image.open(input_path)
        fps = gif.info.get('duration', 100)  # duration in ms/frame
        fps = 1000 / fps if fps > 0 else 10  # convert to FPS
        try:
            while True:
                frame = gif.convert("RGB")
                frame_np = np.array(frame)
                frames.append(cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR))
                gif.seek(gif.tell() + 1)
        except EOFError:
            pass

    else:
        # Handle mp4 or other video formats with OpenCV
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {input_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

    if not frames:
        raise ValueError("No frames read from input file.")

    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)

    for frame in frames:        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_enhanced = clahe.apply(gray)
        edges = cv2.Canny(gray_enhanced, low_threshold, high_threshold)
        out.write(edges)

    out.release()
    print(f"Processed video saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python canny_video.py <input_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_file = f"{base_name}_canny.mp4"
    canny_video(input_file, output_file, 100, 200)

