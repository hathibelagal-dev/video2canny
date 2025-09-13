from rembg import remove
from PIL import Image

def remove_background(input_path, output_path):
    with open(input_path, 'rb') as i:
        with open(output_path, 'wb') as o:
            input = i.read()
            output = remove(input)
            o.write(output)

def _rb(raw_image):
    return remove(raw_image)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python remove_background.py <input_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = "output.png"
    _rb(input_file, output_file)
    print(f"Background removed and saved to {output_file}")