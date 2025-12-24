import os
import glob

from PIL import Image, ImageDraw, ImageFont
import string

# Configuration
FONT_DIR = "/usr/share/fonts"
OUTPUT_DIR = "dataset/images"
IMAGE_SIZE = 64
FONT_SIZE = 48  # Slightly smaller than image size to avoid clipping

# printable characters 32-126
CHARS = [chr(i) for i in range(32, 127)]

def get_fonts():
    # Recursively find all ttf and otf files
    fonts = glob.glob(os.path.join(FONT_DIR, "**", "*.[ot]tf"), recursive=True)
    return fonts

def render_char(char, font_path):
    try:
        font = ImageFont.truetype(font_path, FONT_SIZE)
    except Exception as e:
        print(f"Error loading font {font_path}: {e}")
        return None

    # Create empty image
    image = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), color=0) # black background
    draw = ImageDraw.Draw(image)

    # Get bounding box to center text
    bbox = draw.textbbox((0, 0), char, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Calculate position
    x = (IMAGE_SIZE - text_width) // 2
    y = (IMAGE_SIZE - text_height) // 2

    # Draw text (white on black)
    draw.text((x, y - bbox[1]), char, font=font, fill=255)
    
    return image

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    fonts = get_fonts()
    print(f"Found {len(fonts)} fonts.")

    count = 0
    metadata = [] # List to store (filename, char, font)

    for font_path in fonts:
        font_name = os.path.basename(font_path)
        for char in CHARS:
            try:
                img = render_char(char, font_path)
                if img:
                    # Create safe filename
                    # hex representation of char to avoid filesystem issues
                    char_hex = f"{ord(char):02x}"
                    filename = f"{font_name}_{char_hex}.png"
                    save_path = os.path.join(OUTPUT_DIR, filename)
                    
                    img.save(save_path)
                    metadata.append(f"{filename},{char},{font_name}")
                    count += 1
            except Exception as e:
                # Some fonts might fail for specific chars
                pass

        if count % 1000 == 0:
            print(f"Generated {count} images...")

    # Save metadata
    with open("dataset/metadata.csv", "w") as f:
        f.write("filename,char,font_name\n")
        f.write("\n".join(metadata))

    print(f"Finished. Generated {count} images in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
