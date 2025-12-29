"""
Homoglyph Dataset Generator

This script renders ASCII characters into images using system fonts to create a dataset
for training a homoglyph discovery model.

It iterates through all TrueType (.ttf) and OpenType (.otf) fonts in the system
font directory, renders printable ASCII characters, and saves them as grayscale PNGs.
"""

import os
import glob
from PIL import Image, ImageDraw, ImageFont

# Configuration
FONT_DIR = "/usr/share/fonts"  # Directory containing system fonts
OUTPUT_DIR = "dataset/images"  # Output directory for generated images
IMAGE_SIZE = 64                # Target image dimensions (64x64)
FONT_SIZE = 48                 # Font size (slightly smaller to fit in the image)

# Printable ASCII characters (codes 32-126)
CHARS = [chr(i) for i in range(32, 127)]

def get_fonts():
    """
    Recursively finds all .ttf and .otf font files in the FONT_DIR.
    
    Returns:
        list: A list of file paths to available fonts.
    """
    fonts = glob.glob(os.path.join(FONT_DIR, "**", "*.[ot]tf"), recursive=True)
    return fonts

def render_char(char, font_path):
    """
    Renders a single character using a specific font into a centered 64x64 grayscale image.

    Args:
        char (str): The character to render.
        font_path (str): Path to the font file.

    Returns:
        PIL.Image: The resulting image, or None if rendering failed.
    """
    try:
        font = ImageFont.truetype(font_path, FONT_SIZE)
    except Exception as e:
        # Font loading might fail for corrupt files or unsupported formats
        print(f"Error loading font {font_path}: {e}")
        return None

    # Create empty black image (L mode = 8-bit pixels, black and white)
    image = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), color=0) 
    draw = ImageDraw.Draw(image)

    # Get bounding box to calculate text size for centering
    # bbox returns (left, top, right, bottom)
    bbox = draw.textbbox((0, 0), char, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Calculate top-left position to center the text
    x = (IMAGE_SIZE - text_width) // 2
    y = (IMAGE_SIZE - text_height) // 2

    # Draw text in white (255) on black background
    # Adjust y by bbox[1] (ascent) to align properly
    draw.text((x, y - bbox[1]), char, font=font, fill=255)
    
    return image

def main():
    """
    Main execution loop.
    1. Creates output directory.
    2. Finds all fonts.
    3. Renders every character for every font.
    4. Saves images and generates a metadata CSV.
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    fonts = get_fonts()
    print(f"Found {len(fonts)} fonts.")

    count = 0
    metadata = [] # List to store (filename, char, font_name)

    for font_path in fonts:
        font_name = os.path.basename(font_path)
        for char in CHARS:
            try:
                img = render_char(char, font_path)
                if img:
                    # Create safe filename using hex representation of the character
                    # This avoids issues with special characters in filenames (e.g. /, *)
                    char_hex = f"{ord(char):02x}"
                    filename = f"{font_name}_{char_hex}.png"
                    save_path = os.path.join(OUTPUT_DIR, filename)
                    
                    img.save(save_path)
                    metadata.append(f"{filename},{char},{font_name}")
                    count += 1
            except Exception as e:
                # Some fonts might fail for specific chars (e.g. missing glyphs)
                # We simply skip them.
                pass

        if count % 1000 == 0:
            print(f"Generated {count} images...")

    # Save metadata CSV for easy mapping later
    with open("dataset/metadata.csv", "w") as f:
        f.write("filename,char,font_name\n")
        f.write("\n".join(metadata))

    print(f"Finished. Generated {count} images in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
