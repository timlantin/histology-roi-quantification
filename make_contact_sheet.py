#!/usr/bin/env python3
"""
Generate a low-res contact sheet of all 57 merged ROI images, grouped by animal.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFile
from pathlib import Path

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

MERGED_DIR = Path(__file__).parent / "260330_4OHT_TH" / "output2" / "merged"
OUT_PATH = Path(__file__).parent / "colocalization_results" / "contact_sheet_all_regions.jpg"

SLIDE_TO_ANIMAL = {
    'Slide1-1': 'OHT-1', 'Slide1-2': 'OHT-1',
    'Slide1-3': 'OHT-3', 'Slide1-4': 'OHT-3', 'Slide1-5': 'OHT-3',
    'Slide1-6': 'OHT-2', 'Slide1-7': 'OHT-2',
    'Slide1-8': 'OHT-4', 'Slide1-9': 'OHT-4',
}
SEX = {'OHT-1': 'F', 'OHT-2': 'F', 'OHT-3': 'M', 'OHT-4': 'M'}

THUMB_W = 300  # thumbnail width
PADDING = 4
LABEL_H = 20
BG_COLOR = (30, 30, 30)
LABEL_COLOR = (220, 220, 220)

# Group images by animal
animal_images = {}
for img_path in sorted(MERGED_DIR.glob("*_merged.png")):
    slide = img_path.stem.split('_Region')[0]
    animal = SLIDE_TO_ANIMAL.get(slide)
    if animal is None:
        continue
    animal_images.setdefault(animal, []).append(img_path)

# Calculate layout
animals_order = ['OHT-1', 'OHT-2', 'OHT-3', 'OHT-4']
max_per_animal = max(len(animal_images[a]) for a in animals_order)
n_cols = max_per_animal
n_rows = len(animals_order)

# First pass: compute thumbnail heights (preserve aspect ratio)
thumb_heights = []
for a in animals_order:
    max_h = 0
    for img_path in animal_images[a]:
        img = Image.open(img_path)
        w, h = img.size
        scale = THUMB_W / w
        th = int(h * scale)
        max_h = max(max_h, th)
        img.close()
    thumb_heights.append(min(max_h, 400))  # cap height

# Canvas size
total_w = n_cols * (THUMB_W + PADDING) + PADDING + 120  # 120 for animal labels
total_h = sum(th + LABEL_H + PADDING for th in thumb_heights) + PADDING
canvas = Image.new('RGB', (total_w, total_h), BG_COLOR)
draw = ImageDraw.Draw(canvas)

y_offset = PADDING
for row, animal in enumerate(animals_order):
    row_h = thumb_heights[row]

    # Animal label on the left
    label = f"{animal} ({SEX[animal]})"
    draw.text((8, y_offset + row_h // 2), label, fill=LABEL_COLOR)

    x_offset = 120
    for img_path in animal_images[animal]:
        img = Image.open(img_path).convert('RGB')
        w, h = img.size
        scale = THUMB_W / w
        new_h = min(int(h * scale), row_h)
        thumb = img.resize((THUMB_W, new_h), Image.LANCZOS)
        img.close()

        canvas.paste(thumb, (x_offset, y_offset))

        # Region label below
        region_label = img_path.stem.replace('_merged', '').replace('Slide1-', 'S')
        draw.text((x_offset + 2, y_offset + new_h + 2), region_label,
                  fill=(180, 180, 180))

        x_offset += THUMB_W + PADDING

    y_offset += row_h + LABEL_H + PADDING

canvas.save(OUT_PATH, quality=80)
print(f"Contact sheet saved: {OUT_PATH} ({canvas.size[0]}x{canvas.size[1]})")
