#!/usr/bin/env python3
"""Contact sheet of only the kept regions, grouped by animal."""
import numpy as np
from PIL import Image, ImageDraw, ImageFile
from pathlib import Path

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

MERGED_DIR = Path(__file__).parent / "260330_4OHT_TH" / "output2" / "merged"
OUT_PATH = Path(__file__).parent / "colocalization_results" / "contact_sheet_filtered.jpg"

SLIDE_TO_ANIMAL = {
    'Slide1-1': 'OHT-1', 'Slide1-2': 'OHT-1',
    'Slide1-3': 'OHT-3', 'Slide1-4': 'OHT-3', 'Slide1-5': 'OHT-3',
    'Slide1-6': 'OHT-2', 'Slide1-7': 'OHT-2',
    'Slide1-8': 'OHT-4', 'Slide1-9': 'OHT-4',
}
SEX = {'OHT-1': 'F', 'OHT-2': 'F', 'OHT-3': 'M', 'OHT-4': 'M'}

EXCLUDE = {
    'Slide1-1_Region0001','Slide1-1_Region0002','Slide1-1_Region0003',
    'Slide1-1_Region0004','Slide1-1_Region0005','Slide1-1_Region0006',
    'Slide1-1_Region0000','Slide1-2_Region0000','Slide1-3_Region0000',
    'Slide1-4_Region0000','Slide1-5_Region0000','Slide1-6_Region0000',
    'Slide1-7_Region0000','Slide1-8_Region0000','Slide1-9_Region0000',
    'Slide1-3_Region0001','Slide1-3_Region0002','Slide1-3_Region0003',
    'Slide1-3_Region0004','Slide1-3_Region0005',
    'Slide1-6_Region0001','Slide1-6_Region0002','Slide1-6_Region0003',
    'Slide1-6_Region0004',
    'Slide1-8_Region0001','Slide1-8_Region0002','Slide1-8_Region0003',
    'Slide1-8_Region0004','Slide1-8_Region0005',
    'Slide1-4_Region0006','Slide1-9_Region0001','Slide1-9_Region0006',
    'Slide1-7_Region0004', 'Slide1-4_Region0005',
    'Slide1-9_Region0007',
}

THUMB_W = 300
PADDING = 4
LABEL_H = 20
BG_COLOR = (30, 30, 30)

# Group kept images by animal
animal_images = {}
for img_path in sorted(MERGED_DIR.glob("*_merged.png")):
    slide = img_path.stem.split('_Region')[0]
    region = img_path.stem.split('_merged')[0]
    animal = SLIDE_TO_ANIMAL.get(slide)
    if animal is None or region in EXCLUDE:
        continue
    animal_images.setdefault(animal, []).append(img_path)

animals_order = ['OHT-1', 'OHT-2', 'OHT-3', 'OHT-4']
max_per_row = max(len(animal_images.get(a, [])) for a in animals_order)

# Compute row heights
thumb_heights = []
for a in animals_order:
    max_h = 0
    for img_path in animal_images.get(a, []):
        img = Image.open(img_path)
        w, h = img.size
        th = int(h * (THUMB_W / w))
        max_h = max(max_h, th)
        img.close()
    thumb_heights.append(min(max_h, 400))

total_w = max_per_row * (THUMB_W + PADDING) + PADDING + 120
total_h = sum(th + LABEL_H + PADDING for th in thumb_heights) + PADDING
canvas = Image.new('RGB', (total_w, total_h), BG_COLOR)
draw = ImageDraw.Draw(canvas)

y_offset = PADDING
for row, animal in enumerate(animals_order):
    row_h = thumb_heights[row]
    label = f"{animal} ({SEX[animal]})"
    draw.text((8, y_offset + row_h // 2), label, fill=(220, 220, 220))

    x_offset = 120
    for img_path in animal_images.get(animal, []):
        img = Image.open(img_path).convert('RGB')
        w, h = img.size
        new_h = min(int(h * (THUMB_W / w)), row_h)
        thumb = img.resize((THUMB_W, new_h), Image.LANCZOS)
        img.close()
        canvas.paste(thumb, (x_offset, y_offset))

        region_label = img_path.stem.replace('_merged', '').replace('Slide1-', 'S')
        draw.text((x_offset + 2, y_offset + new_h + 2), region_label, fill=(180, 180, 180))
        x_offset += THUMB_W + PADDING

    y_offset += row_h + LABEL_H + PADDING

canvas.save(OUT_PATH, quality=80)
print(f"Saved: {OUT_PATH} ({canvas.size[0]}x{canvas.size[1]})")
