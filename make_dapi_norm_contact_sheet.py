#!/usr/bin/env python3
"""
Contact sheet showing DAPI-normalized images at threshold 1.0.
For each region: shows pixels classified as yellow (coloc), green-only (TH only),
and dim/background, after per-pixel DAPI normalization.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFile
from pathlib import Path

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

MERGED_DIR = Path(__file__).parent / "260330_4OHT_TH" / "output2" / "merged"
OUT_DIR = Path(__file__).parent / "colocalization_results"

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

NORM_THRESH = 60  # same threshold as analysis
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

# Compute global mean DAPI across all kept regions
dapi_means = []
for a in animals_order:
    for img_path in animal_images.get(a, []):
        img_tmp = np.array(Image.open(img_path).convert('RGB'))
        dapi_means.append(img_tmp[:, :, 2].astype(np.float64).mean())
GLOBAL_DAPI_MEAN = np.mean(dapi_means)
print(f"Global mean DAPI: {GLOBAL_DAPI_MEAN:.1f}")

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
        img = np.array(Image.open(img_path).convert('RGB'))
        red_ch = img[:, :, 0].astype(np.float64)
        green_ch = img[:, :, 1].astype(np.float64)
        blue_ch = img[:, :, 2].astype(np.float64)

        # Region-level DAPI normalization: scale R and G by (global_mean / region_mean)
        region_dapi_mean = blue_ch.mean()
        scale_factor = GLOBAL_DAPI_MEAN / region_dapi_mean if region_dapi_mean > 0 else 1.0

        red_scaled = np.clip(red_ch * scale_factor, 0, 255)
        green_scaled = np.clip(green_ch * scale_factor, 0, 255)
        blue_scaled = np.clip(blue_ch * scale_factor, 0, 255)

        # Full-color DAPI-normalized image
        r8 = np.clip(red_scaled, 0, 255).astype(np.uint8)
        g8 = np.clip(green_scaled, 0, 255).astype(np.uint8)
        b8 = np.clip(blue_scaled, 0, 255).astype(np.uint8)
        color_img = np.stack([r8, g8, b8], axis=-1)

        # Grayscale version
        gray = (0.299 * red_scaled + 0.587 * green_scaled + 0.114 * blue_scaled)
        gray = np.clip(gray, 0, 255).astype(np.uint8)
        gray_img = np.stack([gray, gray, gray], axis=-1)

        # TH+ pixels keep original colors, but cap R <= G (no orange)
        has_green = green_scaled > NORM_THRESH
        r_capped = np.minimum(red_scaled, green_scaled)

        vis = gray_img.copy()
        vis[has_green, 0] = np.clip(r_capped[has_green], 0, 255).astype(np.uint8)
        vis[has_green, 1] = g8[has_green]
        vis[has_green, 2] = 0

        vis_img = Image.fromarray(vis)
        w, h = vis_img.size
        new_h = min(int(h * (THUMB_W / w)), row_h)
        thumb = vis_img.resize((THUMB_W, new_h), Image.LANCZOS)

        canvas.paste(thumb, (x_offset, y_offset))

        region_label = img_path.stem.replace('_merged', '').replace('Slide1-', 'S')
        draw.text((x_offset + 2, y_offset + new_h + 2), region_label, fill=(180, 180, 180))
        x_offset += THUMB_W + PADDING

    y_offset += row_h + LABEL_H + PADDING

out_path = OUT_DIR / "contact_sheet_dapi_normalized.jpg"
canvas.save(out_path, quality=85)
print(f"Saved: {out_path} ({canvas.size[0]}x{canvas.size[1]})")
