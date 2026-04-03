#!/usr/bin/env python3
"""
Process all .nd2 files across animals with globally consistent contrast limits.
Two-pass: (1) compute global percentile limits, (2) apply to all images.
Supports 2-channel (555nm+395nm) and 3-channel (470nm+555nm+395nm) files.
Outputs to output2/ in each animal folder.
"""
import nd2
import numpy as np
from PIL import Image, ImageFile
from pathlib import Path
import json, sys

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

BASE = Path(__file__).parent
ANIMALS = sorted(BASE.glob("260324_OHT-*"))
PERCENTILE_LO = 0.5
PERCENTILE_HI = 99.5

def get_region_nd2s(animal_dir):
    """Return only Region .nd2 files (skip whole-slide DAPI-only scans)."""
    two_ch = sorted(p for p in animal_dir.glob("*Region*_Channel*555*nm*395*nm*.nd2")
                    if '470' not in p.name)
    three_ch = sorted(p for p in animal_dir.glob("*Region*_Channel*470*nm*555*nm*395*nm*.nd2"))
    return two_ch + three_ch

def detect_channels(nd2_path):
    """Detect whether nd2 file has 2 or 3 channels based on filename."""
    if '470' in nd2_path.name and '555' in nd2_path.name and '395' in nd2_path.name:
        return 3
    return 2

# ── PASS 1: Collect percentile stats globally ──
print("=" * 60)
print("PASS 1: Computing global contrast limits")
print("=" * 60)

red_vals = []
blue_vals = []
green_vals = []
file_count = 0
has_green = False

for animal in ANIMALS:
    nd2_files = get_region_nd2s(animal)
    print(f"{animal.name}: {len(nd2_files)} region files")
    for f in nd2_files:
        data = nd2.imread(str(f))
        n_ch = detect_channels(f)
        if n_ch == 3:
            ch_green = data[0].astype(np.float64)
            ch_red = data[1].astype(np.float64)
            ch_blue = data[2].astype(np.float64)
            green_vals.append(ch_green[::4, ::4].ravel())
            has_green = True
        else:
            ch_red = data[0].astype(np.float64)
            ch_blue = data[1].astype(np.float64)
        # Sample to keep memory manageable (every 4th pixel)
        red_vals.append(ch_red[::4, ::4].ravel())
        blue_vals.append(ch_blue[::4, ::4].ravel())
        file_count += 1

red_all = np.concatenate(red_vals)
blue_all = np.concatenate(blue_vals)

limits = {
    'red_lo': float(np.percentile(red_all, PERCENTILE_LO)),
    'red_hi': float(np.percentile(red_all, PERCENTILE_HI)),
    'blue_lo': float(np.percentile(blue_all, PERCENTILE_LO)),
    'blue_hi': float(np.percentile(blue_all, PERCENTILE_HI)),
    'percentile_lo': PERCENTILE_LO,
    'percentile_hi': PERCENTILE_HI,
    'n_files': file_count,
    'n_animals': len(ANIMALS),
    'red_global_min': float(red_all.min()),
    'red_global_max': float(red_all.max()),
    'blue_global_min': float(blue_all.min()),
    'blue_global_max': float(blue_all.max()),
}

if has_green and green_vals:
    green_all = np.concatenate(green_vals)
    limits['green_lo'] = float(np.percentile(green_all, PERCENTILE_LO))
    limits['green_hi'] = float(np.percentile(green_all, PERCENTILE_HI))
    limits['green_global_min'] = float(green_all.min())
    limits['green_global_max'] = float(green_all.max())
    del green_all

print(f"\nGlobal limits ({file_count} files, {len(ANIMALS)} animals):")
print(f"  Red  (tdTomato): [{limits['red_lo']:.1f}, {limits['red_hi']:.1f}]  (range: {limits['red_global_min']:.0f}–{limits['red_global_max']:.0f})")
print(f"  Blue (DAPI):     [{limits['blue_lo']:.1f}, {limits['blue_hi']:.1f}]  (range: {limits['blue_global_min']:.0f}–{limits['blue_global_max']:.0f})")
if has_green:
    print(f"  Green (TH):      [{limits['green_lo']:.1f}, {limits['green_hi']:.1f}]  (range: {limits['green_global_min']:.0f}–{limits['green_global_max']:.0f})")

# Save global limits
limits_path = BASE / 'global_contrast_limits.json'
with open(limits_path, 'w') as f:
    json.dump(limits, f, indent=2)
print(f"  Saved to {limits_path}")

# Free memory
del red_vals, blue_vals, green_vals, red_all, blue_all

# ── PASS 2: Apply limits and save images ──
print(f"\n{'=' * 60}")
print("PASS 2: Generating images with global contrast")
print("=" * 60)

def scale_channel(ch, lo, hi):
    """Scale 16-bit channel to 8-bit using fixed lo/hi limits."""
    return np.clip((ch - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)

r_lo, r_hi = limits['red_lo'], limits['red_hi']
b_lo, b_hi = limits['blue_lo'], limits['blue_hi']
g_lo = limits.get('green_lo', 0)
g_hi = limits.get('green_hi', 1)

for animal in ANIMALS:
    nd2_files = get_region_nd2s(animal)
    out_dir = animal / 'output2'
    print(f"\n{animal.name}: {len(nd2_files)} files → {out_dir}")

    # Also save limits per animal folder
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'contrast_limits.json', 'w') as f:
        json.dump(limits, f, indent=2)

    for nd2_path in nd2_files:
        prefix = nd2_path.stem.split('_Channel')[0]
        data = nd2.imread(str(nd2_path))
        n_ch = detect_channels(nd2_path)

        if n_ch == 3:
            ch_green = data[0].astype(np.float64)
            ch_red = data[1].astype(np.float64)
            ch_blue = data[2].astype(np.float64)
        else:
            ch_green = None
            ch_red = data[0].astype(np.float64)
            ch_blue = data[1].astype(np.float64)

        h, w = ch_red.shape

        red_8 = scale_channel(ch_red, r_lo, r_hi)
        blue_8 = scale_channel(ch_blue, b_lo, b_hi)
        zeros = np.zeros((h, w), dtype=np.uint8)

        if ch_green is not None:
            green_8 = scale_channel(ch_green, g_lo, g_hi)
            merged_img = np.stack([red_8, green_8, blue_8], axis=-1)
        else:
            merged_img = np.stack([red_8, zeros, blue_8], axis=-1)

        outputs = {
            'red_color': (f'{prefix}_tdTomato.png', np.stack([red_8, zeros, zeros], axis=-1)),
            'blue_color': (f'{prefix}_DAPI.png', np.stack([zeros, zeros, blue_8], axis=-1)),
            'merged': (f'{prefix}_merged.png', merged_img),
            'red': (f'{prefix}_tdTomato.png', red_8),          # grayscale
            'blue': (f'{prefix}_DAPI.png', blue_8),             # grayscale
        }

        if ch_green is not None:
            outputs['green_color'] = (f'{prefix}_TH.png', np.stack([zeros, green_8, zeros], axis=-1))
            outputs['green'] = (f'{prefix}_TH.png', green_8)    # grayscale

        for subdir, (fname, img_data) in outputs.items():
            p = out_dir / subdir / fname
            p.parent.mkdir(parents=True, exist_ok=True)
            if img_data.ndim == 2:
                Image.fromarray(img_data, mode='L').save(p)
            else:
                Image.fromarray(img_data).save(p)

        print(f"  ✓ {prefix} ({n_ch}-ch)")

print(f"\nDone! All images saved to output2/ with consistent contrast.")
