#!/usr/bin/env python3
"""
Process 3-channel nd2 files from 260330_4OHT_TH/ with global contrast normalization.
Flat directory structure — files grouped by slide (Slide1-N = one animal/section).
Two-pass: (1) compute global percentile limits, (2) apply to all images.
Outputs to 260330_4OHT_TH/output2/{channel}/
"""
import nd2
import numpy as np
from PIL import Image, ImageFile
from pathlib import Path
import json

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

DATA_DIR = Path(__file__).parent / "260330_4OHT_TH"
OUT_DIR = DATA_DIR / "output2"
PERCENTILE_LO = 0.5
PERCENTILE_HI = 99.5

# Get all 3-channel Region nd2 files
nd2_files = sorted(DATA_DIR.glob("*Region*_Channel470*nm*555*nm*395*nm*.nd2"))
print(f"Found {len(nd2_files)} 3-channel region files\n")

if not nd2_files:
    print("No matching nd2 files found!")
    raise SystemExit(1)

# ── PASS 1: Compute global percentile limits ──
print("=" * 60)
print("PASS 1: Computing global contrast limits")
print("=" * 60)

red_vals, green_vals, blue_vals = [], [], []

for f in nd2_files:
    prefix = f.stem.split('_Channel')[0]
    data = nd2.imread(str(f))
    # ch0=470nm (TH/green), ch1=555nm (tdTomato/red), ch2=395nm (DAPI/blue)
    green_vals.append(data[0][::4, ::4].astype(np.float64).ravel())
    red_vals.append(data[1][::4, ::4].astype(np.float64).ravel())
    blue_vals.append(data[2][::4, ::4].astype(np.float64).ravel())
    print(f"  Sampled {prefix}")

red_all = np.concatenate(red_vals)
green_all = np.concatenate(green_vals)
blue_all = np.concatenate(blue_vals)

limits = {
    'red_lo': float(np.percentile(red_all, PERCENTILE_LO)),
    'red_hi': float(np.percentile(red_all, PERCENTILE_HI)),
    'green_lo': float(np.percentile(green_all, PERCENTILE_LO)),
    'green_hi': float(np.percentile(green_all, PERCENTILE_HI)),
    'blue_lo': float(np.percentile(blue_all, PERCENTILE_LO)),
    'blue_hi': float(np.percentile(blue_all, PERCENTILE_HI)),
    'percentile_lo': PERCENTILE_LO,
    'percentile_hi': PERCENTILE_HI,
    'n_files': len(nd2_files),
}

print(f"\nGlobal limits ({len(nd2_files)} files):")
print(f"  Green (TH):      [{limits['green_lo']:.1f}, {limits['green_hi']:.1f}]")
print(f"  Red  (tdTomato): [{limits['red_lo']:.1f}, {limits['red_hi']:.1f}]")
print(f"  Blue (DAPI):     [{limits['blue_lo']:.1f}, {limits['blue_hi']:.1f}]")

del red_vals, green_vals, blue_vals, red_all, green_all, blue_all

# ── PASS 2: Generate images ──
print(f"\n{'=' * 60}")
print("PASS 2: Generating images with global contrast")
print("=" * 60)

def scale_ch(ch, lo, hi):
    return np.clip((ch - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)

r_lo, r_hi = limits['red_lo'], limits['red_hi']
g_lo, g_hi = limits['green_lo'], limits['green_hi']
b_lo, b_hi = limits['blue_lo'], limits['blue_hi']

OUT_DIR.mkdir(parents=True, exist_ok=True)
with open(OUT_DIR / 'contrast_limits.json', 'w') as f:
    json.dump(limits, f, indent=2)

for nd2_path in nd2_files:
    prefix = nd2_path.stem.split('_Channel')[0]
    data = nd2.imread(str(nd2_path))
    ch_green = data[0].astype(np.float64)
    ch_red = data[1].astype(np.float64)
    ch_blue = data[2].astype(np.float64)
    h, w = ch_red.shape
    zeros = np.zeros((h, w), dtype=np.uint8)

    g8 = scale_ch(ch_green, g_lo, g_hi)
    r8 = scale_ch(ch_red, r_lo, r_hi)
    b8 = scale_ch(ch_blue, b_lo, b_hi)

    outputs = {
        'red_color':   (f'{prefix}_tdTomato.png', np.stack([r8, zeros, zeros], axis=-1)),
        'green_color': (f'{prefix}_TH.png',       np.stack([zeros, g8, zeros], axis=-1)),
        'blue_color':  (f'{prefix}_DAPI.png',     np.stack([zeros, zeros, b8], axis=-1)),
        'merged':      (f'{prefix}_merged.png',   np.stack([r8, g8, b8], axis=-1)),
        'red':         (f'{prefix}_tdTomato.png',  r8),
        'green':       (f'{prefix}_TH.png',        g8),
        'blue':        (f'{prefix}_DAPI.png',       b8),
    }

    for subdir, (fname, img_data) in outputs.items():
        p = OUT_DIR / subdir / fname
        p.parent.mkdir(parents=True, exist_ok=True)
        if img_data.ndim == 2:
            Image.fromarray(img_data, mode='L').save(p)
        else:
            Image.fromarray(img_data).save(p)

    print(f"  ✓ {prefix}")

print(f"\nDone! {len(nd2_files)} files → {OUT_DIR}/")
