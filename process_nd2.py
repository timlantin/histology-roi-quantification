#!/usr/bin/env python3
"""
Process .nd2 microscopy files into output formats.
Supports 2-channel (555nm tdTomato + 395nm DAPI) and
3-channel (470nm TH + 555nm tdTomato + 395nm DAPI) files.
Reverse-engineered from existing outputs on 2026-03-27.
"""
import nd2
import numpy as np
from PIL import Image, ImageFile
from pathlib import Path
import sys

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Fixed scaling factors (determined from existing outputs)
SCALES = {
    'red_color': 0.164411,          # tdTomato standalone
    'blue_color': 0.135063,         # DAPI standalone
    'green_color': 0.164411,        # TH standalone (same as red_color default)
    'merged_red': 0.114860,         # tdTomato in merged
    'merged_blue': 0.161478,        # DAPI in merged
    'merged_green': 0.114860,       # TH in merged (same as merged_red default)
    'reduced_red': 0.106583,        # Reduced red standalone
    'adj_red': 0.076211,            # Adjusted contrast red
}

def detect_channels(nd2_path):
    """Detect whether nd2 file has 2 or 3 channels based on filename and data."""
    name = Path(nd2_path).name
    if '470' in name and '555' in name and '395' in name:
        return 3
    return 2

def process_nd2_file(nd2_path, output_dir):
    """Process a single .nd2 file into all output formats."""
    nd2_path = Path(nd2_path)
    output_dir = Path(output_dir)

    # Parse filename to get region name
    prefix = nd2_path.stem.split('_Channel')[0]
    n_channels = detect_channels(nd2_path)

    print(f"Processing {prefix} ({n_channels}-channel)...")

    # Read nd2 file
    data = nd2.imread(str(nd2_path))

    if n_channels == 3:
        # 3-channel: ch0=470nm (TH/green), ch1=555nm (tdTomato/red), ch2=395nm (DAPI/blue)
        ch_green = data[0].astype(np.float64)
        ch_red = data[1].astype(np.float64)
        ch_blue = data[2].astype(np.float64)
    else:
        # 2-channel: ch0=555nm (tdTomato/red), ch1=395nm (DAPI/blue)
        ch_green = None
        ch_red = data[0].astype(np.float64)
        ch_blue = data[1].astype(np.float64)

    h, w = ch_red.shape
    zeros = np.zeros((h, w), dtype=np.uint8)

    # 1. Red color (standalone tdTomato)
    red_8bit = np.clip(ch_red * SCALES['red_color'], 0, 255).astype(np.uint8)
    red_rgb = np.stack([red_8bit, zeros, zeros], axis=-1)

    # 2. Blue color (standalone DAPI)
    blue_8bit = np.clip(ch_blue * SCALES['blue_color'], 0, 255).astype(np.uint8)
    blue_rgb = np.stack([zeros, zeros, blue_8bit], axis=-1)

    # 3. Merged (different scales)
    merged_red_8bit = np.clip(ch_red * SCALES['merged_red'], 0, 255).astype(np.uint8)
    merged_blue_8bit = np.clip(ch_blue * SCALES['merged_blue'], 0, 255).astype(np.uint8)

    if ch_green is not None:
        merged_green_8bit = np.clip(ch_green * SCALES['merged_green'], 0, 255).astype(np.uint8)
        merged_rgb = np.stack([merged_red_8bit, merged_green_8bit, merged_blue_8bit], axis=-1)
    else:
        merged_rgb = np.stack([merged_red_8bit, zeros, merged_blue_8bit], axis=-1)

    # 4. Reduced red (standalone)
    reduced_red_8bit = np.clip(ch_red * SCALES['reduced_red'], 0, 255).astype(np.uint8)
    reduced_red_rgb = np.stack([reduced_red_8bit, zeros, zeros], axis=-1)

    # 5. Reduced red merged
    if ch_green is not None:
        reduced_merged_rgb = np.stack([reduced_red_8bit, merged_green_8bit, blue_8bit], axis=-1)
    else:
        reduced_merged_rgb = np.stack([reduced_red_8bit, zeros, blue_8bit], axis=-1)

    # 6. Adjusted contrast red (standalone)
    adj_red_8bit = np.clip(ch_red * SCALES['adj_red'], 0, 255).astype(np.uint8)
    adj_red_rgb = np.stack([adj_red_8bit, zeros, zeros], axis=-1)

    # 7. Adjusted contrast merged
    if ch_green is not None:
        adj_merged_rgb = np.stack([adj_red_8bit, merged_green_8bit, blue_8bit], axis=-1)
    else:
        adj_merged_rgb = np.stack([adj_red_8bit, zeros, blue_8bit], axis=-1)

    # Save all outputs
    outputs = [
        ('red_color', f'{prefix}_tdTomato.png', red_rgb),
        ('blue_color', f'{prefix}_DAPI.png', blue_rgb),
        ('merged', f'{prefix}_merged.png', merged_rgb),
        ('reduced_red', f'{prefix}_tdTomato.png', reduced_red_rgb),
        ('reduced_red_merged', f'{prefix}_merged.png', reduced_merged_rgb),
        ('adjusted_contrast_red', f'{prefix}_tdTomato.png', adj_red_rgb),
        ('adjusted_contrast_merged', f'{prefix}_merged.png', adj_merged_rgb),
    ]

    # Add green channel outputs for 3-channel files
    if ch_green is not None:
        green_8bit = np.clip(ch_green * SCALES['green_color'], 0, 255).astype(np.uint8)
        green_rgb = np.stack([zeros, green_8bit, zeros], axis=-1)
        outputs.append(('green_color', f'{prefix}_TH.png', green_rgb))
        outputs.append(('green', f'{prefix}_TH.png', green_8bit))  # grayscale

    for subdir, filename, img_data in outputs:
        out_path = output_dir / subdir / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if img_data.ndim == 2:
            Image.fromarray(img_data, mode='L').save(out_path)
        else:
            Image.fromarray(img_data).save(out_path)

    ch_str = "red_color,blue_color,green_color,merged,..." if ch_green is not None else "red_color,blue_color,merged,..."
    print(f"  ✓ Saved to {output_dir}/{{{ch_str}}}/{prefix}_*")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python process_nd2.py <nd2_file1> [<nd2_file2> ...]")
        print("   or: python process_nd2.py Slide1-1_Region*")
        sys.exit(1)
    
    output_dir = Path(__file__).parent / 'output'
    
    for nd2_file in sys.argv[1:]:
        try:
            process_nd2_file(nd2_file, output_dir)
        except Exception as e:
            print(f"ERROR processing {nd2_file}: {e}")
            import traceback
            traceback.print_exc()

