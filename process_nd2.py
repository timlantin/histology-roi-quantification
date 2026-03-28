#!/usr/bin/env python3
"""
Process .nd2 microscopy files (555nm tdTomato + 395nm DAPI) into 7 output formats.
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
    'merged_red': 0.114860,         # tdTomato in merged
    'merged_blue': 0.161478,        # DAPI in merged
    'reduced_red': 0.106583,        # Reduced red standalone
    'adj_red': 0.076211,            # Adjusted contrast red
}

def process_nd2_file(nd2_path, output_dir):
    """Process a single .nd2 file into all output formats."""
    nd2_path = Path(nd2_path)
    output_dir = Path(output_dir)
    
    # Parse filename to get region name
    # Format: Slide1-1_Region0000_Channel555 nm,395 nm_Seq0001.nd2
    prefix = nd2_path.stem.split('_Channel')[0]  # Slide1-1_Region0000
    
    print(f"Processing {prefix}...")
    
    # Read nd2 file
    data = nd2.imread(str(nd2_path))
    ch_red = data[0].astype(np.float64)   # 555nm = tdTomato
    ch_blue = data[1].astype(np.float64)  # 395nm = DAPI
    
    h, w = ch_red.shape
    
    # 1. Red color (standalone tdTomato)
    red_8bit = np.clip(ch_red * SCALES['red_color'], 0, 255).astype(np.uint8)
    red_rgb = np.stack([red_8bit, np.zeros((h, w), dtype=np.uint8), np.zeros((h, w), dtype=np.uint8)], axis=-1)
    
    # 2. Blue color (standalone DAPI)  
    blue_8bit = np.clip(ch_blue * SCALES['blue_color'], 0, 255).astype(np.uint8)
    blue_rgb = np.stack([np.zeros((h, w), dtype=np.uint8), np.zeros((h, w), dtype=np.uint8), blue_8bit], axis=-1)
    
    # 3. Merged (different scales)
    merged_red_8bit = np.clip(ch_red * SCALES['merged_red'], 0, 255).astype(np.uint8)
    merged_blue_8bit = np.clip(ch_blue * SCALES['merged_blue'], 0, 255).astype(np.uint8)
    merged_rgb = np.stack([merged_red_8bit, np.zeros((h, w), dtype=np.uint8), merged_blue_8bit], axis=-1)
    
    # 4. Reduced red (standalone)
    reduced_red_8bit = np.clip(ch_red * SCALES['reduced_red'], 0, 255).astype(np.uint8)
    reduced_red_rgb = np.stack([reduced_red_8bit, np.zeros((h, w), dtype=np.uint8), np.zeros((h, w), dtype=np.uint8)], axis=-1)
    
    # 5. Reduced red merged  
    reduced_merged_rgb = np.stack([reduced_red_8bit, np.zeros((h, w), dtype=np.uint8), blue_8bit], axis=-1)
    
    # 6. Adjusted contrast red (standalone)
    adj_red_8bit = np.clip(ch_red * SCALES['adj_red'], 0, 255).astype(np.uint8)
    adj_red_rgb = np.stack([adj_red_8bit, np.zeros((h, w), dtype=np.uint8), np.zeros((h, w), dtype=np.uint8)], axis=-1)
    
    # 7. Adjusted contrast merged
    adj_merged_rgb = np.stack([adj_red_8bit, np.zeros((h, w), dtype=np.uint8), blue_8bit], axis=-1)
    
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
    
    for subdir, filename, img_data in outputs:
        out_path = output_dir / subdir / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(img_data).save(out_path)
    
    print(f"  ✓ Saved to {output_dir}/{{red_color,blue_color,merged,...}}/{prefix}_*")

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

