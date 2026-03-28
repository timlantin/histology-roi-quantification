#!/usr/bin/env python3
"""
ROI Quantification App — compare fluorescence expression across mice.
Multiple ROIs per animal, multiple images per animal.
Usage: streamlit run roi_quantify.py
"""
import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFile
from pathlib import Path
import pandas as pd

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

st.set_page_config(layout="wide", page_title="ROI Quantification")
st.title("🔬 ROI Expression Quantification")

BASE = Path(__file__).parent / "260324_4OHT_TRAP2_Ai14_pilot"
ANIMALS = sorted([d.name for d in BASE.iterdir() if d.is_dir() and "OHT" in d.name])
CHANNEL_LABELS = {
    "red_color": "tdTomato (color)",
    "blue_color": "DAPI (color)",
    "merged": "Merged",
    "red": "tdTomato (grayscale)",
    "blue": "DAPI (grayscale)",
}
ROI_COLORS = ["#00FF00", "#FF00FF", "#00FFFF", "#FFFF00", "#FF8000", "#8000FF"]

st.sidebar.header("Settings")
output_folder = st.sidebar.selectbox("Output folder", ["output2", "output"], index=0)
channel = st.sidebar.selectbox("Display channel", list(CHANNEL_LABELS.keys()),
                                format_func=lambda x: CHANNEL_LABELS[x])
quantify_on = st.sidebar.selectbox("Quantify on", ["red (tdTomato)", "blue (DAPI)", "displayed image"])
max_rois = st.sidebar.number_input("Max ROIs per image", 1, 6, 3)

def get_images(animal, channel, output_folder):
    img_dir = BASE / animal / output_folder / channel
    if not img_dir.exists():
        return []
    return sorted([f.name for f in img_dir.glob("*.png")])

def quantify_roi(roi_array, quant_channel):
    if roi_array.size == 0:
        return None
    if quant_channel == "red (tdTomato)":
        vals = roi_array[:, :, 0].astype(float) if roi_array.ndim == 3 else roi_array.astype(float)
        ch = "tdTomato"
    elif quant_channel == "blue (DAPI)":
        vals = roi_array[:, :, 2].astype(float) if roi_array.ndim == 3 else roi_array.astype(float)
        ch = "DAPI"
    else:
        vals = np.mean(roi_array.astype(float), axis=2) if roi_array.ndim == 3 else roi_array.astype(float)
        ch = "displayed"
    return {
        'channel': ch,
        'mean': vals.mean(),
        'median': np.median(vals),
        'std': vals.std(),
        'min': vals.min(),
        'max': vals.max(),
        'area_px': vals.size,
        'integrated': vals.sum(),
        'pct_positive': (vals > 0).mean() * 100,
    }

def draw_rois(img, roi_list):
    overlay = img.copy()
    if overlay.mode != "RGB":
        overlay = overlay.convert("RGB")
    draw = ImageDraw.Draw(overlay)
    lw = max(2, min(img.width, img.height) // 150)
    for i, (x1, y1, x2, y2) in enumerate(roi_list):
        color = ROI_COLORS[i % len(ROI_COLORS)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=lw)
        # Label
        draw.text((x1 + 5, y1 + 5), f"ROI {i+1}", fill=color)
    return overlay

# ── Per-animal tabs ──
results = []

tabs = st.tabs([a.split('_')[-1] for a in ANIMALS])

for tab, animal in zip(tabs, ANIMALS):
    with tab:
        label = animal.split('_')[-1]
        images = get_images(animal, channel, output_folder)
        if not images:
            st.warning("No images")
            continue
        
        # Multi-select images for this animal
        selected_imgs = st.multiselect(
            f"Select images for {label}",
            images,
            default=[images[0]] if images else [],
            key=f"imgs_{animal}"
        )
        
        for img_name in selected_imgs:
            img_path = BASE / animal / output_folder / channel / img_name
            img_full = Image.open(img_path).convert("RGB")
            orig_w, orig_h = img_full.size
            
            st.markdown(f"#### `{img_name}` ({orig_w}×{orig_h})")
            
            # ROI count for this image
            n_rois = st.number_input(
                f"Number of ROIs", 1, max_rois, 1,
                key=f"nroi_{animal}_{img_name}"
            )
            
            roi_coords = []
            slider_cols = st.columns(n_rois)
            
            for r in range(n_rois):
                with slider_cols[r]:
                    color = ROI_COLORS[r % len(ROI_COLORS)]
                    st.markdown(f"**ROI {r+1}** <span style='color:{color}'>■</span>",
                                unsafe_allow_html=True)
                    cx = st.slider("X%", 0, 100, 30 + r * 20, key=f"cx_{animal}_{img_name}_{r}")
                    cy = st.slider("Y%", 0, 100, 50, key=f"cy_{animal}_{img_name}_{r}")
                    rw = st.slider("W%", 1, 50, 10, key=f"rw_{animal}_{img_name}_{r}")
                    rh = st.slider("H%", 1, 50, 10, key=f"rh_{animal}_{img_name}_{r}")
                    
                    px_cx = int(cx / 100 * orig_w)
                    px_cy = int(cy / 100 * orig_h)
                    half_w = int(rw / 200 * orig_w)
                    half_h = int(rh / 200 * orig_h)
                    x1 = max(0, px_cx - half_w)
                    y1 = max(0, px_cy - half_h)
                    x2 = min(orig_w, px_cx + half_w)
                    y2 = min(orig_h, px_cy + half_h)
                    roi_coords.append((x1, y1, x2, y2))
            
            # Display image with all ROIs
            overlay = draw_rois(img_full, roi_coords)
            st.image(overlay, use_container_width=True)
            
            # Quantify each ROI
            img_array = np.array(img_full)
            roi_results_cols = st.columns(n_rois)
            for r, (x1, y1, x2, y2) in enumerate(roi_coords):
                roi_array = img_array[y1:y2, x1:x2]
                stats = quantify_roi(roi_array, quantify_on)
                if stats:
                    with roi_results_cols[r]:
                        color = ROI_COLORS[r % len(ROI_COLORS)]
                        st.markdown(f"**ROI {r+1}** — Mean: **{stats['mean']:.1f}**, "
                                    f"Med: {stats['median']:.1f}, Std: {stats['std']:.1f}")
                    
                    results.append({
                        'Animal': label,
                        'Image': img_name,
                        'ROI #': r + 1,
                        'ROI coords': f"({x1},{y1})→({x2},{y2})",
                        'Channel': stats['channel'],
                        'Mean': round(stats['mean'], 2),
                        'Median': round(stats['median'], 2),
                        'Std': round(stats['std'], 2),
                        'Min': int(stats['min']),
                        'Max': int(stats['max']),
                        'Area (px)': stats['area_px'],
                        'Integrated Density': int(stats['integrated']),
                        '% Positive': f"{stats['pct_positive']:.1f}%",
                    })
            
            st.divider()

# ── Comparison ──
st.header("📊 Comparison")
if results:
    df = pd.DataFrame(results)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Summary stats per animal
    st.subheader("Per-animal summary")
    summary = df.groupby('Animal')['Mean'].agg(['mean', 'std', 'count', 'min', 'max']).round(2)
    summary.columns = ['Mean ± ', 'Std', 'n ROIs', 'Min', 'Max']
    st.dataframe(summary, use_container_width=True)
    
    # Bar chart with error bars via summary
    chart_data = df.groupby('Animal')['Mean'].agg(['mean', 'std']).reset_index()
    chart_data.columns = ['Animal', 'Mean Intensity', 'Std']
    st.bar_chart(chart_data.set_index('Animal')['Mean Intensity'])
    
    # Stats if enough data
    animal_groups = [g['Mean'].values for _, g in df.groupby('Animal')]
    if len(animal_groups) >= 2 and all(len(g) >= 2 for g in animal_groups):
        from scipy.stats import f_oneway, kruskal
        
        st.subheader("Statistical tests")
        f_stat, p_anova = f_oneway(*animal_groups)
        h_stat, p_kruskal = kruskal(*animal_groups)
        
        st.write(f"**One-way ANOVA:** F = {f_stat:.3f}, p = {p_anova:.4f} "
                 f"{'✅ significant' if p_anova < 0.05 else '❌ not significant'}")
        st.write(f"**Kruskal-Wallis:** H = {h_stat:.3f}, p = {p_kruskal:.4f} "
                 f"{'✅ significant' if p_kruskal < 0.05 else '❌ not significant'}")
        
        if p_anova < 0.05:
            from scipy.stats import ttest_ind
            st.write("**Post-hoc pairwise t-tests (uncorrected):**")
            animals_unique = df['Animal'].unique()
            pairs = []
            for i in range(len(animals_unique)):
                for j in range(i+1, len(animals_unique)):
                    a1 = df[df['Animal'] == animals_unique[i]]['Mean'].values
                    a2 = df[df['Animal'] == animals_unique[j]]['Mean'].values
                    t, p = ttest_ind(a1, a2)
                    pairs.append({
                        'Comparison': f"{animals_unique[i]} vs {animals_unique[j]}",
                        't': round(t, 3),
                        'p': round(p, 4),
                        'Significant': '✅' if p < 0.05 else '❌',
                    })
            st.dataframe(pd.DataFrame(pairs), hide_index=True)
    else:
        st.info("Add ≥2 ROIs per animal to enable ANOVA / Kruskal-Wallis tests.")
    
    csv = df.to_csv(index=False)
    st.download_button("📥 Download CSV", csv, "roi_quantification.csv", "text/csv")

