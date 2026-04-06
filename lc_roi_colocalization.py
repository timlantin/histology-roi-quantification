#!/usr/bin/env python3
"""
LC-restricted colocalization analysis.
Uses manually selected TH+ clusters as LC ROIs per region.
Region-level DAPI normalization, threshold=60.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
from pathlib import Path
from scipy.stats import f_oneway, ttest_ind, mannwhitneyu
from scipy.ndimage import label, binary_closing, binary_dilation, generate_binary_structure
import warnings
warnings.filterwarnings('ignore')

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

# Manual LC cluster selections (cluster numbers from roi_candidates)
LC_CLUSTERS = {
    'Slide1-2_Region0001': [1, 3],
    'Slide1-2_Region0002': [1, 2],
    'Slide1-2_Region0003': [1, 2],
    'Slide1-2_Region0004': [2, 5],
    'Slide1-2_Region0005': [2, 7],
    'Slide1-4_Region0001': [1, 2],
    'Slide1-4_Region0002': [1, 8],
    'Slide1-4_Region0003': [4, 7],
    'Slide1-4_Region0004': [2, 3],
    'Slide1-7_Region0001': [1, 2],
    'Slide1-7_Region0002': [1, 3],
    'Slide1-9_Region0002': [1, 2],
    'Slide1-9_Region0003': [1, 2],
    'Slide1-9_Region0004': [1, 2],
    'Slide1-9_Region0005': [2],
}
# Excluded: S5 (no LC detected), S7-R0003, S7-R0005 (none)

GLOBAL_DAPI = 37.18
THRESH = 60
DS = 8

plt.rcParams.update({
    'figure.facecolor': '#FAFAFA',
    'axes.facecolor': '#FAFAFA',
    'font.family': 'Helvetica Neue',
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
})
COLORS = {'OHT-1': '#E07B54', 'OHT-2': '#E8A07A', 'OHT-3': '#5B8BD4', 'OHT-4': '#7BAFD4'}
SEX_COLORS = {'F': '#E07B54', 'M': '#5B8BD4'}

struct = generate_binary_structure(2, 2)

print("LC-Restricted Colocalization Analysis")
print(f"Threshold: {THRESH}, DAPI-normalized, {len(LC_CLUSTERS)} regions\n")

rows = []
for region, cluster_ids in sorted(LC_CLUSTERS.items()):
    slide = region.split('_Region')[0]
    animal = SLIDE_TO_ANIMAL[slide]
    img_path = MERGED_DIR / f"{region}_merged.png"

    img = np.array(Image.open(img_path).convert("RGB"))
    red = img[:, :, 0].astype(np.float64)
    green = img[:, :, 1].astype(np.float64)
    blue = img[:, :, 2].astype(np.float64)

    scale = GLOBAL_DAPI / blue.mean()
    green_s = green * scale
    red_s = red * scale

    # Detect clusters (same method as roi_candidates)
    green_ds = green_s[::DS, ::DS]
    th_mask = green_ds > 100
    th_closed = binary_closing(th_mask, struct, iterations=5)
    th_dilated = binary_dilation(th_closed, struct, iterations=5)
    labeled, n_comp = label(th_dilated)

    # Rank clusters by mean intensity
    clusters = []
    for i in range(1, n_comp + 1):
        mask = labeled == i
        area = mask.sum() * DS * DS
        if area < 5000:
            continue
        mean_int = green_ds[mask].mean()
        clusters.append((i, mean_int, mask))
    clusters.sort(key=lambda c: -c[1])

    # Build LC mask from selected clusters (upsampled)
    lc_mask = np.zeros(img.shape[:2], dtype=bool)
    for k_idx in cluster_ids:
        if k_idx - 1 < len(clusters):
            _, _, mask_ds = clusters[k_idx - 1]
            # Get bounding box with margin and apply as ROI
            ys, xs = np.where(mask_ds)
            margin_y = int((ys.max() - ys.min()) * 0.3)
            margin_x = int((xs.max() - xs.min()) * 0.3)
            y1 = max(0, ys.min() * DS - margin_y * DS)
            y2 = min(img.shape[0], ys.max() * DS + margin_y * DS)
            x1 = max(0, xs.min() * DS - margin_x * DS)
            x2 = min(img.shape[1], xs.max() * DS + margin_x * DS)
            lc_mask[y1:y2, x1:x2] = True

    # Colocalization within LC ROI only
    th_pos = (green_s > THRESH) & lc_mask
    tdtom_pos = (red_s > THRESH) & lc_mask
    yellow = th_pos & tdtom_pos
    green_only = th_pos & ~tdtom_pos

    n_yellow = int(yellow.sum())
    n_green_only = int(green_only.sum())
    n_th_total = n_yellow + n_green_only
    ratio = n_yellow / n_th_total if n_th_total > 0 else 0.0
    roi_pct = lc_mask.sum() / lc_mask.size * 100

    rows.append({
        'Animal': animal, 'Sex': SEX[animal], 'Region': region,
        'Clusters': ','.join(map(str, cluster_ids)),
        'ROI %': round(roi_pct, 1),
        'TH+ pixels': n_th_total, 'Yellow (coloc)': n_yellow,
        'Green-only': n_green_only, 'Coloc %': round(ratio * 100, 1),
    })

    short = region.replace('Slide1-', 'S').replace('_Region', '-R')
    print(f"  {short:15s}  {animal}  clusters={cluster_ids}  ROI={roi_pct:.1f}%  "
          f"TH+={n_th_total:>8,}  coloc={ratio*100:.1f}%")

df = pd.DataFrame(rows)
df.to_csv(OUT_DIR / "lc_roi_per_region.csv", index=False)
print(f"\n{len(df)} regions, {df['Animal'].nunique()} animals")

# Per-animal
print("\n── Per-animal ──")
animals_list = ['OHT-1', 'OHT-2', 'OHT-3', 'OHT-4']
for a in animals_list:
    sub = df[df['Animal'] == a]
    if len(sub) == 0:
        print(f"  {a} ({SEX[a]}): no regions")
        continue
    print(f"  {a} ({SEX[a]}): {sub['Coloc %'].mean():.1f}% +/- {sub['Coloc %'].std():.1f}  n={len(sub)}")

# Per-sex
print("\n── Per-sex ──")
for s in ['F', 'M']:
    sub = df[df['Sex'] == s]
    print(f"  {s}: {sub['Coloc %'].mean():.1f}% +/- {sub['Coloc %'].std():.1f}  n={len(sub)}")

# Stats
print("\n── Statistics ──")
groups = [g['Coloc %'].values for _, g in df.groupby('Animal') if len(g) > 0]
animals_with_data = [a for a in animals_list if len(df[df['Animal'] == a]) > 0]

if len(groups) >= 2 and all(len(g) >= 2 for g in groups):
    f_stat, p_anova = f_oneway(*groups)
    print(f"ANOVA (animals): F={f_stat:.3f}, p={p_anova:.4f} {'*' if p_anova < 0.05 else 'ns'}")

    pairwise_p = {}
    for i in range(len(animals_with_data)):
        for j in range(i + 1, len(animals_with_data)):
            a1 = df[df['Animal'] == animals_with_data[i]]['Coloc %'].values
            a2 = df[df['Animal'] == animals_with_data[j]]['Coloc %'].values
            if len(a1) >= 2 and len(a2) >= 2:
                t, p = ttest_ind(a1, a2)
                pairwise_p[(i, j)] = p
                print(f"  {animals_with_data[i]} vs {animals_with_data[j]}: t={t:.3f}, p={p:.4f} {'*' if p < 0.05 else 'ns'}")

f_vals = df[df['Sex'] == 'F']['Coloc %'].values
m_vals = df[df['Sex'] == 'M']['Coloc %'].values
if len(f_vals) >= 2 and len(m_vals) >= 2:
    t_sex, p_sex = ttest_ind(f_vals, m_vals)
    u_sex, p_mw = mannwhitneyu(f_vals, m_vals, alternative='two-sided')
    print(f"\nSex: F={f_vals.mean():.1f}% vs M={m_vals.mean():.1f}%")
    print(f"  t-test: p={p_sex:.4f} {'*' if p_sex < 0.05 else 'ns'}")
    print(f"  Mann-Whitney: p={p_mw:.4f} {'*' if p_mw < 0.05 else 'ns'}")

# Plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

means = [df[df['Animal'] == a]['Coloc %'].mean() for a in animals_with_data]
sems = [df[df['Animal'] == a]['Coloc %'].sem() for a in animals_with_data]

ax1.bar(animals_with_data, means, yerr=sems, capsize=4,
        color=[COLORS[a] for a in animals_with_data],
        edgecolor='white', linewidth=1.2, alpha=0.85, width=0.6, error_kw={'linewidth': 1.2})
for i, a in enumerate(animals_with_data):
    vals = df[df['Animal'] == a]['Coloc %'].values
    jitter = np.random.default_rng(42).uniform(-0.12, 0.12, size=len(vals))
    ax1.scatter([i] * len(vals) + jitter, vals, color='#333333', s=18, alpha=0.6, zorder=5)
ax1.set_ylabel('TdTom+ & TH+ Colocalization %')
ax1.set_title('LC-Restricted Colocalization by Animal')

# Significance bars
if pairwise_p:
    bar_height = max(means) + max(sems) * 2 + 3
    bar_step = 5
    for (i, j), p in sorted(pairwise_p.items(), key=lambda x: abs(x[0][1] - x[0][0])):
        label_txt = f"p={p:.3f}" if p >= 0.001 else "p<0.001"
        if p >= 0.05:
            label_txt = "ns"
        y = bar_height
        bar_height += bar_step
        ax1.plot([i, i, j, j], [y - 1, y, y, y - 1], 'k-', linewidth=0.8)
        ax1.text((i + j) / 2, y + 0.3, label_txt, ha='center', fontsize=8, color='#333333')
    anova_label = f"ANOVA p={p_anova:.3f}" if p_anova >= 0.001 else "ANOVA p<0.001"
    ax1.text(len(animals_with_data) / 2 - 0.5, bar_height + 1, anova_label,
             ha='center', fontsize=9, color='#333333', fontstyle='italic')
    ax1.set_ylim(0, bar_height + 8)
else:
    ax1.set_ylim(0, max(means) * 1.4)

# Sex plot
sexes = ['F', 'M']
sex_means = [df[df['Sex'] == s]['Coloc %'].mean() for s in sexes]
sex_sems = [df[df['Sex'] == s]['Coloc %'].sem() for s in sexes]
ax2.bar(sexes, sex_means, yerr=sex_sems, capsize=5,
        color=[SEX_COLORS[s] for s in sexes],
        edgecolor='white', linewidth=1.2, alpha=0.85, width=0.45, error_kw={'linewidth': 1.2})
for i, s in enumerate(sexes):
    vals = df[df['Sex'] == s]['Coloc %'].values
    jitter = np.random.default_rng(42).uniform(-0.08, 0.08, size=len(vals))
    ax2.scatter([i] * len(vals) + jitter, vals, color='#333333', s=18, alpha=0.6, zorder=5)
ax2.set_ylabel('TdTom+ & TH+ Colocalization %')
ax2.set_title('LC-Restricted Colocalization by Sex')
if len(f_vals) >= 2 and len(m_vals) >= 2:
    sig_label = f"p={p_sex:.3f}" if p_sex >= 0.001 else "p<0.001"
    y_bar = max(sex_means) + max(sex_sems) * 1.5
    ax2.plot([0, 0, 1, 1], [y_bar - 1, y_bar, y_bar, y_bar - 1], 'k-', linewidth=1)
    ax2.text(0.5, y_bar + 0.5, sig_label, ha='center', fontsize=10, color='#333333')
    ax2.set_ylim(0, y_bar + 8)

plt.tight_layout()
fig.savefig(OUT_DIR / "lc_roi_colocalization.png", dpi=200, bbox_inches='tight')
print(f"\nSaved: {OUT_DIR / 'lc_roi_colocalization.png'}")
