#!/usr/bin/env python3
"""
Region-level DAPI-normalized colocalization analysis.

For each region, scale R (tdTomato) and G (TH) channels by
(global_mean_DAPI / region_mean_DAPI) before thresholding.
This corrects for technical brightness variation (slice thickness,
staining efficiency, imaging settings) while keeping the threshold
in interpretable 8-bit units.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
from pathlib import Path
from scipy.stats import f_oneway, ttest_ind, mannwhitneyu
import os
import warnings
warnings.filterwarnings('ignore')

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

MERGED_DIR = Path(__file__).parent / "260330_4OHT_TH" / "output2" / "merged"
OUT_DIR = Path(__file__).parent / "colocalization_results"

THRESH = int(os.environ.get('THRESH', 60))

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

# ── Pass 1: compute global mean DAPI ──
print("Computing global mean DAPI...")
kept_paths = []
dapi_means = []
for img_path in sorted(MERGED_DIR.glob("*_merged.png")):
    slide = img_path.stem.split('_Region')[0]
    region = img_path.stem.split('_merged')[0]
    animal = SLIDE_TO_ANIMAL.get(slide)
    if animal is None or region in EXCLUDE:
        continue
    kept_paths.append(img_path)
    img = np.array(Image.open(img_path).convert("RGB"))
    dapi_means.append(img[:, :, 2].astype(np.float64).mean())

GLOBAL_DAPI_MEAN = np.mean(dapi_means)
print(f"Global mean DAPI: {GLOBAL_DAPI_MEAN:.2f} (from {len(kept_paths)} regions)")
print(f"Per-region DAPI range: {min(dapi_means):.1f} - {max(dapi_means):.1f}")
print(f"Threshold: {THRESH} (8-bit, applied after DAPI normalization)")
print()

# ── Pass 2: analyze ──
rows = []
for i, img_path in enumerate(kept_paths):
    slide = img_path.stem.split('_Region')[0]
    region = img_path.stem.split('_merged')[0]
    animal = SLIDE_TO_ANIMAL[slide]

    img = np.array(Image.open(img_path).convert("RGB"))
    red_ch = img[:, :, 0].astype(np.float64)
    green_ch = img[:, :, 1].astype(np.float64)
    blue_ch = img[:, :, 2].astype(np.float64)

    # Region-level DAPI normalization
    region_dapi = dapi_means[i]
    scale = GLOBAL_DAPI_MEAN / region_dapi if region_dapi > 0 else 1.0

    green_scaled = green_ch * scale
    red_scaled = red_ch * scale

    th_pos = green_scaled > THRESH
    tdtom_pos = red_scaled > THRESH
    yellow = th_pos & tdtom_pos
    green_only = th_pos & ~tdtom_pos

    n_yellow = int(yellow.sum())
    n_green_only = int(green_only.sum())
    n_th_total = n_yellow + n_green_only
    ratio = n_yellow / n_th_total if n_th_total > 0 else 0.0

    rows.append({
        'Animal': animal, 'Sex': SEX[animal], 'Slide': slide, 'Region': region,
        'TH+ pixels': n_th_total, 'Yellow (coloc)': n_yellow,
        'Green-only (TH)': n_green_only, 'Coloc ratio': round(ratio, 4),
        'Region DAPI': round(region_dapi, 2), 'Scale factor': round(scale, 3),
    })
    print(f"  {region:40s}  {animal}  coloc={ratio*100:.1f}%  DAPI={region_dapi:.1f}  scale={scale:.3f}")

df = pd.DataFrame(rows)
df.to_csv(OUT_DIR / "dapi_normalized_per_region.csv", index=False)
print(f"\n{len(df)} regions analyzed")

# ── Per-animal summary ──
print("\n── Per-animal ──")
animal_summary = df.groupby('Animal').agg(
    Sex=('Sex', 'first'),
    n_regions=('Coloc ratio', 'count'),
    mean_coloc=('Coloc ratio', lambda x: round(x.mean() * 100, 1)),
    std_coloc=('Coloc ratio', lambda x: round(x.std() * 100, 1)),
    mean_dapi=('Region DAPI', 'mean'),
    mean_scale=('Scale factor', 'mean'),
)
for a in ['OHT-1', 'OHT-2', 'OHT-3', 'OHT-4']:
    r = animal_summary.loc[a]
    print(f"  {a} ({r['Sex']}): {r['mean_coloc']}% +/- {r['std_coloc']}  "
          f"DAPI={r['mean_dapi']:.1f}  scale={r['mean_scale']:.3f}  n={r['n_regions']}")

# ── Per-sex ──
print("\n── Per-sex ──")
for s in ['F', 'M']:
    sub = df[df['Sex'] == s]
    print(f"  {s}: {sub['Coloc ratio'].mean()*100:.1f}% +/- {sub['Coloc ratio'].std()*100:.1f}")

# ── Statistics ──
print("\n── Statistics ──")
groups = [g['Coloc ratio'].values for _, g in df.groupby('Animal')]
f_stat, p_anova = f_oneway(*groups)
print(f"ANOVA (animals): F={f_stat:.3f}, p={p_anova:.4f} {'*' if p_anova < 0.05 else 'ns'}")

animals = sorted(df['Animal'].unique())
pairwise_p = {}
for i in range(len(animals)):
    for j in range(i+1, len(animals)):
        a1 = df[df['Animal'] == animals[i]]['Coloc ratio'].values
        a2 = df[df['Animal'] == animals[j]]['Coloc ratio'].values
        t, p = ttest_ind(a1, a2)
        pairwise_p[(i, j)] = p
        print(f"  {animals[i]} vs {animals[j]}: t={t:.3f}, p={p:.4f} {'*' if p < 0.05 else 'ns'}")

f_vals = df[df['Sex'] == 'F']['Coloc ratio'].values
m_vals = df[df['Sex'] == 'M']['Coloc ratio'].values
t_sex, p_sex = ttest_ind(f_vals, m_vals)
u_sex, p_mw = mannwhitneyu(f_vals, m_vals, alternative='two-sided')
print(f"\nSex: F={f_vals.mean()*100:.1f}% vs M={m_vals.mean()*100:.1f}%")
print(f"  t-test: p={p_sex:.4f} {'*' if p_sex < 0.05 else 'ns'}")
print(f"  Mann-Whitney: p={p_mw:.4f} {'*' if p_mw < 0.05 else 'ns'}")

# ── Plots ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

animals_list = ['OHT-1', 'OHT-2', 'OHT-3', 'OHT-4']
means = [df[df['Animal'] == a]['Coloc ratio'].mean() * 100 for a in animals_list]
sems = [df[df['Animal'] == a]['Coloc ratio'].sem() * 100 for a in animals_list]

ax1.bar(animals_list, means, yerr=sems, capsize=4, color=[COLORS[a] for a in animals_list],
        edgecolor='white', linewidth=1.2, alpha=0.85, width=0.6, error_kw={'linewidth': 1.2})
for i, a in enumerate(animals_list):
    vals = df[df['Animal'] == a]['Coloc ratio'].values * 100
    jitter = np.random.default_rng(42).uniform(-0.12, 0.12, size=len(vals))
    ax1.scatter([i] * len(vals) + jitter, vals, color='#333333', s=18, alpha=0.6, zorder=5)
ax1.set_ylabel('TdTom+ & TH+ Colocalization %')
ax1.set_title('DAPI-Normalized Colocalization by Animal')

# Significance bars for all pairwise comparisons
bar_height = max(means) + max(sems) * 2 + 2
bar_step = 5
for (i, j), p in sorted(pairwise_p.items(), key=lambda x: abs(x[0][1] - x[0][0])):
    if p < 0.05:
        label = f"p={p:.3f}" if p >= 0.001 else "p<0.001"
    else:
        label = "ns"
    y = bar_height
    bar_height += bar_step
    ax1.plot([i, i, j, j], [y - 1, y, y, y - 1], 'k-', linewidth=0.8)
    ax1.text((i + j) / 2, y + 0.3, label, ha='center', fontsize=8, color='#333333')

anova_label = f"ANOVA p={p_anova:.3f}" if p_anova >= 0.001 else "ANOVA p<0.001"
ax1.text(1.5, bar_height + 1, anova_label, ha='center', fontsize=9, color='#333333', fontstyle='italic')

ax1.set_ylim(0, bar_height + 8)

sexes = ['F', 'M']
sex_means = [df[df['Sex'] == s]['Coloc ratio'].mean() * 100 for s in sexes]
sex_sems = [df[df['Sex'] == s]['Coloc ratio'].sem() * 100 for s in sexes]
ax2.bar(sexes, sex_means, yerr=sex_sems, capsize=5, color=[SEX_COLORS[s] for s in sexes],
        edgecolor='white', linewidth=1.2, alpha=0.85, width=0.45, error_kw={'linewidth': 1.2})
for i, s in enumerate(sexes):
    vals = df[df['Sex'] == s]['Coloc ratio'].values * 100
    jitter = np.random.default_rng(42).uniform(-0.08, 0.08, size=len(vals))
    ax2.scatter([i] * len(vals) + jitter, vals, color='#333333', s=18, alpha=0.6, zorder=5)
ax2.set_ylabel('TdTom+ & TH+ Colocalization %')
ax2.set_title('DAPI-Normalized Colocalization by Sex')
ax2.set_ylim(0, min(100, max(sex_means) * 1.4))
sig_label = f"p={p_sex:.3f}" if p_sex >= 0.001 else "p<0.001"
y_bar = max(sex_means) + max(sex_sems) * 1.5
ax2.plot([0, 0, 1, 1], [y_bar - 1, y_bar, y_bar, y_bar - 1], 'k-', linewidth=1)
ax2.text(0.5, y_bar + 0.5, sig_label, ha='center', fontsize=10, color='#333333')

plt.tight_layout()
fig.savefig(OUT_DIR / "dapi_normalized_colocalization.png", dpi=200, bbox_inches='tight')
print(f"\nSaved: {OUT_DIR / 'dapi_normalized_colocalization.png'}")
