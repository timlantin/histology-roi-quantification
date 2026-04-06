#!/usr/bin/env python3
"""
Yellow/Green colocalization analysis for 260330_4OHT_TH dataset.

Merged images: R=tdTomato, G=TH, B=DAPI.
- Yellow pixels: TH+ AND tdTomato+ (colocalized)
- Green-only pixels: TH+ but NOT tdTomato+
- Ratio: yellow / (yellow + green-only) = fraction of TH+ neurons expressing tdTomato

Compares across animals (OHT-1 through OHT-4) and sex (F vs M).
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

# ── Config ──
MERGED_DIR = Path(__file__).parent / "260330_4OHT_TH" / "output2" / "merged"
OUT_DIR = Path(__file__).parent / "colocalization_results"
OUT_DIR.mkdir(exist_ok=True)

GREEN_THRESH = int(os.environ.get('GREEN_THRESH', 30))
RED_THRESH = int(os.environ.get('RED_THRESH', 30))

SLIDE_TO_ANIMAL = {
    'Slide1-1': 'OHT-1', 'Slide1-2': 'OHT-1',
    'Slide1-3': 'OHT-3', 'Slide1-4': 'OHT-3', 'Slide1-5': 'OHT-3',
    'Slide1-6': 'OHT-2', 'Slide1-7': 'OHT-2',
    'Slide1-8': 'OHT-4', 'Slide1-9': 'OHT-4',
}

SEX = {
    'OHT-1': 'F', 'OHT-2': 'F',
    'OHT-3': 'M', 'OHT-4': 'M',
}

# Exclusions: bad scans, irrelevant sections, scanner artifacts
EXCLUDE = {
    # All of Slide1-1 (regions 1-6 excluded + region0000 excluded = all gone)
    'Slide1-1_Region0001', 'Slide1-1_Region0002', 'Slide1-1_Region0003',
    'Slide1-1_Region0004', 'Slide1-1_Region0005', 'Slide1-1_Region0006',
    # All Region0000 (overview scans)
    'Slide1-1_Region0000', 'Slide1-2_Region0000', 'Slide1-3_Region0000',
    'Slide1-4_Region0000', 'Slide1-5_Region0000', 'Slide1-6_Region0000',
    'Slide1-7_Region0000', 'Slide1-8_Region0000', 'Slide1-9_Region0000',
    # All of Slide1-3
    'Slide1-3_Region0001', 'Slide1-3_Region0002', 'Slide1-3_Region0003',
    'Slide1-3_Region0004', 'Slide1-3_Region0005',
    # All of Slide1-6
    'Slide1-6_Region0001', 'Slide1-6_Region0002', 'Slide1-6_Region0003',
    'Slide1-6_Region0004',
    # All of Slide1-8
    'Slide1-8_Region0001', 'Slide1-8_Region0002', 'Slide1-8_Region0003',
    'Slide1-8_Region0004', 'Slide1-8_Region0005',
    # Individual exclusions
    'Slide1-4_Region0006',
    'Slide1-9_Region0001',
    'Slide1-9_Region0006',
    # Incomplete/folded sections
    'Slide1-7_Region0004', 'Slide1-4_Region0005',
    # Mostly background, outlier
    'Slide1-9_Region0007',
}

# ── Chart style (Claude artifacts) ──
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
COLORS = {
    'OHT-1': '#E07B54', 'OHT-2': '#E8A07A',
    'OHT-3': '#5B8BD4', 'OHT-4': '#7BAFD4',
}
SEX_COLORS = {'F': '#E07B54', 'M': '#5B8BD4'}

# ── Analysis ──
print(f"Thresholds: green (TH) > {GREEN_THRESH}, red (tdTomato) > {RED_THRESH}")
print(f"Merged dir: {MERGED_DIR}")
print()

rows = []
for img_path in sorted(MERGED_DIR.glob("*_merged.png")):
    slide = img_path.stem.split('_Region')[0]
    region = img_path.stem.split('_merged')[0]
    animal = SLIDE_TO_ANIMAL.get(slide)
    if animal is None:
        continue
    if region in EXCLUDE:
        continue

    img = np.array(Image.open(img_path).convert("RGB"))
    red_ch = img[:, :, 0].astype(float)
    green_ch = img[:, :, 1].astype(float)

    th_pos = green_ch > GREEN_THRESH
    tdtom_pos = red_ch > RED_THRESH
    yellow = th_pos & tdtom_pos       # colocalized
    green_only = th_pos & ~tdtom_pos  # TH+ only

    n_yellow = int(yellow.sum())
    n_green_only = int(green_only.sum())
    n_th_total = n_yellow + n_green_only
    ratio = n_yellow / n_th_total if n_th_total > 0 else 0.0

    rows.append({
        'Animal': animal,
        'Sex': SEX[animal],
        'Slide': slide,
        'Region': region,
        'TH+ pixels': n_th_total,
        'Yellow (coloc)': n_yellow,
        'Green-only (TH)': n_green_only,
        'Yellow/TH+ ratio': round(ratio, 4),
    })
    print(f"  {region:40s}  {animal}  yellow={n_yellow:>8,}  green={n_green_only:>8,}  ratio={ratio:.3f}")

df = pd.DataFrame(rows)
df.to_csv(OUT_DIR / "colocalization_per_region.csv", index=False)
print(f"\n{len(df)} regions analyzed across {df['Animal'].nunique()} animals")

# ── Per-animal summary ──
animal_summary = df.groupby('Animal').agg(
    Sex=('Sex', 'first'),
    n_regions=('Yellow/TH+ ratio', 'count'),
    mean_ratio=('Yellow/TH+ ratio', 'mean'),
    std_ratio=('Yellow/TH+ ratio', 'std'),
    total_TH=('TH+ pixels', 'sum'),
    total_yellow=('Yellow (coloc)', 'sum'),
    total_green=('Green-only (TH)', 'sum'),
).round(4)
animal_summary['overall_ratio'] = (
    animal_summary['total_yellow'] / animal_summary['total_TH']
).round(4)
animal_summary.to_csv(OUT_DIR / "colocalization_per_animal.csv")
print("\n── Per-animal summary ──")
print(animal_summary.to_string())

# ── Per-sex summary ──
sex_summary = df.groupby('Sex').agg(
    n_regions=('Yellow/TH+ ratio', 'count'),
    mean_ratio=('Yellow/TH+ ratio', 'mean'),
    std_ratio=('Yellow/TH+ ratio', 'std'),
).round(4)
print("\n── Per-sex summary ──")
print(sex_summary.to_string())

# ── Statistics ──
print("\n── Statistical tests ──")

# One-way ANOVA across animals
animal_groups = [g['Yellow/TH+ ratio'].values for _, g in df.groupby('Animal')]
if len(animal_groups) >= 2 and all(len(g) >= 2 for g in animal_groups):
    f_stat, p_anova = f_oneway(*animal_groups)
    print(f"One-way ANOVA (animals): F={f_stat:.3f}, p={p_anova:.4f} {'*' if p_anova < 0.05 else 'ns'}")

    # Post-hoc pairwise t-tests
    animals = sorted(df['Animal'].unique())
    print("\nPairwise t-tests (uncorrected):")
    for i in range(len(animals)):
        for j in range(i+1, len(animals)):
            a1 = df[df['Animal'] == animals[i]]['Yellow/TH+ ratio'].values
            a2 = df[df['Animal'] == animals[j]]['Yellow/TH+ ratio'].values
            t, p = ttest_ind(a1, a2)
            print(f"  {animals[i]} vs {animals[j]}: t={t:.3f}, p={p:.4f} {'*' if p < 0.05 else 'ns'}")

# Sex comparison (M vs F)
f_vals = df[df['Sex'] == 'F']['Yellow/TH+ ratio'].values
m_vals = df[df['Sex'] == 'M']['Yellow/TH+ ratio'].values
if len(f_vals) >= 2 and len(m_vals) >= 2:
    t_sex, p_sex = ttest_ind(f_vals, m_vals)
    u_sex, p_mw = mannwhitneyu(f_vals, m_vals, alternative='two-sided')
    print(f"\nSex comparison (F vs M):")
    print(f"  t-test:        t={t_sex:.3f}, p={p_sex:.4f} {'*' if p_sex < 0.05 else 'ns'}")
    print(f"  Mann-Whitney:  U={u_sex:.1f}, p={p_mw:.4f} {'*' if p_mw < 0.05 else 'ns'}")

# ── Plots ──

# 1. Per-animal bar plot with individual ROI points
fig, ax = plt.subplots(figsize=(7, 5))
animals = ['OHT-1', 'OHT-2', 'OHT-3', 'OHT-4']
means = [df[df['Animal'] == a]['Yellow/TH+ ratio'].mean() * 100 for a in animals]
sems = [df[df['Animal'] == a]['Yellow/TH+ ratio'].sem() * 100 for a in animals]
bar_colors = [COLORS[a] for a in animals]

bars = ax.bar(animals, means, yerr=sems, capsize=4, color=bar_colors, edgecolor='white',
              linewidth=1.2, alpha=0.85, width=0.6, error_kw={'linewidth': 1.2})

# Overlay individual data points
for i, a in enumerate(animals):
    vals = df[df['Animal'] == a]['Yellow/TH+ ratio'].values * 100
    jitter = np.random.default_rng(42).uniform(-0.12, 0.12, size=len(vals))
    ax.scatter([i] * len(vals) + jitter, vals, color='#333333', s=18, alpha=0.6, zorder=5)

ax.set_ylabel('TdTom+ & TH+ Colocalization %')
ax.set_title('TH-tdTomato Colocalization by Animal')
ax.set_ylim(0, min(100, max(means) * 1.5))

plt.tight_layout()
fig.savefig(OUT_DIR / "colocalization_by_animal.png", dpi=200, bbox_inches='tight')
print(f"\nSaved: {OUT_DIR / 'colocalization_by_animal.png'}")

# 2. Per-sex bar plot
fig2, ax2 = plt.subplots(figsize=(4, 5))
sexes = ['F', 'M']
sex_means = [df[df['Sex'] == s]['Yellow/TH+ ratio'].mean() * 100 for s in sexes]
sex_sems = [df[df['Sex'] == s]['Yellow/TH+ ratio'].sem() * 100 for s in sexes]

ax2.bar(sexes, sex_means, yerr=sex_sems, capsize=5, color=[SEX_COLORS[s] for s in sexes],
        edgecolor='white', linewidth=1.2, alpha=0.85, width=0.45, error_kw={'linewidth': 1.2})

for i, s in enumerate(sexes):
    vals = df[df['Sex'] == s]['Yellow/TH+ ratio'].values * 100
    jitter = np.random.default_rng(42).uniform(-0.08, 0.08, size=len(vals))
    ax2.scatter([i] * len(vals) + jitter, vals, color='#333333', s=18, alpha=0.6, zorder=5)

ax2.set_ylabel('TdTom+ & TH+ Colocalization %')
ax2.set_title('TH-tdTomato Colocalization by Sex')
ax2.set_ylim(0, min(100, max(sex_means) * 1.5))

if len(f_vals) >= 2 and len(m_vals) >= 2:
    sig_label = f"p={p_sex:.3f}" if p_sex >= 0.001 else "p<0.001"
    y_bar = max(sex_means) + max(sex_sems) * 1.5
    ax2.plot([0, 0, 1, 1], [y_bar - 1, y_bar, y_bar, y_bar - 1], 'k-', linewidth=1)
    ax2.text(0.5, y_bar + 0.5, sig_label, ha='center', fontsize=10, color='#333333')

plt.tight_layout()
fig2.savefig(OUT_DIR / "colocalization_by_sex.png", dpi=200, bbox_inches='tight')
print(f"Saved: {OUT_DIR / 'colocalization_by_sex.png'}")

# 3. Per-region strip chart grouped by animal
fig3, ax3 = plt.subplots(figsize=(10, 5))
for i, a in enumerate(animals):
    sub = df[df['Animal'] == a].sort_values('Region')
    x_pos = np.arange(len(sub)) + i * (len(sub) + 1)
    ax3.bar(x_pos, sub['Yellow/TH+ ratio'].values, color=COLORS[a], alpha=0.8,
            edgecolor='white', linewidth=0.5, label=a)
    for j, (_, row) in enumerate(sub.iterrows()):
        ax3.text(x_pos[j], -0.02, row['Region'].replace('Slide1-', 'S').replace('_Region', '-R'),
                 rotation=90, ha='center', va='top', fontsize=6, color='#888888')

ax3.set_ylabel('Yellow / TH+ Ratio')
ax3.set_title('Colocalization per Region')
ax3.set_xticks([])
ax3.legend(loc='upper right', framealpha=0.8)
ax3.set_ylim(0, min(1.0, df['Yellow/TH+ ratio'].max() * 1.3))
plt.tight_layout()
fig3.savefig(OUT_DIR / "colocalization_per_region.png", dpi=200, bbox_inches='tight')
print(f"Saved: {OUT_DIR / 'colocalization_per_region.png'}")

print("\nDone!")
