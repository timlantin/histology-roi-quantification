#!/usr/bin/env python3
"""
Analyze TH+ (green channel) staining intensity across animals and sex.
Uses same filtered regions and threshold as colocalization analysis.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
from pathlib import Path
from scipy.stats import f_oneway, ttest_ind, mannwhitneyu
import warnings
warnings.filterwarnings('ignore')

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

MERGED_DIR = Path(__file__).parent / "260330_4OHT_TH" / "output2" / "merged"
OUT_DIR = Path(__file__).parent / "colocalization_results"

GREEN_THRESH = 60

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

rows = []
for img_path in sorted(MERGED_DIR.glob("*_merged.png")):
    slide = img_path.stem.split('_Region')[0]
    region = img_path.stem.split('_merged')[0]
    animal = SLIDE_TO_ANIMAL.get(slide)
    if animal is None or region in EXCLUDE:
        continue

    img = np.array(Image.open(img_path).convert("RGB"))
    green_ch = img[:, :, 1].astype(float)

    th_pos = green_ch > GREEN_THRESH
    n_th = int(th_pos.sum())
    total_px = green_ch.size
    th_fraction = n_th / total_px
    mean_intensity = green_ch[th_pos].mean() if n_th > 0 else 0

    rows.append({
        'Animal': animal,
        'Sex': SEX[animal],
        'Region': region,
        'TH+ pixels': n_th,
        'Total pixels': total_px,
        'TH+ fraction': round(th_fraction, 6),
        'Mean TH+ intensity': round(mean_intensity, 2),
    })

df = pd.DataFrame(rows)

print("── TH+ staining per animal ──")
for a in ['OHT-1', 'OHT-2', 'OHT-3', 'OHT-4']:
    sub = df[df['Animal'] == a]
    print(f"  {a} ({SEX[a]}): fraction={sub['TH+ fraction'].mean():.4f}+/-{sub['TH+ fraction'].std():.4f}  "
          f"intensity={sub['Mean TH+ intensity'].mean():.1f}+/-{sub['Mean TH+ intensity'].std():.1f}  n={len(sub)}")

print("\n── TH+ staining per sex ──")
for s in ['F', 'M']:
    sub = df[df['Sex'] == s]
    print(f"  {s}: fraction={sub['TH+ fraction'].mean():.4f}+/-{sub['TH+ fraction'].std():.4f}  "
          f"intensity={sub['Mean TH+ intensity'].mean():.1f}+/-{sub['Mean TH+ intensity'].std():.1f}")

# Stats on TH+ fraction
print("\n── Statistics: TH+ fraction ──")
groups = [g['TH+ fraction'].values for _, g in df.groupby('Animal')]
if len(groups) >= 2 and all(len(g) >= 2 for g in groups):
    f_stat, p_anova = f_oneway(*groups)
    print(f"ANOVA (animals): F={f_stat:.3f}, p={p_anova:.4f} {'*' if p_anova < 0.05 else 'ns'}")

f_vals = df[df['Sex'] == 'F']['TH+ fraction'].values
m_vals = df[df['Sex'] == 'M']['TH+ fraction'].values
t_sex, p_sex = ttest_ind(f_vals, m_vals)
u_sex, p_mw = mannwhitneyu(f_vals, m_vals, alternative='two-sided')
print(f"Sex t-test: t={t_sex:.3f}, p={p_sex:.4f} {'*' if p_sex < 0.05 else 'ns'}")
print(f"Sex Mann-Whitney: U={u_sex:.1f}, p={p_mw:.4f} {'*' if p_mw < 0.05 else 'ns'}")

# Stats on mean TH+ intensity
print("\n── Statistics: Mean TH+ intensity ──")
groups_int = [g['Mean TH+ intensity'].values for _, g in df.groupby('Animal')]
if len(groups_int) >= 2 and all(len(g) >= 2 for g in groups_int):
    f_stat2, p_anova2 = f_oneway(*groups_int)
    print(f"ANOVA (animals): F={f_stat2:.3f}, p={p_anova2:.4f} {'*' if p_anova2 < 0.05 else 'ns'}")

f_int = df[df['Sex'] == 'F']['Mean TH+ intensity'].values
m_int = df[df['Sex'] == 'M']['Mean TH+ intensity'].values
t_int, p_int = ttest_ind(f_int, m_int)
u_int, p_mw_int = mannwhitneyu(f_int, m_int, alternative='two-sided')
print(f"Sex t-test: t={t_int:.3f}, p={p_int:.4f} {'*' if p_int < 0.05 else 'ns'}")
print(f"Sex Mann-Whitney: U={u_int:.1f}, p={p_mw_int:.4f} {'*' if p_mw_int < 0.05 else 'ns'}")

# Plot: TH+ fraction by sex
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 5))

# By animal
animals = ['OHT-1', 'OHT-2', 'OHT-3', 'OHT-4']
means = [df[df['Animal'] == a]['TH+ fraction'].mean() for a in animals]
sems = [df[df['Animal'] == a]['TH+ fraction'].sem() for a in animals]
ax1.bar(animals, means, yerr=sems, capsize=4, color=[COLORS[a] for a in animals],
        edgecolor='white', linewidth=1.2, alpha=0.85, width=0.6, error_kw={'linewidth': 1.2})
for i, a in enumerate(animals):
    vals = df[df['Animal'] == a]['TH+ fraction'].values
    jitter = np.random.default_rng(42).uniform(-0.12, 0.12, size=len(vals))
    ax1.scatter([i] * len(vals) + jitter, vals, color='#333333', s=18, alpha=0.6, zorder=5)
    pass  # no sex labels on x-axis
ax1.set_ylabel('TH+ Fraction (of total pixels)')
ax1.set_title('TH+ Staining by Animal')

# By sex
sexes = ['F', 'M']
sex_means = [df[df['Sex'] == s]['TH+ fraction'].mean() for s in sexes]
sex_sems = [df[df['Sex'] == s]['TH+ fraction'].sem() for s in sexes]
ax2.bar(sexes, sex_means, yerr=sex_sems, capsize=5, color=[SEX_COLORS[s] for s in sexes],
        edgecolor='white', linewidth=1.2, alpha=0.85, width=0.45, error_kw={'linewidth': 1.2})
for i, s in enumerate(sexes):
    vals = df[df['Sex'] == s]['TH+ fraction'].values
    jitter = np.random.default_rng(42).uniform(-0.08, 0.08, size=len(vals))
    ax2.scatter([i] * len(vals) + jitter, vals, color='#333333', s=18, alpha=0.6, zorder=5)
ax2.set_ylabel('TH+ Fraction')
ax2.set_title('TH+ Staining by Sex')
sig_label = f"p={p_sex:.3f}" if p_sex >= 0.001 else "p<0.001"
y_bar = max(sex_means) + max(sex_sems) * 1.5
ax2.plot([0, 0, 1, 1], [y_bar - 0.002, y_bar, y_bar, y_bar - 0.002], 'k-', linewidth=1)
ax2.text(0.5, y_bar + 0.001, sig_label, ha='center', fontsize=10, color='#333333')

plt.tight_layout()
fig.savefig(OUT_DIR / "th_staining_analysis.png", dpi=200, bbox_inches='tight')
print(f"\nSaved: {OUT_DIR / 'th_staining_analysis.png'}")

df.to_csv(OUT_DIR / "th_staining_per_region.csv", index=False)
print(f"Saved: {OUT_DIR / 'th_staining_per_region.csv'}")
