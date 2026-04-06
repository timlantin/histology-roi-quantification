#!/usr/bin/env python3
"""
DAPI-normalized TH+ (green) expression analysis.
Proxy for LC neuron density — are there differences across animals and sex?

Region-level normalization: scale green channel by (global_mean_DAPI / region_mean_DAPI)
before thresholding, to control for technical variation.
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

THRESH = 60

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

# ── Pass 1: compute global mean DAPI and collect paths ──
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
print(f"Global mean DAPI: {GLOBAL_DAPI_MEAN:.2f} ({len(kept_paths)} regions)")
print(f"Threshold: {THRESH} (applied after DAPI normalization)\n")

# ── Pass 2: analyze TH+ expression ──
rows = []
for i, img_path in enumerate(kept_paths):
    slide = img_path.stem.split('_Region')[0]
    region = img_path.stem.split('_merged')[0]
    animal = SLIDE_TO_ANIMAL[slide]

    img = np.array(Image.open(img_path).convert("RGB"))
    green_ch = img[:, :, 1].astype(np.float64)
    blue_ch = img[:, :, 2].astype(np.float64)

    # Region-level DAPI normalization
    region_dapi = dapi_means[i]
    scale = GLOBAL_DAPI_MEAN / region_dapi if region_dapi > 0 else 1.0
    green_scaled = green_ch * scale

    th_pos = green_scaled > THRESH
    n_th = int(th_pos.sum())
    total_px = green_ch.size
    th_fraction = n_th / total_px
    mean_intensity = green_scaled[th_pos].mean() if n_th > 0 else 0

    rows.append({
        'Animal': animal, 'Sex': SEX[animal], 'Region': region,
        'TH+ pixels': n_th, 'Total pixels': total_px,
        'TH+ fraction': round(th_fraction, 6),
        'Mean TH+ intensity': round(mean_intensity, 2),
        'Region DAPI': round(region_dapi, 2),
        'Scale factor': round(scale, 3),
    })
    print(f"  {region:40s}  {animal}  TH+={n_th:>10,}  frac={th_fraction:.4f}  int={mean_intensity:.1f}")

df = pd.DataFrame(rows)
df.to_csv(OUT_DIR / "th_dapi_normalized_per_region.csv", index=False)
print(f"\n{len(df)} regions")

# ── Per-animal summary ──
print("\n── TH+ expression per animal (DAPI-normalized) ──")
for a in ['OHT-1', 'OHT-2', 'OHT-3', 'OHT-4']:
    sub = df[df['Animal'] == a]
    print(f"  {a} ({SEX[a]}): fraction={sub['TH+ fraction'].mean():.4f}+/-{sub['TH+ fraction'].std():.4f}  "
          f"intensity={sub['Mean TH+ intensity'].mean():.1f}+/-{sub['Mean TH+ intensity'].std():.1f}  n={len(sub)}")

print("\n── TH+ expression per sex ──")
for s in ['F', 'M']:
    sub = df[df['Sex'] == s]
    print(f"  {s}: fraction={sub['TH+ fraction'].mean():.4f}+/-{sub['TH+ fraction'].std():.4f}  "
          f"intensity={sub['Mean TH+ intensity'].mean():.1f}+/-{sub['Mean TH+ intensity'].std():.1f}")

# ── Statistics: TH+ fraction (proxy for LC neuron count) ──
print("\n── Statistics: TH+ fraction ──")
groups_frac = {a: df[df['Animal'] == a]['TH+ fraction'].values for a in ['OHT-1','OHT-2','OHT-3','OHT-4']}
f_stat_frac, p_anova_frac = f_oneway(*groups_frac.values())
print(f"ANOVA (animals): F={f_stat_frac:.3f}, p={p_anova_frac:.4f} {'*' if p_anova_frac < 0.05 else 'ns'}")

animals_list = ['OHT-1', 'OHT-2', 'OHT-3', 'OHT-4']
pairwise_p_frac = {}
for i in range(len(animals_list)):
    for j in range(i+1, len(animals_list)):
        t, p = ttest_ind(groups_frac[animals_list[i]], groups_frac[animals_list[j]])
        pairwise_p_frac[(i, j)] = p
        print(f"  {animals_list[i]} vs {animals_list[j]}: t={t:.3f}, p={p:.4f} {'*' if p < 0.05 else 'ns'}")

f_frac = df[df['Sex'] == 'F']['TH+ fraction'].values
m_frac = df[df['Sex'] == 'M']['TH+ fraction'].values
t_sex_frac, p_sex_frac = ttest_ind(f_frac, m_frac)
u_sex_frac, p_mw_frac = mannwhitneyu(f_frac, m_frac, alternative='two-sided')
print(f"\nSex: F={f_frac.mean():.4f} vs M={m_frac.mean():.4f}")
print(f"  t-test: p={p_sex_frac:.4f} {'*' if p_sex_frac < 0.05 else 'ns'}")
print(f"  Mann-Whitney: p={p_mw_frac:.4f} {'*' if p_mw_frac < 0.05 else 'ns'}")

# ── Statistics: Mean TH+ intensity ──
print("\n── Statistics: Mean TH+ intensity ──")
groups_int = {a: df[df['Animal'] == a]['Mean TH+ intensity'].values for a in animals_list}
f_stat_int, p_anova_int = f_oneway(*groups_int.values())
print(f"ANOVA (animals): F={f_stat_int:.3f}, p={p_anova_int:.4f} {'*' if p_anova_int < 0.05 else 'ns'}")

pairwise_p_int = {}
for i in range(len(animals_list)):
    for j in range(i+1, len(animals_list)):
        t, p = ttest_ind(groups_int[animals_list[i]], groups_int[animals_list[j]])
        pairwise_p_int[(i, j)] = p
        print(f"  {animals_list[i]} vs {animals_list[j]}: t={t:.3f}, p={p:.4f} {'*' if p < 0.05 else 'ns'}")

f_int = df[df['Sex'] == 'F']['Mean TH+ intensity'].values
m_int = df[df['Sex'] == 'M']['Mean TH+ intensity'].values
t_sex_int, p_sex_int = ttest_ind(f_int, m_int)
u_sex_int, p_mw_int = mannwhitneyu(f_int, m_int, alternative='two-sided')
print(f"\nSex: F={f_int.mean():.1f} vs M={m_int.mean():.1f}")
print(f"  t-test: p={p_sex_int:.4f} {'*' if p_sex_int < 0.05 else 'ns'}")
print(f"  Mann-Whitney: p={p_mw_int:.4f} {'*' if p_mw_int < 0.05 else 'ns'}")

# ── Plots ──
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# -- Top row: TH+ fraction (proxy for LC neuron count) --
# By animal
ax = axes[0, 0]
means_frac = [df[df['Animal'] == a]['TH+ fraction'].mean() * 100 for a in animals_list]
sems_frac = [df[df['Animal'] == a]['TH+ fraction'].sem() * 100 for a in animals_list]
ax.bar(animals_list, means_frac, yerr=sems_frac, capsize=4,
       color=[COLORS[a] for a in animals_list], edgecolor='white',
       linewidth=1.2, alpha=0.85, width=0.6, error_kw={'linewidth': 1.2})
for i, a in enumerate(animals_list):
    vals = df[df['Animal'] == a]['TH+ fraction'].values * 100
    jitter = np.random.default_rng(42).uniform(-0.12, 0.12, size=len(vals))
    ax.scatter([i] * len(vals) + jitter, vals, color='#333333', s=18, alpha=0.6, zorder=5)
ax.set_ylabel('TH+ Fraction (%)')
ax.set_title('TH+ Expression by Animal (DAPI-normalized)')

bar_h = max(means_frac) + max(sems_frac) * 2 + 0.05
bar_step = 0.12
for (i, j), p in sorted(pairwise_p_frac.items(), key=lambda x: abs(x[0][1] - x[0][0])):
    label = f"p={p:.3f}" if p >= 0.001 else "p<0.001"
    if p >= 0.05:
        label = "ns"
    y = bar_h
    bar_h += bar_step
    ax.plot([i, i, j, j], [y - 0.02, y, y, y - 0.02], 'k-', linewidth=0.8)
    ax.text((i + j) / 2, y + 0.005, label, ha='center', fontsize=8, color='#333333')
anova_label = f"ANOVA p={p_anova_frac:.3f}" if p_anova_frac >= 0.001 else "ANOVA p<0.001"
ax.text(1.5, bar_h + 0.02, anova_label, ha='center', fontsize=9, color='#333333', fontstyle='italic')
ax.set_ylim(0, bar_h + 0.15)

# By sex
ax = axes[0, 1]
sexes = ['F', 'M']
sex_means_frac = [df[df['Sex'] == s]['TH+ fraction'].mean() * 100 for s in sexes]
sex_sems_frac = [df[df['Sex'] == s]['TH+ fraction'].sem() * 100 for s in sexes]
ax.bar(sexes, sex_means_frac, yerr=sex_sems_frac, capsize=5,
       color=[SEX_COLORS[s] for s in sexes], edgecolor='white',
       linewidth=1.2, alpha=0.85, width=0.45, error_kw={'linewidth': 1.2})
for i, s in enumerate(sexes):
    vals = df[df['Sex'] == s]['TH+ fraction'].values * 100
    jitter = np.random.default_rng(42).uniform(-0.08, 0.08, size=len(vals))
    ax.scatter([i] * len(vals) + jitter, vals, color='#333333', s=18, alpha=0.6, zorder=5)
ax.set_ylabel('TH+ Fraction (%)')
ax.set_title('TH+ Expression by Sex (DAPI-normalized)')
sig = f"p={p_sex_frac:.3f}" if p_sex_frac >= 0.001 else "p<0.001"
y_bar = max(sex_means_frac) + max(sex_sems_frac) * 1.5
ax.plot([0, 0, 1, 1], [y_bar - 0.02, y_bar, y_bar, y_bar - 0.02], 'k-', linewidth=0.8)
ax.text(0.5, y_bar + 0.005, sig, ha='center', fontsize=9, color='#333333')
ax.set_ylim(0, y_bar + 0.15)

# -- Bottom row: Mean TH+ intensity --
# By animal
ax = axes[1, 0]
means_int = [df[df['Animal'] == a]['Mean TH+ intensity'].mean() for a in animals_list]
sems_int = [df[df['Animal'] == a]['Mean TH+ intensity'].sem() for a in animals_list]
ax.bar(animals_list, means_int, yerr=sems_int, capsize=4,
       color=[COLORS[a] for a in animals_list], edgecolor='white',
       linewidth=1.2, alpha=0.85, width=0.6, error_kw={'linewidth': 1.2})
for i, a in enumerate(animals_list):
    vals = df[df['Animal'] == a]['Mean TH+ intensity'].values
    jitter = np.random.default_rng(42).uniform(-0.12, 0.12, size=len(vals))
    ax.scatter([i] * len(vals) + jitter, vals, color='#333333', s=18, alpha=0.6, zorder=5)
ax.set_ylabel('Mean TH+ Intensity (a.u.)')
ax.set_title('TH+ Intensity by Animal (DAPI-normalized)')

bar_h = max(means_int) + max(sems_int) * 2 + 2
bar_step = 4
for (i, j), p in sorted(pairwise_p_int.items(), key=lambda x: abs(x[0][1] - x[0][0])):
    label = f"p={p:.3f}" if p >= 0.001 else "p<0.001"
    if p >= 0.05:
        label = "ns"
    y = bar_h
    bar_h += bar_step
    ax.plot([i, i, j, j], [y - 0.5, y, y, y - 0.5], 'k-', linewidth=0.8)
    ax.text((i + j) / 2, y + 0.2, label, ha='center', fontsize=8, color='#333333')
anova_label = f"ANOVA p={p_anova_int:.3f}" if p_anova_int >= 0.001 else "ANOVA p<0.001"
ax.text(1.5, bar_h + 1, anova_label, ha='center', fontsize=9, color='#333333', fontstyle='italic')
ax.set_ylim(0, bar_h + 6)

# By sex
ax = axes[1, 1]
sex_means_int = [df[df['Sex'] == s]['Mean TH+ intensity'].mean() for s in sexes]
sex_sems_int = [df[df['Sex'] == s]['Mean TH+ intensity'].sem() for s in sexes]
ax.bar(sexes, sex_means_int, yerr=sex_sems_int, capsize=5,
       color=[SEX_COLORS[s] for s in sexes], edgecolor='white',
       linewidth=1.2, alpha=0.85, width=0.45, error_kw={'linewidth': 1.2})
for i, s in enumerate(sexes):
    vals = df[df['Sex'] == s]['Mean TH+ intensity'].values
    jitter = np.random.default_rng(42).uniform(-0.08, 0.08, size=len(vals))
    ax.scatter([i] * len(vals) + jitter, vals, color='#333333', s=18, alpha=0.6, zorder=5)
ax.set_ylabel('Mean TH+ Intensity (a.u.)')
ax.set_title('TH+ Intensity by Sex (DAPI-normalized)')
sig = f"p={p_sex_int:.3f}" if p_sex_int >= 0.001 else "p<0.001"
y_bar = max(sex_means_int) + max(sex_sems_int) * 1.5
ax.plot([0, 0, 1, 1], [y_bar - 0.5, y_bar, y_bar, y_bar - 0.5], 'k-', linewidth=0.8)
ax.text(0.5, y_bar + 0.2, sig, ha='center', fontsize=9, color='#333333')
ax.set_ylim(0, y_bar + 5)

plt.tight_layout()
fig.savefig(OUT_DIR / "th_dapi_normalized_analysis.png", dpi=200, bbox_inches='tight')
print(f"\nSaved: {OUT_DIR / 'th_dapi_normalized_analysis.png'}")
