import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, mannwhitneyu

'''
Stats for coil unqiueness paper. We need to do two things:

1. Check whether there are significant decreases from coil to coil config. friedman test,
and then post-hoc dunn's test. ie) 2 K-W tests + 4x2 dunn per real/unreal

2. Check whether there are differences between DL and CG-SENSE at every coil config.
ie) 5x2 tests for realistic, 5x2 for unrealistic
'''
exp = 'realistic'
R = 6
folder_path = f'/home/pasala/ENSF619_Final_Project/results/{exp}_results'


if exp == 'realistic':
    smaps_versions = ['circle_R_8', 'circle_R_9', 'circle_R_10', 'circle_R_11', 'circle_R_12']
else:
    smaps_versions = ['smap_20', 'smap_40', 'smap_60', 'smap_80', 'smap_100']

for j, smap in enumerate(smaps_versions):
    base_results = pd.read_csv(f'{folder_path}/trad_results_accel_{R}_{smap}_noise_sim.csv')
    model_results = pd.read_csv(f'{folder_path}/model_results_smap_branch_v{smap}_noise_final_accel_{R}_{smap}.csv')


model_version = 'linear_v14'
smap_version = 'smap_9_square_new'  # loking at 9, 10, 101, 102, 103, 104, 11
R = 8

base_results = pd.read_csv(f'{folder_path}/trad_results_{smap_version}.csv')
model_results = pd.read_csv(f'{folder_path}/results_{model_version}_{smap_version}.csv')
base_results = pd.read_csv(f'{folder_path}/trad_results_R_{R}_square_new.csv')
model_results = pd.read_csv(f'{folder_path}/results_linear_v14_R_{R}_square_new.csv')


# look at distribution for each R factor, determine if its normal
fig, axes = plt.subplots(nrows=6, ncols=2, figsize=(10, 14))
fig.tight_layout()
for i, R in enumerate([2, 4, 6, 8, 10, 12]):
    base_r = base_results.loc[base_results['R_factor'] == R, 'ssim_trad'].to_numpy()
    model_r = model_results.loc[model_results['R_factor'] == R, 'ssim_model'].to_numpy()

    axes[i, 0].hist(base_r)
    axes[i, 0].set_title(f'R={R} Baseline SSIM')
    axes[i, 1].hist(model_r)
    axes[i, 1].set_title(f'R={R} Model SSIM')

plt.savefig('plots/analysis/normality_analysis', bbox_inches='tight')

# not normal...can do wilcoxon signed rank test for differences of paired samples
smaps = ['smap_9', 'smap_10', 'smap_101', 'smap_102', 'smap_103', 'smap_104', 'smap_11']
for i, smap in enumerate(smaps):
    results = pd.merge(base_results, model_results, how='left')
    base_r = results.loc[results['smap_style'] == smap, 'ssim_trad'].to_numpy()
    model_r = results.loc[results['smap_style'] == smap, 'ssim_model'].to_numpy()

    diff_ssim = model_r - base_r
    stat, p = wilcoxon(diff_ssim, alternative='greater')
    base_med = np.median(base_r)
    model_med = np.median(model_r)
    base_mean = np.mean(base_r)
    model_mean = np.mean(model_r)
    base_std = np.std(base_r)
    model_std = np.std(model_r)
    mean_diff = np.mean(diff_ssim)
    print(f'Mean diff: {mean_diff:.4f}')
    print(f'Base median: {base_med:.3f}, model median: {model_med:.3f}')
    #print(f'Base mean: {base_mean:.3f}, model mean: {model_mean:.3f}')
    print(f'Base std: {base_std:.3f}, model std: {model_std:.3f}')
    print(f'smap={smap} has p value {p:.5f} \n')


# for determining if there's a signifcant diff between map1 and map 7 for DL model can use Mann whitney U-test
for R in [2, 4, 6, 8]:
    model_results = pd.read_csv(f'{folder_path}/results_linear_v14_R_{R}_square_new.csv')
    map1 = model_results.loc[results['smap_style'] == 'smap_9', 'psnr_model'].to_numpy()
    map7 = model_results.loc[results['smap_style'] == 'smap_11', 'psnr_model'].to_numpy()
    stat, p = mannwhitneyu(map1, map7, alternative='greater')

    print(f'\nP value between first and last map for R {R}: {p:.4f}')
