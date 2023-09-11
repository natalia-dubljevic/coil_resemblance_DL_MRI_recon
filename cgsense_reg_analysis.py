import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from sigpy.mri.app import SenseRecon
from metrics import SSIM, pSNR

# get system arguments
radius, R = int(sys.argv[1]), int(sys.argv[2])

smap = np.load(f'sensitivity_maps/218_170/circle_ring/circle_R_{radius}.npy')
mask = np.load(f'undersampling_masks/218_170/uniform_mask_R={R}.npy')
mask = np.repeat(mask[None, :, :], 8, axis=0)

folder_path = '/home/natalia.dubljevic/Data/12-channel/'
slice_ids = pd.read_csv(folder_path + 'slice_ids_v2.csv')

slice_ids = slice_ids.loc[(slice_ids['split'] == 'train') &
                          (slice_ids['slice'] >= 77) & 
                          (slice_ids['slice'] < 178), :]
patient_ids = slice_ids['patient_id'].unique()

exp_ranges = np.linspace(-4, -11, 5)
lamdas = np.append(np.exp(exp_ranges), 0)
exponents = np.linspace(0, -7, 10)
lamdas = 10 ** exponents
lamdas = np.array([1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7])

print(lamdas, flush=True)

#lamdas = [0.01, 0.001, 0.0005, 0.00025, 0.0001]
mses = {}
ssims = {}
psnrs = {}

for p_id in patient_ids[0:5]:
    slices = slice_ids.loc[slice_ids['patient_id'] == p_id, 'nlinv_path'].tolist()
    patient_mses = []
    patient_ssims = []
    patient_psnrs = []

    for ind, lamda in enumerate(lamdas):   
        patient_mse = []
        patient_ssim = []
        patient_psnr = []
        print(lamda, flush=True)

        for slice in slices:
            target_img = np.load(slice)
            if target_img.shape[-1] != 170:
                diff = int((target_img.shape[-1] - 170) / 2)  # difference per side
                target_img = target_img[:, :, diff:-diff]
            noise = np.random.normal(0, 2/1000, target_img.shape) + 1j * np.random.normal(0, 2/1000, target_img.shape)
            input_kspace = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(target_img * smap + noise, axes=(-1, -2))), axes=(-1, -2)) * mask 
            input_img = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(input_kspace, axes=(-1, -2))), axes=(-1, -2)) 

            pic = SenseRecon(input_kspace, smap, max_iter=60, lamda=lamda, show_pbar=False)
            pic = pic.run()

            pic /= np.max(np.abs(pic))
            target_img /= np.max(np.abs(target_img))

            mse = np.mean((pic - np.squeeze(target_img)) ** 2)
            ssim = SSIM(np.abs(pic), np.abs(np.squeeze(target_img)))
            psnr = pSNR(np.abs(pic), np.abs(np.squeeze(target_img)))
            patient_mse.append(mse)
            patient_ssim.append(ssim)
            patient_psnr.append(psnr)

        patient_mses.append(np.mean(np.array(patient_mse)))
        patient_ssims.append(np.mean(np.array(patient_ssim)))
        patient_psnrs.append(np.mean(np.array(patient_psnr)))

    mses[p_id] = patient_mses
    ssims[p_id] = patient_ssims
    psnrs[p_id] = patient_psnrs

plt.style.use('seaborn')
fig, axes = plt.subplots(1)
for p_id in mses.keys():
    p_mses = mses[p_id]
    axes.plot(lamdas, p_mses)
    axes.set_xscale('log')
axes.set_xlabel('Lamda')
axes.set_ylabel('MSE')
plt.savefig(f'mse_vs_reg_R={R}, radius={radius}_with_noise.png', dpi=300)

fig, axes = plt.subplots(1)
for p_id in ssims.keys():
    p_ssims = ssims[p_id]
    axes.plot(lamdas, p_ssims)
axes.set_xscale('log')
axes.set_xlabel('Lamda')
axes.set_ylabel('SSIM')
plt.savefig(f'ssim_vs_reg_R={R}, radius={radius}_with_noise.png', dpi=300)

fig, axes = plt.subplots(1)
for p_id in psnrs.keys():
    p_psnrs = psnrs[p_id]
    axes.plot(lamdas, p_psnrs)
    axes.set_xscale('log')
axes.set_xlabel('Lamda')
axes.set_ylabel('pSNR')
plt.savefig(f'pSNR_vs_reg_R={R}, radius={radius}_with_noise.png', dpi=300)

        