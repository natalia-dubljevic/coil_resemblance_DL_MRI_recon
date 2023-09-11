import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from sigpy.mri.app import SenseRecon

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


iters = [10, 20, 30, 40, 60, 80, 100, 120]
mses = {}

for p_id in patient_ids[0:6]:
    slices = slice_ids.loc[slice_ids['patient_id'] == p_id, 'nlinv_path'].tolist()
    patient_mses = []
    for ind, iter in enumerate(iters):   
        patient_mse = []
        for slice in slices:
            target_img = np.load(slice)
            if target_img.shape[-1] != 170:
                diff = int((target_img.shape[-1] - 170) / 2)  # difference per side
                target_img = target_img[:, :, diff:-diff]
            noise = np.random.normal(0, 2/1000, target_img.shape) + 1j * np.random.normal(0, 2/1000, target_img.shape)
            input_kspace = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(target_img * smap + noise, axes=(-1, -2))), axes=(-1, -2)) * mask 
            input_img = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(input_kspace, axes=(-1, -2))), axes=(-1, -2)) 

            pic = SenseRecon(input_kspace, smap, max_iter=iter, show_pbar=False)
            pic = pic.run()

            pic /= np.max(np.abs(pic))
            target_img /= np.max(np.abs(target_img))

            mse = np.mean((pic - np.squeeze(target_img)) ** 2)
            patient_mse.append(mse)
        patient_mses.append(np.mean(np.array(patient_mse)))
    mses[p_id] = patient_mses

plt.style.use('seaborn')
fig, axes = plt.subplots(1)
for p_id in mses.keys():
    p_mses = mses[p_id]
    axes.plot(iters, p_mses)
axes.set_xlabel('Iterations')
axes.set_ylabel('MSE')
plt.savefig(f'mse_vs_iter_R={R}, radius={radius}_with_noise.png', dpi=300)

        