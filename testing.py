import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import sys
import torch
from torchvision.transforms import ToTensor, Compose
from torch.utils.data import DataLoader

from cascaded_map_branch import CascadedModel, to_complex
from dataset import SliceTestDataset, ReImgChannels
from metrics import SSIM, PSNR, phase_metric
from sigpy.mri.app import SenseRecon


# get system arguments
coils, R, smap_style = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
rand, smap, test_model = sys.argv[4], sys.argv[5], int(sys.argv[6])
# model stuff
model_type, version = sys.argv[8], sys.argv[9], 
blocks, block_depth, filters = int(sys.argv[10]), int(sys.argv[11]), int(sys.argv[12])
smap_layers, smap_filters, lam = int(sys.argv[13]), int(sys.argv[14]), float(sys.argv[15])

# initiate some random seeds and check cuda
np.random.seed(0)
random.seed(0)
torch.manual_seed(0)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)

device='cuda:0' if torch.cuda.is_available() else 'cpu'

# load data and choose transforms
folder_path = '/home/pasala/Data/12-channel/'

slice_ids = pd.read_csv(folder_path + 'slice_ids_v2.csv')
slice_ids = slice_ids.loc[((slice_ids['slice'] >= 77) 
                           & (slice_ids['slice'] < 178)), :] # keep only central 100 for testing

test_transforms = Compose([ToTensor(), ReImgChannels()])
target_transforms = Compose([ToTensor()])

smaps_170 = sorted(glob.glob(os.path.join('sensitivity_maps', '218_170', smap_style, f'{smap}.npy')))
smaps_174 = sorted(glob.glob(os.path.join('sensitivity_maps', '218_174', smap_style, f'{smap}.npy')))
smaps_180 = sorted(glob.glob(os.path.join('sensitivity_maps', '218_180', smap_style, f'{smap}.npy')))
smaps = [smaps_170, smaps_174, smaps_180]

masks_170 = [f'undersampling_masks/218_170/uniform_mask_R={R}.npy']
masks_174 = [f'undersampling_masks/218_174/uniform_mask_R={R}.npy']
masks_180 = [f'undersampling_masks/218_180/uniform_mask_R={R}.npy']
masks = [masks_170, masks_174, masks_180]

if test_model:
    model = CascadedModel(coils * 2, reps=blocks, block_depth=block_depth, filters=filters, 
                        smap_layers=smap_layers, smap_filters=smap_filters).type(torch.float32)

    checkpoint = torch.load(os.path.join('model_weights', f'{model_type}_cascaded_model_v{version}.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

# make datasets
test_data = SliceTestDataset(slice_ids, 'test', smaps, masks, 'nlinv', coils, 
                                data_transforms=test_transforms, target_transforms=target_transforms, 
                                rand=rand, smap_choice=0, mask_choice=0)
test_loader = DataLoader(test_data, batch_size=1, shuffle=True, worker_init_fn=seed_worker, generator=g)

# lists for later
img_id = []

if test_model:
    ssim_model = []
    psnr_model = []
    phase_model = []

else:
    ssim_trad = []
    psnr_trad = []
    phase_trad = []


fig_img = plt.figure(figsize=(24, 18))
fig_smap = plt.figure(figsize=(24, 18))
for i, data in enumerate(test_loader):
    input_img, img_label, smap_label = data[0].to(device, dtype=torch.float32), data[1].numpy(), data[2].numpy()
    scale_val, kspace, mask = data[3].to(dtype=torch.float32), data[4].to(device, dtype=torch.complex64), data[5].to(device, dtype=torch.float32)
    smap_path, mask_path, img_path = data[6], data[7], data[8]

    img_id.append(img_path[0].split(os.sep)[-1][:-4])

    img_label = np.squeeze(img_label)
    smap_label = np.squeeze(smap_label)

    if test_model:
        # get prediction
        pred_img, pred_smap = model((input_img, kspace, mask))
        pred_img = pred_img.cpu().detach().numpy()
        pred_img = np.squeeze(to_complex(pred_img))

        pred_smap = pred_smap.cpu().detach().numpy()
        pred_smap = np.squeeze(to_complex(pred_smap))

        # test model results
        ssim = SSIM(np.abs(pred_img), np.abs(img_label))
        psnr = PSNR(np.abs(pred_img), np.abs(img_label))
        phase = phase_metric(pred_img, img_label)

        ssim_model.append(ssim)
        psnr_model.append(psnr)
        phase_model.append(phase)

        if i in np.arange(0, 10):
            # do images
            ax = fig_img.add_subplot(5, 4, (i + 1) * 2 - 1)
            ax.imshow(np.abs(pred_img))
            ax.text(3, pred_img.shape[0] - 13, f'SSIM: {ssim:.2f}', c='y', fontsize='14')
            ax.text(3, pred_img.shape[0] - 25, f'PSNR: {psnr:.2f}', c='y', fontsize='14')
            ax.set_title(f'Prediction')

            ax = fig_img.add_subplot(5, 4, (i + 1) * 2)
            ax.imshow(np.abs(img_label))
            ax.set_title(f'Label')
            fig_img.savefig(f'results/results/model_{model_type}_v{version}_results_accel_{R}_{smap}.png')

            # now do smaps
            ax = fig_smap.add_subplot(5, 4, (i + 1) * 2 - 1)
            ax.imshow(np.abs(pred_smap)[0, :, :])
            ax = fig_smap.add_subplot(5, 4, (i + 1) * 2)
            ax.imshow(np.abs(smap_label)[0, :, :])
            fig_smap.savefig(f'results/results/model_{model_type}_v{version}_smap_results_accel_{R}_{smap}.png')


    else:
        # test traditional results
        kspace = kspace.cpu().numpy()
        kspace = np.moveaxis(kspace, 1, -1)

        # using the original sensitivity maps
        smap_og = np.squeeze(np.load(smap_path[0]))
        smap_og = smap_og[None, :, :, :]

        rec_img_app = SenseRecon(np.squeeze(np.moveaxis(kspace, -1, 0)), np.squeeze(smap_og), max_iter=60, lamda=lam, show_pbar=False)
        rec_img = rec_img_app.run()
        
        # scale img/label prior to calculaitng metrics for CG-SENSE
        rec_img = rec_img / np.max(np.abs(rec_img))
        img_label = img_label / np.max(np.abs(img_label))

        # calculate metrics
        ssim = SSIM(np.abs(rec_img), np.abs(img_label))
        psnr = PSNR(np.abs(rec_img), np.abs(img_label))
        phase = phase_metric(rec_img, img_label)

        ssim_trad.append(ssim)
        psnr_trad.append(psnr)
        phase_trad.append(phase)

        if i in np.arange(0, 10):
            ax = fig_img.add_subplot(5, 4, (i + 1) * 2 - 1)
            ax.imshow(np.abs(rec_img))
            ax.text(3, rec_img.shape[0] - 13, f'SSIM: {ssim:.2f}', c='y', fontsize='14')
            ax.text(3, rec_img.shape[0] - 25, f'PSNR: {psnr:.2f}', c='y', fontsize='14')
            ax.set_title(f'Prediction')
            
            ax = fig_img.add_subplot(5, 4, (i + 1) * 2)
            ax.imshow(np.abs(img_label))
            ax.set_title(f'Label')
            plt.savefig(f'results/results/trad_results_accel_{R}_{smap}.png')

if test_model:
    results = pd.DataFrame({'img_id': img_id, 'R_factor': R, 'smap_style': smap, 
                            'ssim_model': ssim_model, 'psnr_model': psnr_model, 'phase_model': phase_model})
    results.to_csv(f'results/results/model_results_{model_type}_v{version}_accel_{R}_{smap}.csv', index=False)

else:
    results = pd.DataFrame({'img_id': img_id, 'R_factor': R, 'smap_style': smap, 
                            'ssim_trad': ssim_trad, 'psnr_trad': psnr_trad, 'phase_trad': phase_trad})
    results.to_csv(f'results/results/trad_results_accel_{R}_{smap}.csv', index=False)

