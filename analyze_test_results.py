import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Subset
import random
from scipy.ndimage import binary_fill_holes
import sys
from mpl_toolkits.axes_grid1 import make_axes_locatable

from models.cascaded_map_branch import CascadedModel
from dataset import SliceSmapTestDataset, ReImgChannels
from metrics import SSIM, pSNR, phase_metric, mae, nRMSE
from sigpy.mri.app import SenseRecon
import matplotlib as mpl

# get system arguments
#coils, R, smap_style = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
#rand, smap_choice, mask_choice = sys.argv[4], int(sys.argv[5]), int(sys.argv[6])
#test_model, check_valid = int(sys.argv[7]), int(sys.argv[8])
## model stuff
#model_type, version = sys.argv[9], sys.argv[10], 
#blocks, block_depth, filters = int(sys.argv[11]), int(sys.argv[12]), int(sys.argv[13])
#smap_layers, smap_filters = int(sys.argv[14]), int(sys.argv[15])

# if local 
coils = 8
R = 6
style = 'circle_ring'
smap = 'circle_R_12'
analysis = 'phase'

rand = 'neither'
smap_choice = 0  # order is 104, 103, 101, 9, 10, 11, 102, or 10, 11, 12, 8, 9
mask_choice = 0  # 4 and 8 are 0 and 2 respectively

test_model = 0
check_valid = 0
style
model_type = 'smap_branch'
version = f'{smap}_noise_final'

blocks = 5
block_depth = 5
filters = 80
smap_layers = 12
smap_filters = 50

input_channels = coils * 2

def to_complex(img):
    # img is shape batch, c, h , w
    img = img[:, ::2, :, :] + 1j * img[:, 1::2, :, :]
    return img

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
if sys.platform == 'linux' or sys.platform == 'linux2':
    if sys.argv[0].split('/')[2] == 'pasala':
        folder_path = '/home/pasala/Data/12-channel/'
    else:
        folder_path = '/home/natalia.dubljevic/Data/12-channel/'
else:
    folder_path = r'C:\\Users\\natal\Documents\\Data\\12-channel\\'

#folder_path = '/home/natalia.dubljevic/Data/12-channel/'
slice_ids = pd.read_csv(folder_path + 'slice_ids_v2.csv')
slice_ids = slice_ids.loc[(slice_ids['slice'] >= 77) & (slice_ids['slice'] < 178), :]
test_transforms = transforms.Compose([ToTensor(), ReImgChannels()])
target_transforms = transforms.Compose([ToTensor()])


smaps_170 = glob.glob(os.path.join('sensitivity_maps', '218_170', style, f'{smap}.npy'))
smaps_174 = glob.glob(os.path.join('sensitivity_maps', '218_174', style, f'{smap}.npy'))
smaps_180 = glob.glob(os.path.join('sensitivity_maps', '218_180', style, f'{smap}.npy'))
smaps = [smaps_170, smaps_174, smaps_180]

masks_170 = [f'undersampling_masks/218_170/uniform_mask_R={R}.npy']
masks_174 = [f'undersampling_masks/218_174/uniform_mask_R={R}.npy']
masks_180 = [f'undersampling_masks/218_180/uniform_mask_R={R}.npy']
masks = [masks_170, masks_174, masks_180]

model = CascadedModel(input_channels, reps=blocks, block_depth=block_depth, filters=filters, 
                    smap_layers=smap_layers, smap_filters=smap_filters).type(torch.float32)

checkpoint = torch.load(os.path.join('select_model_weights', f'{model_type}_cascaded_model_v{version}.pt'))
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# generate test results

# make datasets
test_data = SliceSmapTestDataset(slice_ids, 'test', smaps, masks, 'nlinv', coils, 
                                data_transforms=test_transforms, target_transforms=target_transforms, 
                                rand=rand, smap_choice=smap_choice, mask_choice=mask_choice)

# selection is usually 0, 100, 300
for index, i in enumerate([300]):  # 1 for iw-apd, 300 for diff image. Jk, same for both (300)
    data = test_data[i]    
    # load data
    input_img, img_label, smap_label = data[0][None, :, :, :].to(device, dtype=torch.float32), data[1].numpy(), data[2].numpy()
    scale_val, kspace, mask = data[3].to(dtype=torch.float32), data[4][None, :, :, :].to(device, dtype=torch.complex64), data[5][None, :, :, :].to(device, dtype=torch.float32)
    smap_path, mask_path, img_path = data[6], data[7], data[8]

    img_id = img_path.split(os.sep)[-1][:-4]
    r_factor= mask_path.split(os.sep)[-1].split('=')[-1].split('.')[0]
    smap_style= smap_path.split(os.sep)[-1].replace('.npy', '')

    img_label = np.squeeze(img_label)
    smap_label = np.squeeze(smap_label)

    # get model prediction
    pred_img, pred_smap = model((input_img, kspace, mask))
    pred_img = pred_img.cpu().detach().numpy()
    pred_img = np.squeeze(to_complex(pred_img))

    pred_smap = pred_smap.cpu().detach().numpy()
    pred_smap = np.squeeze(to_complex(pred_smap))

    # get SENSE pred
    kspace = kspace.cpu().numpy()
    kspace = np.moveaxis(kspace, 1, -1)  # was size 1, 4, 218, 170
    smap_og = np.squeeze(np.load(smap_path))
    smap_og = smap_og[None, :, :, :]

    pic = SenseRecon(np.squeeze(np.moveaxis(kspace, -1, 0)), np.squeeze(smap_og), max_iter=60, lamda=0)
    pic = pic.run()

    if analysis == 'phase':
        phase_map = phase_metric(pred_img, img_label, return_map=True)

        pic = pic / np.max(np.abs(pic))
        img_label = img_label / np.max(np.abs(img_label))
        trad_phase_map = phase_metric(pic, img_label, return_map=True)
        print(max(np.max(phase_map), np.max(trad_phase_map)))

    # plot model results
    #map = mae(pred_img, img_label, return_map=True)
        v_max = max(np.max(phase_map), np.max(trad_phase_map))
        v_min = min(np.min(phase_map), np.min(trad_phase_map))
        print(v_min, v_max)
        if R == 8:
            vmax = 0.513
            vmin = 0
        elif R == 6:
            vmax = 0.229
            vmin = 0
        print(img_id)
        plt.imshow(phase_map, vmin=vmin, vmax=vmax)
        plt.axis('off')
        #plt.colorbar(pad=0.01, label='IW-APD', aspect=50)
        #plt.savefig(f'plots/analysis/IW_phasemap_accel_{img_id}_{model_type}_v{version}_accel_{R}_colorbar', dpi=400, bbox_inches='tight', transparent=True)

        plt.savefig(f'plots/analysis/IW_phasemap_{img_id}_{model_type}_v{version}_accel_{R}_{smap_style}.png', dpi=400, bbox_inches='tight', transparent=True)
        plt.close()

        # plot trad results
        #map = mae(pred_img, img_label, return_map=True)
        plt.imshow(trad_phase_map, vmin=vmin, vmax=vmax)
        #plt.colorbar(pad=0.01, label='IW-APD')
        plt.axis('off')

        plt.savefig(f'plots/analysis/IW_phasemap_{img_id}_trad_accel_{R}_{smap_style}.png', dpi=400, bbox_inches='tight', transparent=True)
        plt.close()

    elif analysis == 'diff':
        #pred_img = pred_img / np.max(np.abs(pred_img))
        #img_label = img_label / np.max(np.abs(img_label))
        model_diff = np.abs(np.abs(pred_img) - np.abs(img_label)) / np.mean(np.abs(img_label)) * 10

        pic = pic / np.max(np.abs(pic))
        img_label = img_label / np.max(np.abs(img_label))
        trad_diff = np.abs(np.abs(pic) - np.abs(img_label)) / np.mean(np.abs(img_label)) * 10

        trad_min, trad_max = np.min(trad_diff), np.max(trad_diff)
        model_min, model_max = np.min(model_diff), np.max(model_diff)
#
        vmax, vmin = max(trad_max, model_max), min(trad_min, model_min)
        print(vmax, vmin)
        if R == 6:
            #vmax, vmin = 9.607, 0
            vmax, vmin = 8.34, 0
        elif R == 8:
            #vmax, vmin = 18.027, 0
            vmax, vmin = 16.89, 0

        #model_norm = mpl.colors.TwoSlopeNorm(vmin=model_min, vcenter=0., vmax=model_max)
        #trad_norm = mpl.colors.TwoSlopeNorm(vmin=trad_min, vcenter=0., vmax=trad_max)

        plt.imshow(model_diff, cmap='gray', vmax=vmax, vmin=vmin)
        #plt.colorbar(mpl.cm.ScalarMappable(norm=model_norm, cmap='RdBu_r'), pad=0.01, label='Difference')
        #plt.colorbar(pad=0.01, label='|Difference|')
        #plt.colorbar(pad=0.01, label='NAE x 10', aspect=50)
        plt.axis('off')
        #plt.colorbar(pad=0.01, label='NAE x 10', aspect=50)
        #plt.savefig(f'plots/analysis/abs_diffmap_{img_id}_{model_type}_v{version}_accel_{R}_colorbar.png', dpi=400, bbox_inches='tight', transparent=True)

        plt.savefig(f'plots/analysis/abs_diffmap_{img_id}_{model_type}_v{version}_accel_{R}_{smap_style}.png', dpi=400, bbox_inches='tight', transparent=True)
        plt.close()

        # plot trad results
        #map = mae(pred_img, img_label, return_map=True)
        plt.imshow(trad_diff, cmap='gray', vmax=vmax, vmin=vmin)
        #plt.colorbar(mpl.cm.ScalarMappable(norm=trad_norm, cmap='RdBu_r'), pad=0.01, label='Difference')
        #plt.colorbar(pad=0.01, label='|Difference|')
        plt.axis('off')

        plt.savefig(f'plots/analysis/abs_diffmap_{img_id}_trad_accel_{R}_{smap_style}.png', dpi=400, bbox_inches='tight', transparent=True)
        plt.close()

    elif analysis == 'nrmse':
        model_diff = nRMSE(np.abs(pred_img), np.abs(img_label))

        pic = pic / np.max(np.abs(pic))
        img_label = img_label / np.max(np.abs(img_label))
        trad_diff = nRMSE(np.abs(pic), np.abs(img_label))

        trad_min, trad_max = np.min(trad_diff), np.max(trad_diff)
        model_min, model_max = np.min(model_diff), np.max(model_diff)

        plt.imshow(model_diff, cmap='RdBu_r')
        plt.colorbar(pad=0.01, label='nRMSE')
        plt.axis('off')

        plt.savefig(f'plots/analysis/nrmse_map_{img_id}_{model_type}_v{version}_accel_{R}_{smap_style}.png', dpi=300, bbox_inches='tight', transparent=True)
        plt.close()

        # plot trad results
        #map = mae(pred_img, img_label, return_map=True)
        plt.imshow(trad_diff, cmap='RdBu_r')
        plt.colorbar(pad=0.01, label='nRMSE')
        plt.axis('off')

        plt.savefig(f'plots/analysis/nrmse_map_{img_id}_trad_accel_{R}_{smap_style}.png', dpi=300, bbox_inches='tight', transparent=True)
        plt.close()





