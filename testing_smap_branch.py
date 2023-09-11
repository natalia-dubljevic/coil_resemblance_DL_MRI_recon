import bart
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import random
from scipy.ndimage import binary_fill_holes
import sys

from models.cascaded_map_branch import CascadedModel
from dataset import SliceSmapTestDataset, ReImgChannels, SliceSmapDataset
from metrics import SSIM, pSNR, phase_metric
from sigpy.mri.app import SenseRecon
#from pygrappa import sense1d, cgsense

# get system arguments
coils, R, smap_style = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
rand, smap_choice, mask_choice = sys.argv[4], int(sys.argv[5]), int(sys.argv[6])
test_model, check_valid = int(sys.argv[7]), int(sys.argv[8])
# model stuff
model_type, version = sys.argv[9], sys.argv[10], 
blocks, block_depth, filters = int(sys.argv[11]), int(sys.argv[12]), int(sys.argv[13])
smap_layers, smap_filters, lam = int(sys.argv[14]), int(sys.argv[15]), float(sys.argv[16])

# if local 
#coils = 8
#R = 8
#smap_style = 'circle_ring'
#
#rand = 'mask'  # usually neither when working with uniform masks
#smap_choice = 2 # order is 104, 103, 101, 9, 10, 11, 102
## or, 10, 11, 12, 8, 9
#mask_choice = 0  # 4 and 8 are 0 and 2 respectively
#
#test_model = 0
#check_valid = 0
#
#model_type = 'smap_branch'
#version = 'circle_R_8_2'
#
#blocks = 5
#block_depth = 5
#filters = 80
#smap_layers = 12
#smap_filters = 50

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

slice_ids = pd.read_csv(folder_path + 'slice_ids_v2.csv')
# remove this if you want results for nearly everything!
slice_ids = slice_ids.loc[(slice_ids['slice'] >= 77) & (slice_ids['slice'] < 178), :]

test_transforms = transforms.Compose([ToTensor(), ReImgChannels()])
target_transforms = transforms.Compose([ToTensor()])

smaps_170 = sorted(glob.glob(os.path.join('sensitivity_maps', '218_170', smap_style, '*.npy')))
smaps_174 = sorted(glob.glob(os.path.join('sensitivity_maps', '218_174', smap_style, '*.npy')))
smaps_180 = sorted(glob.glob(os.path.join('sensitivity_maps', '218_180', smap_style, '*.npy')))
smaps = [smaps_170, smaps_174, smaps_180]

#masks_170 = glob.glob(f'gauss_undersampling_masks/218_170/gauss_mask_R={R}*.npy')
#masks_174 = glob.glob(f'gauss_undersampling_masks/218_174/gauss_mask_R={R}*.npy')
#masks_180 = glob.glob(f'gauss_undersampling_masks/218_180/gauss_mask_R={R}*.npy')

#masks = masks0 + masks1 + masks2



masks_170 = [f'undersampling_masks/218_170/uniform_mask_R={R}.npy']
masks_174 = [f'undersampling_masks/218_174/uniform_mask_R={R}.npy']
masks_180 = [f'undersampling_masks/218_180/uniform_mask_R={R}.npy']
masks = [masks_170, masks_174, masks_180]


if test_model:
    model = CascadedModel(input_channels, reps=blocks, block_depth=block_depth, filters=filters, 
                        smap_layers=smap_layers, smap_filters=smap_filters).type(torch.float32)

    checkpoint = torch.load(os.path.join('model_weights', f'{model_type}_cascaded_model_v{version}.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    #params = list(model.parameters())
    #max_p, min_p = 0, 0
    #for i in range(len(params)):
    #    p_max = torch.max(torch.flatten(params[i])).item()
    #    p_min = torch.min(torch.flatten(params[i])).item()
    #    if p_max > max_p:
    #        max_p = p_max
    #    if p_min < min_p:
    #        min_p = p_min
    #    else:
    #        pass
    #print(max_p, min_p)


if check_valid is True:
    valid_data = SliceSmapDataset(slice_ids, 'val', smaps, masks_170, 'nlinv', coils, data_transforms=test_transforms, target_transforms=test_transforms)
    valid_loader = DataLoader(valid_data, batch_size=1, shuffle=True)

    plt.figure(figsize=(24, 18))
    with torch.no_grad():
        for i, data in enumerate(valid_loader, 0):
            if (i+1)*2 == 22:
                break
            input, label =  data[0].to(device,dtype=torch.float32),  data[1].numpy()
            scale_value, input_kspace, mask  = data[2].to(device,dtype=torch.float32), data[3].to(device,dtype=torch.complex128), data[4].to(device,dtype=torch.float32)
            
            output = model(input, input_kspace, mask, scale_value)
            output = output.cpu().detach().numpy()
            output, label = to_complex(output), to_complex(label)
            ssim = SSIM(np.abs(output), np.abs(label))
            psnr = pSNR(np.abs(output), np.abs(label))
            print(f'SSIM: {ssim:.3f}')
            print(f'pSNR: {psnr:.1f}')

            plt.subplot(5, 4, (i+1)*2-1)
            plt.imshow(np.abs(output))
            plt.title(f'SSIM: {ssim:.3f}')
            plt.subplot(5, 4, (i+1)*2)
            plt.imshow(np.abs(label))
            plt.title(f'pSNR: {psnr:.1f}')

    plt.savefig('plots/valid_results.png', dpi=300)

# otherwise, generate test results
else:
    # make datasets
    test_data = SliceSmapTestDataset(slice_ids, 'test', smaps, masks, 'nlinv', coils, 
                                 data_transforms=test_transforms, target_transforms=target_transforms, 
                                 rand=rand, smap_choice=smap_choice, mask_choice=mask_choice)

    test_loader = DataLoader(test_data, batch_size=1, shuffle=True, worker_init_fn=seed_worker, generator=g)

    # lists for later
    img_id = []
    r_factor = []
    smap_styles = []

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
        print(i, flush=True)
        # load data
        input_img, img_label, smap_label = data[0].to(device, dtype=torch.float32), data[1].numpy(), data[2].numpy()
        scale_val, kspace, mask = data[3].to(dtype=torch.float32), data[4].to(device, dtype=torch.complex64), data[5].to(device, dtype=torch.float32)
        smap_path, mask_path, img_path = data[6], data[7], data[8]

        img_id.append(img_path[0].split(os.sep)[-1][:-4])
        r_factor.append(mask_path[0].split(os.sep)[-1].split('=')[-1].split('.')[0])
        smap_styles.append(smap_path[0].split(os.sep)[-1].replace('.npy', ''))

        img_label = np.squeeze(img_label)
        smap_label = np.squeeze(smap_label)

        if test_model:
            # get prediction
            pred_img, pred_smap = model((input_img, kspace, mask))
            pred_img = pred_img.cpu().detach().numpy()
            pred_img = np.squeeze(to_complex(pred_img))

            pred_smap = pred_smap.cpu().detach().numpy()
            pred_smap = np.squeeze(to_complex(pred_smap))

            # get mask
            #roi_mask = np.where(np.abs(img_label) < 0.25, 0, 1)
            #roi_mask = binary_fill_holes(roi_mask)

            # test model results
            ssim = SSIM(np.abs(pred_img), np.abs(img_label))
            psnr = pSNR(np.abs(pred_img), np.abs(img_label))
            phase = phase_metric(pred_img, img_label)

            ssim_model.append(ssim)
            psnr_model.append(psnr)
            phase_model.append(phase)

            #ssim_mask = SSIM(np.abs(pred_img) * roi_mask, np.abs(img_label) * roi_mask)
            #psnr_mask = pSNR(np.abs(pred_img) * roi_mask, np.abs(img_label) * roi_mask)
            #ssim_model_mask.append(ssim_mask)
            #psnr_model_mask.append(psnr_mask)

            if i in np.arange(0, 10):
                # do images
                ax = fig_img.add_subplot(5, 4, (i+1)*2-1)
                ax.imshow(np.abs(pred_img))
                ax.set_title(f'SSIM: {ssim:.3f}')
                ax = fig_img.add_subplot(5, 4, (i+1)*2)
                ax.imshow(np.abs(img_label))
                ax.set_title(f'pSNR: {psnr:.1f}')
                fig_img.savefig(f'results/realistic_results/model_{model_type}_v{version}_results_R_{r_factor[-1]}_{smap_styles[-1]}.png')

                # now do smaps
                ax = fig_smap.add_subplot(5, 4, (i+1)*2-1)
                ax.imshow(np.abs(pred_smap)[0, :, :])
                ax = fig_smap.add_subplot(5, 4, (i+1)*2)
                ax.imshow(np.abs(smap_label)[0, :, :])
                fig_smap.savefig(f'results/realistic_results/model_{model_type}_v{version}_smap_results_R_{r_factor[-1]}_{smap_styles[-1]}.png')


        else:
            # test traditional results
            kspace = kspace.cpu().numpy()
            kspace = np.moveaxis(kspace, 1, -1)
            #x, y = kspace.shape[1], kspace.shape[2]
            #k = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img_label, axes=(0, 1))), axes=(0, 1))
            #low_res_k = k[int(x / 2) - 10 : int(x / 2) + 10, int(y / 2) - 10 : int(y / 2) + 10]
            #low_res_whole_img = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(low_res_k, axes=(0, 1))), axes=(0, 1))
#
            #low_res_k = kspace[0, int(x / 2) - 10 : int(x / 2) + 10, int(y / 2) - 10 : int(y / 2) + 10, :]
            #low_res_channel_imgs = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(low_res_k, axes=(0, 1))), axes=(0, 1))
            
            # using ecalib for maps
            #kspace_fftmod = bart.bart(1, 'fftmod 6', kspace)
            #smap_ecal = bart.bart(1, 'ecalib -m 1', kspace)

            # using the maps we know are true
            smap_og = np.squeeze(np.load(smap_path[0]))
            #smap_og = np.moveaxis(smap_og, 0, -1)  # for square maps
            smap_og = smap_og[None, :, :, :]

            #pic = np.squeeze(bart.bart(1, 'pics -i 40', kspace, smap_og))
            pic = SenseRecon(np.squeeze(np.moveaxis(kspace, -1, 0)), np.squeeze(smap_og), max_iter=60, lamda=lam, show_pbar=False)
            pic = pic.run()
            #picy = cgsense(np.squeeze(kspace), np.squeeze(smap_og), coil_axis=-1)

            pic = pic / np.max(np.abs(pic))
            #pica = pica / np.max(np.abs(pica))
            #picy = picy / np.max(np.abs(picy))
            img_label = img_label / np.max(np.abs(img_label))

            # get mask
            #roi_mask = np.where(np.abs(img_label) < 0.25, 0, 1)
            #roi_mask = binary_fill_holes(roi_mask)

            # calculate results
            ssim = SSIM(np.abs(pic), np.abs(img_label))
            psnr = pSNR(np.abs(pic), np.abs(img_label))
            phase = phase_metric(pic, img_label)

            #phasea = phase_metric(pica, img_label)
            #ssima = SSIM(np.abs(pica), np.abs(img_label))
            #psnra = pSNR(np.abs(pica), np.abs(img_label))
#
            #phasey = phase_metric(picy, img_label)
            #ssimy = SSIM(np.abs(picy), np.abs(img_label))
            #psnry = pSNR(np.abs(picy), np.abs(img_label))
#
            #print(f'SSIM bart: {ssim}, SSIM sigpy: {ssima}, SSIM mrpy: {ssimy}')
            #print(f'psnr bart: {psnr}, psnr sigpy: {psnra}, psnr mrpy: {psnry}')
            #print(f'phase bart: {phase}, phase sigpy: {phasea}, phase mrpy: {phasey}')

            ssim_trad.append(ssim)
            psnr_trad.append(psnr)
            phase_trad.append(phase)

            print('-----')
            print(f'SSIM: {ssim:.3f}')
            print(f'pSNR: {psnr:.1f}')
            print('-----')
            if i in np.arange(0, 10):
                plt.subplot(5, 4, (i+1)*2-1)
                plt.imshow(np.abs(pic))
                plt.title(f'SSIM: {ssim:.3f}')
                plt.subplot(5, 4, (i+1)*2)
                plt.imshow(np.abs(img_label))
                plt.title(f'pSNR: {psnr:.1f}')
                plt.savefig(f'results/unrealistic_results/espirit_new_results_R_{r_factor[-1]}_{smap_styles[-1]}.png')

    if test_model:
        results = pd.DataFrame({'img_id':img_id, 'R_factor':r_factor, 'smap_style':smap_styles, 
                                'ssim_model':ssim_model, 'psnr_model':psnr_model, 'phase_model':phase_model})
        results.to_csv(f'results/unrealistic_results/model_results_{model_type}_v{version}_accel_{r_factor[-1]}_{smap_styles[-1]}.csv', index=False)

    else:
        results = pd.DataFrame({'img_id':img_id, 'R_factor':r_factor, 'smap_style':smap_styles, 
                                'ssim_trad':ssim_trad, 'psnr_trad':psnr_trad, 'phase_trad':phase_trad})
        results.to_csv(f'results/unrealistic_results/trad_results_accel_{r_factor[-1]}_{smap_styles[-1]}_noise_sim.csv', index=False)

