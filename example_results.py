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
from sys import platform

from models.cascaded_map_branch import CascadedModel
from dataset import SliceSmapTestDataset, ReImgChannels, SliceSmapTestDataset
from metrics import SSIM, pSNR, phase_metric
from sigpy.mri.app import SenseRecon
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

'''
Generate some example model results
'''

def plot_inset(img, ssim, psnr, zoom=2, bbox_anchor=(1.15, 0.235), x_min=130, x_max=155, y_min=100, y_max=125, stats=True):
    extent = (0, img.shape[1], 0, img.shape[0])

    fig, axe = plt.subplots(nrows=1, ncols=1)
    axe.imshow(img, extent=extent, cmap='gray')
    axe.axis('off')
    if stats:
        axe.text(3, img.shape[0] - 13, f'SSIM: {ssim:.2f}', c='y', fontsize='14')
        axe.text(3, img.shape[0] - 25, f'PSNR: {psnr:.1f}', c='y', fontsize='14')

    zoom_axe = zoomed_inset_axes(axe, zoom=zoom, bbox_to_anchor=bbox_anchor, bbox_transform= axe.transAxes, borderpad=0)
    zoom_axe.imshow(img, extent=extent, cmap='gray')
    zoom_axe.set_xlim(x_min, x_max)
    zoom_axe.set_ylim(y_min, y_max)

    zoom_axe.tick_params(top=False,
               bottom=False,
               left=False,
               right=False,
               labelleft=False,
               labelbottom=False)
    for axis in ['top','bottom','left','right']:
        zoom_axe.spines[axis].set_color('y')
        zoom_axe.spines[axis].set_linewidth(3)

    mark_inset(axe, zoom_axe, loc1=1, loc2=3, fc="none", ec="y")
    return fig


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
if platform == 'linux' or platform == 'linux2':
    #folder_path = '/home/natalia.dubljevic/Data/12-channel/'
    folder_path = '/home/pasala/Data/12-channel/'
else:
    folder_path = r'C:\\Users\\natal\Documents\\Data\\12-channel\\'

#img_id, slice = 'e15613s4_P52736', 100  #unrealistic best
img_id, slice = 'e14618s3_P51200', 87   # realistic best
# diff map one
img_id, slice = 'e13993s4_P16896', 175  # for diff maps
x_min = 117 # 117 for R=6,8 real. 60 for unreal
x_max = x_min + 25
y_min = 90  # 90 for R=6,8 real. 70 for unreal
y_max = y_min+25
# increased y goes up, increased x goes right

#img_id, slice = 'e15866s13_P72192', 59  # new worst

# realistic best and worst
#img_id, slice = 'e14091s3_P67584', 113  # best
#img_id, slice = 'e16031s3_P63488', 57  # worst

slice_ids = pd.read_csv(folder_path+'slice_ids_v2.csv')
slice_ids = slice_ids.loc[(slice_ids['patient_id']==img_id)  & (slice_ids['slice']==slice), :]
test_transforms = transforms.Compose(
    [
    ToTensor(),
    ReImgChannels()
    ]
)

target_transforms = transforms.Compose([ToTensor()])

style = 'circle_ring'
smap = 'circle_R_8'  # or '*'
smaps_170 = glob.glob(os.path.join('sensitivity_maps', '218_170', style, f'{smap}.npy'))
smaps_174 = glob.glob(os.path.join('sensitivity_maps', '218_174', style, f'{smap}.npy'))
smaps_180 = glob.glob(os.path.join('sensitivity_maps', '218_180', style, f'{smap}.npy'))
smaps = [smaps_170, smaps_174, smaps_180]

#masks_170 = glob.glob(os.path.join('undersampling_masks', '218_170', '*.npy'))
#masks_174 = glob.glob(os.path.join('undersampling_masks', '218_174', '*.npy'))
#masks_180 = glob.glob(os.path.join('undersampling_masks', '218_180', '*.npy'))
masks_170 = glob.glob(os.path.join('undersampling_masks', '218_170', '*R=6.npy'))
masks_174 = glob.glob(os.path.join('undersampling_masks', '218_174', '*R=6.npy'))
masks_180 = glob.glob(os.path.join('undersampling_masks', '218_180', '*R=6.npy'))

masks = [masks_170, masks_174, masks_180]

coils = 8
model_type = 'smap_branch'
version = f'{smap}_noise_final'
input_channels = coils * 2
reps = 5
block_depth = 5
filters = 80
smap_layers = 12
smap_filters = 50
model = CascadedModel(input_channels, reps=reps, block_depth=block_depth, filters=filters, 
                        smap_layers=smap_layers, smap_filters=smap_filters).type(torch.float32)
checkpoint = torch.load(os.path.join('select_model_weights', f'{model_type}_cascaded_model_v{version}.pt'))
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# make datasets
rand = 'neither'
smap_choice = 0  # order is 104, 103, 101, 9, 10, 11, 102, but doesn't matter cause we define things above in this script and specify the maps already
mask_choice = 0  # 4 and 8 are 0 and 2 respectively
test_data = SliceSmapTestDataset(slice_ids, 'test', smaps, masks, 'nlinv', coils, 
                                 data_transforms=test_transforms, target_transforms=target_transforms, 
                                 rand=rand, smap_choice=smap_choice, mask_choice=mask_choice)

test_loader = DataLoader(test_data, batch_size=1, shuffle=True, worker_init_fn=seed_worker, generator=g)


plt.figure(figsize=(24, 18))

for i, data in enumerate(test_loader):
    print(i)
    # load data
    input_img, label, smap = data[0].to(device, dtype=torch.float32), data[1].numpy(), data[2].numpy()
    scale_val, kspace, mask = data[3].to(dtype=torch.float32), data[4].to(device, dtype=torch.complex128), data[5].to(device, dtype=torch.float32)
    smap_path, mask_path, img_path = data[6], data[7], data[8]

    img_id = img_path[0].split(os.sep)[-1][:-4]
    r_factor = mask_path[0].split(os.sep)[-1].split('=')[-1].split('.')[0]
    smap_style = smap_path[0].split(os.sep)[-1].replace('.npy', '')

    # plot reference
    label = np.squeeze(label)
    inset_fig = plot_inset(np.abs(label), '', '', stats=False, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
    #label = np.squeeze(label)
    #plt.imshow(np.abs(label), cmap='gray')
    #plt.axis('off')
    inset_fig.savefig(f'plots/unrealistic_examples/{img_id}_label.png', dpi=300, bbox_inches='tight', transparent=True)
    plt.close()

    # test_model
    pred_img, pred_smap = model((input_img, kspace, mask))
    pred_img = pred_img.cpu().detach().numpy()
    pred_img = np.squeeze(to_complex(pred_img))

    #vmin, vmax = min(np.min(np.abs(label)), np.min(np.abs(pred_img))), max(np.max(np.abs(label)), np.max(np.abs(pred_img)))

    ssim_model = SSIM(np.abs(pred_img), np.abs(label))
    psnr_model = pSNR(np.abs(pred_img), np.abs(label))
    phase_model = phase_metric(np.abs(pred_img), np.abs(label))

    #plt.imshow(np.abs(pred_img), cmap='gray', vmin=vmin, vmax=vmax)
    #plt.axis('off')
    #plt.text(3, 10, f'SSIM: {ssim_model:.2f}', c='y', fontsize='14')
    #plt.text(3, 20, f'pSNR: {psnr_model:.1f}', c='y', fontsize='14')
    #plt.savefig(f'plots/unrealistic_examples/specific_models/{img_id}_accel_{r_factor}_{smap_style}_model_pred.png', dpi=300, bbox_inches='tight')
    #plt.close()

    # 108, 133
    inset_fig = plot_inset(np.abs(pred_img), ssim_model, psnr_model, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
    inset_fig.savefig(f'plots/realistic_examples/{img_id}_accel_{r_factor}_{smap_style}_model_pred.png', 
                      dpi=300, bbox_inches='tight', transparent=True)
    plt.close()

    print(f'\nmodel ssim: {ssim_model:.3f}')
    print(f'model psnr: {psnr_model:.1f}')
    print(f'model phase: {phase_model:3f}')

    # test traditional results
    kspace = kspace.cpu().numpy()
    kspace = np.moveaxis(kspace, 1, -1)  # was size 1, 4, 218, 170

    smap_og = np.squeeze(np.load(smap_path[0]))
    smap_og = smap_og[None, :, :, :]

    #pic = np.squeeze(bart.bart(1, 'pics', kspace, smap_og))
    pic = SenseRecon(np.squeeze(np.moveaxis(kspace, -1, 0)), np.squeeze(smap_og), max_iter=60)
    pic = pic.run()
    pic = pic / np.max(np.abs(pic))
    label = label / np.max(np.abs(label))

    ssim_trad = SSIM(np.abs(pic), np.abs(label))
    psnr_trad = pSNR(np.abs(pic), np.abs(label))
    phase_trad = phase_metric(pic, label)

    inset_fig = plot_inset(np.abs(pic), ssim_trad, psnr_trad, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
    inset_fig.savefig(f'plots/realistic_examples/{img_id}_accel_{r_factor}_{smap_style}_trad_pred.png', 
                      dpi=300, bbox_inches='tight', transparent=True)

    print(f'\ntrad ssim: {ssim_trad:.3f}')
    print(f'trad psnr: {psnr_trad:.1f}')
    print(f'trad phase: {phase_trad:3f}')




