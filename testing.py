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

from cascaded_model import CascadedModel, to_complex
from dataset import SliceTestDataset, ReImgChannels
from metrics import SSIM, PSNR, phase_metric
from sigpy.mri.app import SenseRecon, L1WaveletRecon


"""
Example arguments:

coils = 8
R = 6
smap_style = '8ch_head'

smap = 'circle_R_9'
test_model = 1

model_type = 'cascaded'
version = 'circle_R_9_final'

blocks = 5
block_depth = 5
filters = 80

lam = 1e-4
iters = 100

us_style = 'vdpd'
trad_model = 'CS'
snr_factor = 2
"""

# get system arguments
coils, R, smap_style, smap = (
    int(sys.argv[1]),
    int(sys.argv[2]),
    sys.argv[3],
    sys.argv[4],
)
test_model = int(sys.argv[5])

# model arguments
model_type, version = (
    sys.argv[6],
    sys.argv[7],
)
blocks, block_depth, filters = int(sys.argv[8]), int(sys.argv[9]), int(sys.argv[10])

lam, iters = float(sys.argv[11]), int(sys.argv[12])
us_style, trad_model, snr_factor = sys.argv[13], sys.argv[14], float(sys.argv[15])

snr = int(200 / snr_factor)

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

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# load data and choose transforms
folder_path = "/home/pasala/Data/12-channel/"

slice_ids = pd.read_csv(folder_path + "slice_ids_v2.csv")
slice_ids = slice_ids.loc[
    ((slice_ids["slice"] >= 77) & (slice_ids["slice"] < 178)), :
]  # keep only central 100 for testing

test_transforms = Compose([ToTensor(), ReImgChannels()])
target_transforms = Compose([ToTensor()])

smaps_170 = f"sensitivity_maps/218_170/{smap_style}/{smap}.npy"
smaps_174 = f"sensitivity_maps/218_174/{smap_style}/{smap}.npy"
smaps_180 = f"sensitivity_maps/218_180/{smap_style}/{smap}.npy"
smaps = [smaps_170, smaps_174, smaps_180]


if us_style == "vdpd":
    masks_170 = glob.glob(f"vdpd_undersampling_masks/218_170/vdpd_mask_R={R}*.npy")
    masks_174 = glob.glob(f"vdpd_undersampling_masks/218_174/vdpd_mask_R={R}*.npy")
    masks_180 = glob.glob(f"vdpd_undersampling_masks/218_180/vdpd_mask_R={R}*.npy")
else:
    masks_170 = [f"undersampling_masks/218_170/uniform_mask_R={R}.npy"]
    masks_174 = [f"undersampling_masks/218_174/uniform_mask_R={R}.npy"]
    masks_180 = [f"undersampling_masks/218_180/uniform_mask_R={R}.npy"]

masks = [masks_170, masks_174, masks_180]

if test_model:
    model = CascadedModel(
        coils * 2, reps=blocks, block_depth=block_depth, filters=filters
    ).type(torch.float32)

    checkpoint = torch.load(
        os.path.join("model_weights", f"{us_style}_{model_type}_model_v{version}.pt")
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

# make datasets
test_data = SliceTestDataset(
    slice_ids,
    "test",
    smaps,
    masks,
    "nlinv",
    coils,
    data_transforms=test_transforms,
    target_transforms=target_transforms,
    snr_factor=snr_factor,
)
test_loader = DataLoader(
    test_data, batch_size=1, shuffle=True, worker_init_fn=seed_worker, generator=g
)

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

# set up some figures so we can see the first few reconstructions
fig_img = plt.figure(figsize=(24, 18))

for i, data in enumerate(test_loader):
    input_img, img_label, smap_label = (
        data[0].to(device, dtype=torch.float32),
        data[1].numpy(),
        data[2].to(device, dtype=torch.float32),
    )
    scale_val, kspace, mask = (
        data[3].to(dtype=torch.float32),
        data[4].to(device, dtype=torch.complex64),
        data[5].to(device, dtype=torch.float32),
    )
    smap_path, mask_path, img_path = data[6]

    img_id.append(img_path[0].split(os.sep)[-1][:-4])

    img_label = np.squeeze(img_label)
    smap_label = np.squeeze(smap_label)

    if test_model:  # if we want to test the DL model
        pred_img = model((input_img, kspace, mask, smap_label))
        pred_img = pred_img.detach().cpu().numpy()
        pred_img = np.squeeze(to_complex(pred_img))

        pred_img /= np.max(np.abs(pred_img))
        img_label /= np.max(np.abs(img_label))

        # get DL model metrics
        ssim = SSIM(np.abs(pred_img), np.abs(img_label))
        psnr = PSNR(np.abs(pred_img), np.abs(img_label))
        phase = phase_metric(pred_img, img_label)

        ssim_model.append(ssim)
        psnr_model.append(psnr)
        phase_model.append(phase)

        if i in np.arange(0, 10):
            # plot some reconstruction/reference pairs
            ax = fig_img.add_subplot(5, 4, (i + 1) * 2 - 1)
            ax.imshow(np.abs(pred_img))
            ax.text(
                3, pred_img.shape[0] - 13, f"SSIM: {ssim:.2f}", c="y", fontsize="14"
            )
            ax.text(
                3, pred_img.shape[0] - 25, f"PSNR: {psnr:.2f}", c="y", fontsize="14"
            )
            ax.set_title(f"Prediction")

            ax = fig_img.add_subplot(5, 4, (i + 1) * 2)
            ax.imshow(np.abs(img_label))
            ax.set_title(f"Label")
            fig_img.savefig(
                f"results/model_{us_style}_{model_type}_v{version}_results_accel_{R}_{smap}_snr_{snr}.png"
            )

    else:  # otherwise, we are doing a CG-SENSE or CS reconstruction
        kspace = kspace.cpu().numpy()
        kspace = np.moveaxis(kspace, 1, -1)

        # using the original sensitivity maps
        smap_og = np.squeeze(np.load(smap_path[0]))
        smap_og = smap_og[None, :, :, :]

        if trad_model == "CS":
            rec_img_app = L1WaveletRecon(
                np.squeeze(np.moveaxis(kspace, -1, 0)),
                np.squeeze(smap_og),
                max_iter=iters,
                lamda=lam,
                show_pbar=False,
            )
        else:
            rec_img_app = SenseRecon(
                np.squeeze(np.moveaxis(kspace, -1, 0)),
                np.squeeze(smap_og),
                max_iter=iters,
                lamda=lam,
                show_pbar=False,
            )

        rec_img = rec_img_app.run()

        # scale img/label prior to calculaitng metrics for CG-SENSE since its
        # reconstructed image scale is VERY different from label
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
            ax.text(3, rec_img.shape[0] - 13, f"SSIM: {ssim:.2f}", c="y", fontsize="14")
            ax.text(3, rec_img.shape[0] - 25, f"PSNR: {psnr:.2f}", c="y", fontsize="14")
            ax.set_title(f"Prediction")

            ax = fig_img.add_subplot(5, 4, (i + 1) * 2)
            ax.imshow(np.abs(img_label))
            ax.set_title(f"Label")
            plt.savefig(f"results/{trad_model}_results_accel_{R}_{smap}_snr_{snr}.png")

if test_model:
    results = pd.DataFrame(
        {
            "img_id": img_id,
            "R_factor": R,
            "smap_style": smap,
            "ssim_model": ssim_model,
            "psnr_model": psnr_model,
            "phase_model": phase_model,
        }
    )
    results.to_csv(
        f"results/model_results_{us_style}_{model_type}_v{version}_accel_{R}_{smap}_snr_{snr}.csv",
        index=False,
    )

else:
    results = pd.DataFrame(
        {
            "img_id": img_id,
            "R_factor": R,
            "smap_style": smap,
            "ssim_trad": ssim_trad,
            "psnr_trad": psnr_trad,
            "phase_trad": phase_trad,
        }
    )
    results.to_csv(
        f"results/{trad_model}_results_accel_{R}_{smap}_snr_{snr}.csv", index=False
    )
