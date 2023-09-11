from datetime import datetime
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import sys
import time
import torch
from torch import nn
from torchmetrics import StructuralSimilarityIndexMeasure as SSIM
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.transforms import ToTensor
import wandb

### CHECK AND SEE WHAT MODEL YOU'RE IMPORTING!!!
from loss_functions import SSIMLoss
from metrics import SSIM as SSIM_numpy
from models.cascaded_map_branch import CascadedModel, to_complex  # CHECK WHAT IS BEING IMPORTED HERE
from dataset import SliceSmapDataset, ReImgChannels
from training_utils import EarlyStopper, wandb_scale_img

dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

# get system arguments
smap, coils, epochs = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
version, blocks, block_depth, filters= sys.argv[4], int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7])
smap_layers, smap_filters = int(sys.argv[8]), int(sys.argv[9])
batch_size, lr, stopper_patience, plateau_patience = int(sys.argv[10]), float(sys.argv[11]), int(sys.argv[12]), int(sys.argv[13])
clip, optim_name, weight_decay,  = float(sys.argv[14]), sys.argv[15], float(sys.argv[16])
smap_style, project, note, loss_fn = sys.argv[17], sys.argv[18], sys.argv[19], sys.argv[20]

#smap = "smap_20"
#epochs = 100
#coils = 4
#version = 'testy'
#blocks = 5
#block_depth = 5
#filters = 80
#smap_layers = 12
#smap_filters = 50
#batch_size = 4
#lr = 0.001
#stopper_patience = 10
#plateau_patience = 6
#clip = -1
#optim_name = 'ADAM'
#weight_decay = -1
#smap_style = ''
#note=  "local test"
#project='mri_reconstruction'
#loss_fn = 'MSE'

id = wandb.util.generate_id()
config = {
    "model_type": "smap_branch_cascaded",
    "version": version,
    "coils": coils,
    "epochs": epochs,
    "blocks": blocks,
    "block_depth": block_depth,
    "filters": filters,
    "smap_layers": smap_layers,
    "smap_filters": smap_filters,
    "batch_size": batch_size,
    "learning_rate": lr,
    "loss_function": loss_fn,
    "optimizer": optim_name,
    "weight_decay": weight_decay,
    "gradient_clipping": clip,
    "reduce_lr_patience": plateau_patience,
    "early_stopper_patience": stopper_patience,
    "date/time": dt_string,
    "run_id": id,
    "smap_style": smap_style,
    "version" : version
}
run_name = f"smap_branch_cascaded_{smap}"
run = wandb.init(project=project, id=id, name=run_name, config=config, notes=note)  # resume is True when resuming


# initiate some random seeds and check cuda
np.random.seed(0)
random.seed(0)
torch.manual_seed(0)

device='cuda:0' if torch.cuda.is_available() else 'cpu'
print(device, flush=True)

# load data and choose transforms
if sys.platform == 'linux' or sys.platform == 'linux2':
    if sys.argv[0].split('/')[2] == 'pasala':
        folder_path = '/home/pasala/Data/12-channel/'
    else:
        folder_path = '/home/natalia.dubljevic/Data/12-channel/'
else:
    folder_path = r'C:\\Users\\natal\Documents\\Data\\12-channel\\'

slice_ids = pd.read_csv(folder_path + 'slice_ids_v2.csv')
# remove this if you want nearly all slices!
slice_ids = slice_ids.loc[(slice_ids['slice'] >= 55) & (slice_ids['slice'] <= 201), :]
slice_ids_val = slice_ids.loc[(slice_ids['slice'] >= 77) & (slice_ids['slice'] < 178), :]
test_transforms = transforms.Compose(
    [
    ToTensor(),
    ReImgChannels()
    ]
)

# generate datasets
#smaps = glob.glob(f'sensitivity_maps/218_170/circle_ring/{smap}.npy')
smaps = glob.glob(f'sensitivity_maps/218_170/final_square/{smap}.npy')
#masks = glob.glob(r'undersampling_masks/218_170/*.npy')

# for nonuniform undersampling
#masks0 = glob.glob(r'gauss_undersampling_masks/218_170/gauss_mask_R=6_*.npy')
#masks1 = glob.glob(r'gauss_undersampling_masks/218_170/gauss_mask_R=8_*.npy')
#masks2 = glob.glob(r'gauss_undersampling_masks/218_170/gauss_mask_R=10_*.npy')
#
#masks = masks0 + masks1 + masks2

#masks = glob.glob(r'undersampling_masks/218_170/*R=[2468].npy')

# for uniform undersampling
masks = ['undersampling_masks/218_170/uniform_mask_R=6.npy',
         'undersampling_masks/218_170/uniform_mask_R=8.npy',
         'undersampling_masks/218_170/uniform_mask_R=10.npy']

masks = ['undersampling_masks/218_170/uniform_mask_R=2.npy',
         'undersampling_masks/218_170/uniform_mask_R=4.npy']

train_data = SliceSmapDataset(slice_ids, 'train', smaps, masks, 'nlinv', coils, data_transforms=test_transforms, target_transforms=test_transforms)
valid_data = SliceSmapDataset(slice_ids_val, 'val', smaps, masks, 'nlinv', coils, data_transforms=test_transforms, target_transforms=test_transforms)

if (sys.platform == 'linux' or platform == 'linux2') and sys.argv[0].split('/')[2] == 'pasala':
    train_data = Subset(train_data, list(range(50)))
    valid_data = Subset(valid_data, list(range(50)))

# create dataloaders
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=1, shuffle=True)

# define model
input_channels = coils * 2
model = CascadedModel(input_channels, reps=blocks, block_depth=block_depth, filters=filters, 
                      smap_layers=smap_layers, smap_filters=smap_filters).type(torch.float32)
model.to(device)
model_save_path = f'model_weights/smap_branch_cascaded_model_v{version}.pt'

# define hyperparameters
criterion_mse = nn.MSELoss() 
#criterion_ssim = SSIMLoss()
#criterion_l1 = nn.L1Loss()

if optim_name == 'ADAM':
    if weight_decay > -1:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
elif optim_name == 'SGD':
    if weight_decay > -1:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=plateau_patience)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 100], gamma=0.1)
early_stopper = EarlyStopper(patience=stopper_patience)


best_loss = 1e20
### TRAIN LOOP ###
print(f'Started training model version {version}', flush=True)
for epoch in range(epochs):
    train_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, img_labels, smap_labels =  data[0].to(device, dtype=torch.float32),  data[1].to(device, dtype=torch.float32), data[2].to(device, dtype=torch.float32)
        scale_values, input_kspaces, masks  = data[3].to(dtype=torch.float32), data[4].to(device, dtype=torch.complex64), data[5].to(device, dtype=torch.float32)

        optimizer.zero_grad()
        output_imgs, output_smaps = model((inputs, input_kspaces, masks))
        #loss_ssim = criterion_ssim(torch.abs(to_complex(output_imgs.detach().cpu())), torch.abs(to_complex(img_labels.detach().cpu())), data_range=torch.Tensor([1.0]))
        loss = criterion_mse(output_imgs, img_labels) + criterion_mse(output_smaps, smap_labels) #+ loss_ssim
        #loss = criterion_l1(output_imgs, img_labels) + criterion_l1(output_smaps, smap_labels) + loss_ssim 
        loss.backward()
        if clip > -1:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()
        train_loss += loss.item()

    train_loss /= (i + 1)
    print(f'{epoch + 1},  train loss: {train_loss:.6f}', flush=True)
    scheduler.step()
    
    val_loss = 0
    ### VALIDATION LOOP ###
    with torch.no_grad():
        preds = []
        labels = []
        smaps = []
        for i, data in enumerate(valid_loader, 0):
            input, img_label, smap_label =  data[0].to(device, dtype=torch.float32),  data[1].to(device, dtype=torch.float32), data[2].to(device, dtype=torch.float32)
            scale_value, input_kspace, mask  = data[3].to(dtype=torch.float32), data[4].to(device, dtype=torch.complex64), data[5].to(device, dtype=torch.float32)
            
            output_img, output_smap = model((input, input_kspace, mask))
            #loss_ssim = criterion_ssim(torch.abs(to_complex(output_img.detach().cpu())), torch.abs(to_complex(img_label.detach().cpu())), data_range=torch.Tensor([1.0]))
            loss = criterion_mse(output_img, img_label) + criterion_mse(output_smap, smap_label) #+ loss_ssim
            #loss = criterion_l1(output_img, img_label) + criterion_l1(output_smap, smap_label) + loss_ssim
            val_loss += loss.item()

            if i in range(4):
                mask = mask.detach().cpu().numpy()
                R = 1 / (np.sum(mask) / np.size(mask))
                img_pred = output_img.detach().cpu().numpy()
                img_label = img_label.detach().cpu().numpy()
                img_pred, img_label = np.abs(img_pred[0, 0, :, :] + 1j * img_pred[0, 1, :, :]), np.abs(img_label[0, 0, :, :] + 1j * img_label[0, 1, :, :])
                ssim = SSIM_numpy(img_pred, img_label)
                caption = f"R={R:.1f}, SSIM: {ssim:.3f}"
                smap_pred = output_smap.detach().cpu().numpy()
                smap_pred = np.abs(smap_pred[0, 0, :, :] + 1j * smap_pred[0, 1, :, :])

                preds.append(wandb.Image(wandb_scale_img(img_pred), caption=caption))
                labels.append(wandb.Image(wandb_scale_img(img_label)))
                smaps.append(wandb.Image(wandb_scale_img(smap_pred)))

        val_loss /= (i + 1)
        print(f'val loss: {val_loss:.6f}', flush=True)
        #scheduler.step(val_loss)


        wandb.log({"train_loss": train_loss, 
                    "val_loss": val_loss,
                    'pred': preds,
                    'pred_label': labels,
                    'smap': smaps}, step=epoch+1)
        
        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            print("Saving model", flush=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss
                }, model_save_path
            )

    if early_stopper.early_stop(val_loss):
        nepochs = epoch + 1            
        break

print('Finished Training! :D', flush=True)



