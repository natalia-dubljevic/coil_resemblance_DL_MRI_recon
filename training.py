import glob
import numpy as np
import pandas as pd
import random
import sys
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose

from cascaded_map_branch import CascadedModel
from dataset import SliceDataset, ReImgChannels
from training_utils import EarlyStopper


# get system arguments
smap, smap_style, coils, epochs = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
version, blocks, block_depth, filters= sys.argv[5], int(sys.argv[6]), int(sys.argv[7]), int(sys.argv[8])
smap_layers, smap_filters = int(sys.argv[9]), int(sys.argv[10])
batch_size, lr, stopper_patience = int(sys.argv[11]), float(sys.argv[12]), int(sys.argv[13])

# initiate some random seeds and check cuda
np.random.seed(0)
random.seed(0)
torch.manual_seed(0)

device='cuda:0' if torch.cuda.is_available() else 'cpu'
print(device, flush=True)

# load data and choose transforms
folder_path = '/home/pasala/Data/12-channel/'

slice_ids = pd.read_csv(folder_path + 'slice_ids_v2.csv')
slice_ids = slice_ids.loc[((slice_ids['slice'] >= 55) 
                            & (slice_ids['slice'] <= 201)), :]  # remove first and last 55 for training
slice_ids_val = slice_ids.loc[((slice_ids['slice'] >= 77) 
                               & (slice_ids['slice'] < 178)), :] # keep only central 100 for validation + testing
test_transforms = Compose([ToTensor(), ReImgChannels()])

# generate datasets
smaps = glob.glob(f'sensitivity_maps/218_170/{smap_style}/{smap}.npy')

if smap_style == 'circle_ring':
    masks = ['undersampling_masks/218_170/uniform_mask_R=6.npy',
            'undersampling_masks/218_170/uniform_mask_R=8.npy']
else:
    masks = ['undersampling_masks/218_170/uniform_mask_R=2.npy',
            'undersampling_masks/218_170/uniform_mask_R=4.npy']

train_data = SliceDataset(slice_ids, 'train', smaps, masks, 'nlinv', coils, data_transforms=test_transforms, target_transforms=test_transforms)
valid_data = SliceDataset(slice_ids_val, 'val', smaps, masks, 'nlinv', coils, data_transforms=test_transforms, target_transforms=test_transforms)

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
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
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
        loss = criterion_mse(output_imgs, img_labels) + criterion_mse(output_smaps, smap_labels)

        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= (i + 1)
    print(f'{epoch + 1},  train loss: {train_loss:.6f}', flush=True)
    scheduler.step()
    
    ### VALIDATION LOOP ###
    val_loss = 0.0
    with torch.no_grad():
        preds = []
        labels = []
        smaps = []
        for i, data in enumerate(valid_loader, 0):
            input, img_label, smap_label =  data[0].to(device, dtype=torch.float32),  data[1].to(device, dtype=torch.float32), data[2].to(device, dtype=torch.float32)
            scale_value, input_kspace, mask  = data[3].to(dtype=torch.float32), data[4].to(device, dtype=torch.complex64), data[5].to(device, dtype=torch.float32)
            
            output_img, output_smap = model((input, input_kspace, mask))
            loss = criterion_mse(output_img, img_label) + criterion_mse(output_smap, smap_label)
            val_loss += loss.item()

        val_loss /= (i + 1)
        print(f'val loss: {val_loss:.6f}', flush=True)
        
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
        break

print('Finished Training! :)', flush=True)



