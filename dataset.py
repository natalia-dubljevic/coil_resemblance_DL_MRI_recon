import numpy as np
import random
import torch
from torch.utils.data import Dataset
import bart


class ReImgChannels(object):
    def __call__(self, img):
        '''
        Convert a complex tensor into a 2 channel real/imaginary tensor
        Args:
            img (torch tensor): Compelx valued torch tensor
        '''
        c, h, w = img.shape
        img = torch.empty((c*2, h, w), dtype=torch.float64)
        re_img, im_img = torch.real(img), torch.imag(img)
        img[::2, :, :] = re_img
        img[1::2, :, :] = im_img

        return img


class SliceDataset(Dataset):
    def __init__(self, data_df, split, smaps, us_masks, target_type, coils, 
                 data_transforms=None, target_transforms=None, snr_factor=2) -> None:
        super().__init__()
        '''
        Dataset class for 2D slices 
        Args:
            data_df (DataFrame): Contains slice paths, patient ids, slice numbers, and splits
            split (str): One of train, val, test
            smaps (List): Contains paths to the various sensitivity maps
            us_masks (List): Contains paths to the various undersampling masks
            target_type (str): ESPiRIT or NLINV used for recosntructing the target
            coils (int): how many coils in this multi-coil image
            data_transforms(callable, optional): Optional composition of tranforms for the input data
            target_transforms(callable, optional): Optional composition of transforms for the target data
        '''
        self.file_paths = data_df.loc[data_df['split']==split, 'nlinv_path'].tolist() 
        self.smaps = smaps
        self.us_masks = us_masks
        self.data_transforms = data_transforms
        self.target_transforms = target_transforms
        self.coils = coils
        self.snr_factor = snr_factor

    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        img_path = self.file_paths[idx]  # recall this is the nicely done reconstruction
        smap_path = random.choice(self.smaps) 
        # if you're doing uniform, there's only one choice
        # if it's VDPD, randomly choose one of 50
        us_mask_path = random.choice(self.us_masks)  
        target_img = np.load(img_path)  # has size 1, 218, 170 (assumed to be pre-cropped to 218x170)
        smap = np.load(smap_path)  # size channels, h, w 
        mask = np.load(us_mask_path)
        mask = np.repeat(mask[None, :, :], self.coils, axis=0)

        noise = np.random.normal(0, self.snr_factor / 1000, target_img.shape) + \
           1j * np.random.normal(0, self.snr_factor / 1000, target_img.shape)
        input_kspace = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(target_img * smap + noise, axes=(-1, -2))), axes=(-1, -2)) * mask
        input_img = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(input_kspace, axes=(-1, -2))), axes=(-1, -2))   # want shape channel h w

        input_img =  np.moveaxis(input_img, 0, -1) # In numpy, want channels at end. Torch tensor transform will move them to the front
        target_img = np.moveaxis(target_img, 0, -1)
        smap = np.moveaxis(smap, 0, -1)
        input_kspace = torch.from_numpy(input_kspace)
        mask = torch.from_numpy(mask)

        if self.data_transforms:  # realistically, just ToTensor and then ReImgChannels
            input_img = self.data_transforms(input_img)
        if self.target_transforms:
            target_img = self.target_transforms(target_img)
            smap = self.target_transforms(smap)
        
        # scale by dividing all elements by the max value
        if input_img.dtype == torch.cdouble:
            input_max = torch.max(torch.abs(torch.view_as_real(input_img)))
        else:
            input_max = torch.max(torch.abs(input_img))

        input_img = torch.div(input_img, input_max)
        target_img = torch.div(target_img, input_max)
        input_kspace = torch.div(input_kspace, input_max)
        input_max = torch.reshape(input_max, (1, 1, 1))

        return input_img, target_img, smap, input_max, input_kspace, mask


class SliceTestDataset(Dataset):
    def __init__(self, data_df, split, smaps, us_masks, target_type, channels, 
                 data_transforms=None, target_transforms=None, smap_choice=None, 
                 mask_choice=None, snr_factor=2, crop=False) -> None:
        super().__init__()
        '''
        Dataset class for 2D test slices. Only difference is now we can choose to set a certain smap or us mask. Also, we find out afterwards
        what was chosen
        Args:
            data_df (DataFrame): Contains slice paths, patient ids, slice numbers, and splits
            split (str): One of train, val, test
            smaps (List): Contains paths to the various sensitivity maps
            us_masks (List): Contains paths to the various undersampling masks
            target_type (str): ESPiRIT or NLINV used for recosntructing the target
            data_transforms(callable, optional): Optional composition of tranforms for the input data
            target_transforms(callable, optional): Optional composition of transforms for the target data
        '''
        if target_type == 'nlinv':
            self.file_paths = data_df.loc[data_df['split']==split, 'nlinv_path'].tolist() 
        else:
            self.file_paths = data_df.loc[data_df['split']==split, 'espirit_path'].tolist() 
        self.smaps = smaps
        self.us_masks = us_masks
        self.data_transforms = data_transforms
        self.target_transforms = target_transforms
        self.channels = channels

        self.snr_factor = snr_factor
        self.crop = crop
        if self.crop:
            self.slice_paths = data_df.loc[data_df['split']==split, 'slice_path'].tolist() 

    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        target_img = np.load(img_path) 

        if self.crop:
            slice_path = self.slice_paths[idx]
            slice = np.load(slice_path)
            diff = int((slice.shape[1] - 170) / 2)
            slice = np.expand_dims(slice, 0)
            slice = bart.bart(1, 'fftmod 6', slice)
            slice = slice[:, :, diff : -diff, :]

            target_img = bart.bart(1, 'nlinv', slice)

        if target_img.shape[-1] == 170:
            smap_path = self.smaps[0]
            us_masks = self.us_masks[0]
        elif target_img.shape[-1] == 174:
            smap_path = self.smaps[1]
            us_masks = self.us_masks[1]
        else:
            smap_path = self.smaps[2]
            us_masks = self.us_masks[2]

        us_mask_path = random.choice(us_masks)
        smap = np.load(smap_path)  # size channels, h, w 
        
        mask = np.load(us_mask_path)
        mask = np.repeat(mask[None, :, :], self.channels, axis=0)

        noise = np.random.normal(0, self.snr_factor / 1000, target_img.shape) + 1j * np.random.normal(0, self.snr_factor / 1000, target_img.shape)
        input_kspace = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(target_img * smap + noise, axes=(-1, -2))), axes=(-1, -2)) * mask
        input_img = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(input_kspace, axes=(-1, -2))), axes=(-1, -2)) 

        input_img =  np.moveaxis(input_img, 0, -1)
        target_img = np.moveaxis(target_img, 0, -1)
        smap = np.moveaxis(smap, 0, -1)
        input_kspace = torch.from_numpy(input_kspace)
        mask = torch.from_numpy(mask)

        if self.data_transforms:
            input_img = self.data_transforms(input_img)
        if self.target_transforms:
            target_img = self.target_transforms(target_img)
            smap = self.target_transforms(smap)
        
        # scale by dividing all elements by the maximum absolute re/img value
        if input_img.dtype == torch.cdouble:
            input_max = torch.max(torch.abs(torch.view_as_real(input_img)))
        else:
            input_max = torch.max(torch.abs(input_img))

        input_img = torch.div(input_img, input_max)
        target_img = torch.div(target_img, input_max)
        input_kspace = torch.div(input_kspace, input_max)
        input_max = torch.reshape(input_max, (1, 1, 1))

        return input_img, target_img, smap, input_max, input_kspace, mask, (smap_path, us_mask_path, img_path)
        

