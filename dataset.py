import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import ToTensor
import numpy as np
import random
from scipy.ndimage import binary_fill_holes

# actually don't ened this if using torchcomplex package?
class ReImgChannels(object):
    def __call__(self, img):
        '''
        Convert a complex tensor into a 2 channel real/imaginary tensor
        Args:
            img (torch tensor): Compelx valued torch tensor
        '''
        c, h, w = img.shape
        empty_img = torch.empty((c*2, h, w), dtype=torch.float64)
        re_img, im_img = torch.real(img), torch.imag(img)
        empty_img[::2, :, :] = re_img
        empty_img[1::2, :, :] = im_img
        return empty_img


class SliceSmapDataset(Dataset):
    def __init__(self, data_df, split, smaps, us_masks, target_type, coils, data_transforms=None, target_transforms=None) -> None:
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
        if target_type == 'nlinv':
            self.file_paths = data_df.loc[data_df['split']==split, 'nlinv_path'].tolist() 
        else:
            self.file_paths = data_df.loc[data_df['split']==split, 'espirit_path'].tolist() 
        self.smaps = smaps
        #self.file_paths = random.sample(self.file_paths, 50)
        self.us_masks = us_masks
        self.data_transforms = data_transforms
        self.target_transforms = target_transforms
        self.coils = coils

    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        img_path = self.file_paths[idx]  # recall this is the nicely done reconstruction
        smap_path = random.choice(self.smaps) 
        us_mask_path = random.choice(self.us_masks)

        target_img = np.load(img_path)  # has size 1, 218, 170. 1 channel image, dims 218 x 170. 
        if target_img.shape[-1] != 170:
            diff = int((target_img.shape[-1] - 170) / 2)  # difference per side
            target_img = target_img[:, :, diff:-diff]
        smap = np.load(smap_path)  # size channels, h, w 
        mask = np.load(us_mask_path)
        mask = np.repeat(mask[None, :, :], self.coils, axis=0)

        #noise = np.random.normal(0, 2/1000, target_img.shape) + 1j * np.random.normal(0, 2/1000, target_img.shape)
        input_kspace = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(target_img * smap, axes=(-1, -2))), axes=(-1, -2)) * mask
        input_img = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(input_kspace, axes=(-1, -2))), axes=(-1, -2))   # want shape channel h w

        input_img =  np.moveaxis(input_img, 0, -1) # In numpy, want channels at end. Torch tensor transform will move them to the front
        target_img = np.moveaxis(target_img, 0, -1)
        smap = np.moveaxis(smap, 0, -1)
        input_kspace = torch.from_numpy(input_kspace)
        mask = torch.from_numpy(mask)

        if self.data_transforms:  # realistically, ToTensor and then ReImgChannels
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


class SliceSmapTestDataset(Dataset):
    def __init__(self, data_df, split, smaps, us_masks, target_type, channels, data_transforms=None, target_transforms=None, rand='smap', smap_choice=None, mask_choice=None) -> None:
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

        self.rand = rand
        self.smap_choice = smap_choice
        self.mask_choice = mask_choice

    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        img_path = self.file_paths[idx]  # recall this is the nicely done reconstruction
        target_img = np.load(img_path)  # has size 1, 218, 170. 1 channel image, dims 218 x 170. 
        if target_img.shape[-1] == 170:
            smaps = self.smaps[0]
            us_masks = self.us_masks[0]
        elif target_img.shape[-1] == 174:
            smaps = self.smaps[1]
            us_masks = self.us_masks[1]
        else:
            smaps = self.smaps[2]
            us_masks = self.us_masks[2]

        if self.rand == 'smap':
            smap_path = random.choice(smaps) 
            us_mask_path = us_masks[self.mask_choice]
        elif self.rand == 'smap_mask':
            smap_path = random.choice(smaps) 
            us_mask_path = random.choice(us_masks)
        elif self.rand == 'mask':
            smap_path = smaps[self.smap_choice]
            us_mask_path = random.choice(us_masks)
        else:  # pick it urself
            smap_path = smaps[self.smap_choice]
            us_mask_path = us_masks[self.mask_choice]
            
        smap = np.load(smap_path)  # size channels, h, w 
        # JUST FOR THE NEW CIRCLE SMAPS
        #smap = np.moveaxis(smap, -1, 0)

        mask = np.load(us_mask_path)
        mask = np.repeat(mask[None, :, :], self.channels, axis=0)

        noise = np.random.normal(0, 2/1000, target_img.shape) + 1j * np.random.normal(0, 2/1000, target_img.shape)
        input_kspace = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(target_img * smap + noise, axes=(-1, -2))), axes=(-1, -2)) * mask
        #input_kspace = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(target_img * smap, axes=(-1, -2))), axes=(-1, -2)) * mask
        input_img = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(input_kspace, axes=(-1, -2))), axes=(-1, -2)) 

        input_img =  np.moveaxis(input_img, 0, -1) # In numpy, want channels at end. Torch tensor transform will move them to the front
        target_img = np.moveaxis(target_img, 0, -1)
        smap = np.moveaxis(smap, 0, -1)
        input_kspace = torch.from_numpy(input_kspace)
        mask = torch.from_numpy(mask)

        if self.data_transforms:  # realistically, ToTensor and then ReImgChannels
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

        return input_img, target_img, smap, input_max, input_kspace, mask, smap_path, us_mask_path, img_path


class SliceSmapTestDatasetGfactor(Dataset):
    def __init__(self, data_df, split, smaps, us_masks, target_type, channels, 
                 data_transforms=None, target_transforms=None, rand='smap', 
                 smap_choice=None, mask_choice=None) -> None:
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

        self.rand = rand
        self.smap_choice = smap_choice
        self.mask_choice = mask_choice

    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        img_path = self.file_paths[idx]  # recall this is the nicely done reconstruction
        target_img = np.load(img_path)  # has size 1, 218, 170. 1 channel image, dims 218 x 170. 
        if target_img.shape[-1] == 170:
            smaps = self.smaps[0]
            us_masks = self.us_masks[0]
        elif target_img.shape[-1] == 174:
            smaps = self.smaps[1]
            us_masks = self.us_masks[1]
        else:
            smaps = self.smaps[2]
            us_masks = self.us_masks[2]

        if self.rand == 'smap':
            smap_path = random.choice(smaps) 
            us_mask_path = us_masks[self.mask_choice]
        elif self.rand == 'smap_mask':
            smap_path = random.choice(smaps) 
            us_mask_path = random.choice(us_masks)
        elif self.rand == 'mask':
            smap_path = smaps[self.smap_choice]
            us_mask_path = random.choice(us_masks)
        else:  # pick it urself
            smap_path = smaps[self.smap_choice]
            us_mask_path = us_masks[self.mask_choice]
            
        smap = np.load(smap_path)  # size channels, h, w 
        # JUST FOR THE NEW CIRCLE SMAPS
        #smap = np.moveaxis(smap, -1, 0)

        mask = np.load(us_mask_path)
        mask = np.repeat(mask[None, :, :], self.channels, axis=0)
        no_mask = np.ones(mask.shape)

        roi_mask = np.where(np.abs(np.squeeze(target_img)) < 0.25, 0, 1)
        roi_mask = binary_fill_holes(roi_mask)

        noise = (np.random.randn(*target_img.shape) + 1j * np.random.randn(*target_img.shape)) #* 0.25
        #target_img += noise  # CHANGE BACK

        input_kspace = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(target_img * smap + noise, axes=(-1, -2))), axes=(-1, -2)) * mask
        input_img = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(input_kspace, axes=(-1, -2))), axes=(-1, -2)) 

        input_kspace_nm = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(noise, axes=(-1, -2))), axes=(-1, -2)) * no_mask
        input_img_nm = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(input_kspace_nm, axes=(-1, -2))), axes=(-1, -2))

        input_img =  np.moveaxis(input_img, 0, -1) # In numpy, want channels at end. Torch tensor transform will move them to the front
        target_img = np.moveaxis(target_img, 0, -1)
        smap = np.moveaxis(smap, 0, -1)
        input_kspace = torch.from_numpy(input_kspace)
        mask = torch.from_numpy(mask)[None, :, :, :]

        input_img_nm =  np.moveaxis(input_img_nm, 0, -1) # In numpy, want channels at end. Torch tensor transform will move them to the front
        input_kspace_nm = torch.from_numpy(input_kspace_nm)
        no_mask = torch.from_numpy(no_mask)[None, :, :, :]

        if self.data_transforms:  # realistically, ToTensor and then ReImgChannels
            input_img = self.data_transforms(input_img)
            input_img_nm = self.data_transforms(input_img_nm)
        if self.target_transforms:
            target_img = self.target_transforms(target_img)
            smap = self.target_transforms(smap)
        
        # scale by dividing all elements by the max value
        if input_img.dtype == torch.cdouble:
            input_max = torch.max(torch.view_as_real(input_img))
            input_max_nm = torch.max(torch.view_as_real(input_img_nm))
        else:
            input_max = torch.max(torch.abs(input_img))
            input_max_nm = torch.max(torch.abs(input_img_nm))

        input_img = torch.div(input_img, input_max)[None, :, :, :]
        target_img = torch.div(target_img, input_max)
        input_kspace = torch.div(input_kspace, input_max)[None, :, :, :]
        input_max = torch.reshape(input_max, (1, 1, 1))

        input_img_nm = torch.div(input_img_nm, input_max_nm)[None, :, :, :]
        target_img_nm = torch.div(target_img, input_max_nm)
        input_kspace_nm = torch.div(input_kspace_nm, input_max_nm)[None, :, :, :]
        input_max_nm = torch.reshape(input_max_nm, (1, 1, 1))

        return (input_img, target_img, input_max, input_kspace, mask), (input_img_nm, target_img_nm, input_max_nm, input_kspace_nm, no_mask), roi_mask, smap_path, us_mask_path, img_path


if __name__=="__main__":
    import pandas as pd
    import matplotlib.pyplot as plt
    from sys import platform

    folder_path = '/home/pasala/Data/12-channel/'
    if platform == 'linux' or platform == 'linux2':
        folder_path = '/home/pasala/Data/12-channel/'
    else:
        folder_path = r'C:\\Users\\natal\Documents\\Data\\12-channel\\'
    slice_ids = pd.read_csv(folder_path+'slice_ids_v2.csv')
    test_transforms = transforms.Compose(
        [
        ToTensor(),
        ReImgChannels()
        ]
    )
    smap = ['test_map.npy']
    smap = ['sensitivity_maps/218_170/smap_10.npy']
    mask = ['test_mask.npy']
    channels = 4
    training_data = SliceSmapDataset(slice_ids, 'train', smap, mask, 'nlinv', channels, data_transforms=test_transforms, target_transforms=test_transforms)
    sample_indices = np.random.choice(len(training_data), size=20, replace=False)

    plt.figure(figsize = (24,18))
    for iter, ind in enumerate(sample_indices):
        img, label, img_max, ref_kspace, mask = training_data[ind]
        plt.subplot(4, 5, iter+1)
        complex_slice = label[0, :, :] + 1j * label[1, :, :]
        plt.imshow(np.abs(complex_slice))
    plt.savefig('plots/processed_data_labels.png')
        

