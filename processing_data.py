import bart
import csv
import glob
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random 


'''
Calling these two functions consecutively will break the h5 volumes into .np
slices, save those and a csv of slice ids + paths, and then reconstruct each
slice using NLINV.
'''

def vol_to_slices(data_path='C:\\Users\\natal\\Documents\\Data\\12-channel'):

    train_path = os.path.join(data_path, r"Train\*")
    val_path = os.path.join(data_path, r"Val\*")
    test_path = os.path.join(data_path, r"Test\*")

    train_files = list(glob.glob(train_path))
    val_files =list(glob.glob(val_path))
    test_files = list(glob.glob(test_path))

    # start the csv
    f = open('C:\\Users\\natal\\Documents\\slice_ids.csv', 'w')
    writer = csv.writer(f)
    writer.writerow(['split', 'patient_id', 'slice', 'slice_path'])

    for file in train_files:
        split = "train"
        patient_id = file.split('\\')[-1].split('.')[0]
        mod_train_path = train_path.split("*")[0] + 'slices'
        sample_kspace = h5py.File(file, 'r')['kspace'][:]
        for i in range(sample_kspace.shape[0]):
            new_name = "{}_s{}.npy".format(patient_id, i)
            new_path = os.path.join(mod_train_path, new_name)

            slice = sample_kspace[i, :, :, :]
            slice = slice[:,:,::2] + 1j*slice[:,:,1::2]  # convert to complex
            np.save(new_path, slice)
            writer.writerow([split, patient_id, i, new_path])

    for file in val_files:
        split = "val"
        patient_id = file.split('\\')[-1].split('.')[0]
        mod_train_path = val_path.split("*")[0] + 'slices'
        sample_kspace = h5py.File(file, 'r')['kspace'][:]
        for i in range(sample_kspace.shape[0]):
            new_name = "{}_s{}.npy".format(patient_id, i)
            new_path = os.path.join(mod_train_path, new_name)

            slice = sample_kspace[i, :, :, :]
            slice = slice[:,:,::2] + 1j*slice[:,:,1::2]

            np.save(new_path, slice)
            writer.writerow([split, patient_id, i, new_path])

    for file in test_files:
        split = "test"
        patient_id = file.split('\\')[-1].split('.')[0]
        mod_train_path = test_path.split("*")[0] + 'slices'
        sample_kspace = h5py.File(file, 'r')['kspace'][:]
        for i in range(sample_kspace.shape[0]):
            new_name = "{}_s{}.npy".format(patient_id, i)
            new_path = os.path.join(mod_train_path, new_name)

            slice = sample_kspace[i, :, :, :]
            slice = slice[:,:,::2] + 1j*slice[:,:,1::2]

            np.save(new_path, slice)
            writer.writerow([split, patient_id, i, new_path])

    f.close()        
    return



def slices_to_NLINV(folder_path='/home/pasala/Data/12-channel/'):
    slice_ids = pd.read_csv(folder_path +'slice_ids.csv')

    train_patient_ids = slice_ids.loc[slice_ids['split'] == 'train', 'patient_id'].unique()
    val_patient_ids = slice_ids.loc[slice_ids['split'] == 'val', 'patient_id'].unique()
    test_patient_ids = slice_ids.loc[slice_ids['split'] == 'test', 'patient_id'].unique()

    slice_range = list(np.arange(0, 256))

    for patient in train_patient_ids:
        for slice in slice_range:
            img_path = folder_path + r'train/target_slice_images/' + f'{patient}_s{slice}_nlinv_img.npy'
            if os.path.exists(img_path):
                continue

            file_path = folder_path + f'train/slices/{patient}_s{slice}.npy'
            kspace = np.load(file_path)
            kspace = np.expand_dims(kspace, 0)
            kspace_fftmod = bart.bart(1, 'fftmod 6', kspace)

            img = bart.bart(1, 'nlinv', kspace_fftmod)
            np.save(img_path, img)

            print(f'\n \n Patient {patient} slice {slice} done \n \n')   

    for patient in val_patient_ids:
        for slice in slice_range:
            img_path = folder_path + r'val/target_slice_images/' + f'{patient}_s{slice}_nlinv_img.npy'
            if os.path.exists(img_path):
                continue

            file_path = folder_path + f'val/slices/{patient}_s{slice}.npy'
            kspace = np.load(file_path)
            kspace = np.expand_dims(kspace, 0)
            kspace_fftmod = bart.bart(1, 'fftmod 6', kspace)

            img = bart.bart(1, 'nlinv', kspace_fftmod)
            np.save(img_path, img)

            print(f'\n \n Patient {patient} slice {slice} done \n \n')

    for patient in test_patient_ids:
        for slice in slice_range:
            img_path = folder_path + r'test/target_slice_images/' + f'{patient}_s{slice}_nlinv_img.npy'
            if os.path.exists(img_path):
                continue

            file_path = folder_path + f'test/slices/{patient}_s{slice}.npy'
            kspace = np.load(file_path)
            kspace = np.expand_dims(kspace, 0)
            kspace_fftmod = bart.bart(1, 'fftmod 6', kspace)

            img = bart.bart(1, 'nlinv', kspace_fftmod)
            np.save(img_path, img)

            print(f'\n \n Patient {patient} slice {slice} done \n \n')
    
    return


vol_to_slices()
slices_to_NLINV()

add_to_csv = True
if add_to_csv:   
    folder_path = '/home/pasala/Data/12-channel/'
    slice_ids = pd.read_csv(folder_path + 'slice_ids.csv')
    slice_ids['nlinv_path'] = folder_path + slice_ids['split'].astype(str) + r'/target_slice_images/' + \
                                slice_ids['patient_id'].astype(str) + '_s' + slice_ids['slice'].astype(str) + '_nlinv_img.npy'
    slice_ids = slice_ids.loc[(slice_ids['slice'] >= 54 ) & (slice_ids['slice'] < 201 ), :]
    slice_ids.to_csv('/home/pasala/Data/12-channel/slice_ids.csv', index=False)


