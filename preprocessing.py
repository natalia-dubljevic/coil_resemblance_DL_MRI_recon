import numpy as np
import pandas as pd
import h5py
import bart
import matplotlib.pyplot as plt
import os
import sys
import time
import random 
random.seed(0)

'''
Will do several things.
1. Determine which slices to keep. Based off of visuals, let's remove first and last 55
2. Develop ESPIRIT reference images (or to do on the fly?)
3. Calculate normalization parameters of volume (mean, std). Just want max value out of real and imag parts. Must do on fly...
'''
def main():
    try:
        # folder_path = r'C:\\Users\\natal\Documents\\Data\\12-channel\\'
        folder_path = '/home/pasala/Data/12-channel/'
        slice_ids = pd.read_csv(folder_path+'slice_ids_v2.csv')
        #slice_ids = slice_ids.loc[slice_ids['split'] == 'train', :]

        #not_170 = ['e15862s3_P33792', 'e16972s3_P31232', 'e15802s3_P42496', 
        #   'e15578s13_P08192','e15828s3_P57856', 'e16971s3_P23040',  
        #   'e16882s4_P38912']
        #
        #slice_ids = slice_ids[~slice_ids['patient_id'].isin(not_170)]
        #patients = np.array(slice_ids['patient_id'].unique().tolist())
        #sample = [31, 6, 19, 35, 18, 7, 1, 21, 34, 13, 32, 7, 28, 5, 24, 20, 36, 22, 9, 33]
        #train_patient_ids = list(patients[sample])

        train_patient_ids = slice_ids.loc[slice_ids['split'] == 'train', 'patient_id'].unique()
        val_patient_ids = slice_ids.loc[slice_ids['split'] == 'val', 'patient_id'].unique()
        test_patient_ids = slice_ids.loc[slice_ids['split'] == 'test', 'patient_id'].unique()
        #val_patient_ids = []
        #test_patient_ids = []

        slice_range = np.arange(54, 201)  # we have 256 slices total and are excluding first/last 55
        slice_range = list(np.arange(0, 256))
        #slice_range = list(np.concatenate((np.arange(0, 54), np.arange(201, 256))))

        for patient in train_patient_ids:
            for slice in slice_range:
                img_path = folder_path + r'train/target_slice_images/' + f'{patient}_s{slice}_nlinv_img.npy'
                if os.path.exists(img_path):
                    continue
                file_path = folder_path + f'train/slices/{patient}_s{slice}.npy'
                #file_path = slice_ids.loc[(slice_ids['patient_id']==patient) &
                #                          (slice_ids['slice']==slice), 'slice_path'].item()
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
                #file_path = slice_ids.loc[(slice_ids['patient_id']==patient) &
                #                        (slice_ids['slice']==slice), 'slice_path'].item()
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
                #file_path = slice_ids.loc[(slice_ids['patient_id']==patient) &
                #                          (slice_ids['split']=='test')&
                #                          (slice_ids['slice']==slice), 'slice_path'].item()
                kspace = np.load(file_path)
                kspace = np.expand_dims(kspace, 0)
                kspace_fftmod = bart.bart(1, 'fftmod 6', kspace)

                img = bart.bart(1, 'nlinv', kspace_fftmod)
                np.save(img_path, img)

                print(f'\n \n Patient {patient} slice {slice} done \n \n')

    except Exception as e:
        print(e)
        return main()

main()
add_to_csv = False
if add_to_csv:   
    #folder_path = r'C:\\Users\\natal\Documents\\Data\\12-channel\\'
    folder_path = '/home/pasala/Data/12-channel/'
    slice_ids = pd.read_csv(folder_path+'slice_ids.csv')
    slice_ids['espirit_path'] = folder_path + slice_ids['split'].astype(str) + r'/target_slice_images/' + \
                                slice_ids['patient_id'].astype(str) + '_s' + slice_ids['slice'].astype(str) + '_img.npy'
    slice_ids['nlinv_path'] = folder_path + slice_ids['split'].astype(str) + r'/target_slice_images/' + \
                                slice_ids['patient_id'].astype(str) + '_s' + slice_ids['slice'].astype(str) + '_nlinv_img.npy'
    slice_ids = slice_ids.loc[(slice_ids['slice'] >= 54 ) & (slice_ids['slice'] < 201 ), :]
    slice_ids.to_csv('/home/pasala/Data/12-channel/slice_ids_v3.csv', index=False)


