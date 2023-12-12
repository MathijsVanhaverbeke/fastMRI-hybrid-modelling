## Set up a resource limiter, such that the script doesn't take up more than a certain amount of RAM (normally 40GB is the limit). In that case, an error will be thrown

import resource

resource.setrlimit(resource.RLIMIT_AS, (40_000_000_000, 40_000_000_000))


print('Resource limit set. Importing libraries...')


## Import libraries

import h5py, os
import numpy as np
from numpy import fft 
import matplotlib.pyplot as plt
from utils import estimate_mdgrappa_kernel, calculate_mask, comp_sub_kspace, comp_img
from pathlib import Path
from itertools import chain
import gc
import time
import pickle


print('Libraries imported. Starting to prepare to load the dataset...')


## Prepare dataset

download_path = '/usr/local/micapollo01/MIC/DATA/SHARED/NYU_FastMRI'
dicom_path = os.path.join(download_path,'fastMRI_brain_DICOM')
train_path = os.path.join(download_path,'multicoil_train')
validation_path = os.path.join(download_path,'multicoil_val')
test_path = os.path.join(download_path,'multicoil_test')
fully_sampled_test_path = os.path.join(download_path,'multicoil_test_full')

training_files = Path(train_path).glob('**/*')
validation_files = Path(validation_path).glob('**/*')
test_files = Path(test_path).glob('**/*')
fully_sampled_test_files = Path(fully_sampled_test_path).glob('**/*')
all_files = chain(training_files, validation_files, test_files, fully_sampled_test_files)

clustered_data_2 = np.load("/usr/local/micapollo01/MIC/DATA/STUDENTS/mvhave7/Results/Preprocessing/exploration/16coil_slice_size_clustered_fastmri_data.npy", allow_pickle=True)
clustered_data_2 = clustered_data_2.item()

files_16_640_320 = clustered_data_2[(640,320)]
training_files = sorted(files_16_640_320)

crop_size = (32,640,320)


print('All variables are loaded. Start preprocessing data in batches...')



batch_number = 1
batch_size = 1
# Note: batch_size here has unit 'number of files', not 'number of slices'



## Loop across training data to preprocess in batches

num_batches = len(training_files) // batch_size

for batch in range(num_batches):

    batch_files = training_files[batch*batch_size:(batch+1)*batch_size]

    print('Preprocessing batch '+str(batch_number)+'...')


    ## Create training data pairs and estimate GRAPPA kernels

    print('Starting dataset construction and GRAPPA kernel estimation...')

    cnt = 1
    last_mask = None
    X_train = []
    Y_train = []
    grappa_wt = []
    grappa_p = []

    for mri_f in batch_files:
        filename = os.path.basename(mri_f)
        filename = filename.replace(".h5","")
        with h5py.File(mri_f,'r') as f:

            k = f['kspace'][()]
            sequence = f.attrs['acquisition']
            nSL, nCh, nFE, nPE = k.shape
            
            # Create the subsampled training data. In the GrappaNet paper, they say they performed experiments for R=4 and R=8, but they never mention the number or fraction of ACS lines used...
            mask = calculate_mask(nFE,nPE,0.08,4)
            last_mask = mask
            subsampled_k = k*mask
        
            for slice in range(nSL): 
                target_img = np.zeros((nCh,nFE,nPE),dtype=np.float)
                sub_kspace = np.zeros((nCh*2,nFE,nPE),dtype=np.float)
                wt,ps = estimate_mdgrappa_kernel(kspace=subsampled_k[slice,:,:,:],calib=None,kernel_size=(5,5),coil_axis=0) 
                for iCh in range(nCh):
                        target_img[iCh,:,:] = abs(fft.fftshift(fft.ifft2(k[slice,iCh,:,:])))
                        sub_kspace[iCh,:,:] = subsampled_k[slice,iCh,:,:].real
                        sub_kspace[iCh+nCh,:,:] = subsampled_k[slice,iCh,:,:].imag
                X_train.append(list(comp_sub_kspace(sub_kspace,crop_size)))
                Y_train.append(list(comp_img(target_img,(crop_size[0]//2,crop_size[1],crop_size[2]))))
                grappa_wt.append(wt)
                grappa_p.append(ps)
            
            print(cnt,filename,sequence,nSL,nCh,nFE,nPE,sub_kspace.shape)
            cnt += 1

    Y_train = np.array(Y_train).astype(np.float32)
    X_train = np.array(X_train).astype(np.float32)


    print('Done. Calculating RSS images as references for the loss function...')


    ## Calculate RSS images that will be used as references for the loss function

    X_train = np.transpose(X_train,(0,2,3,1))
    Y_rss = np.sqrt(np.sum(np.square(Y_train),axis=1))
    Y_rss = Y_rss.astype(np.float32)
    print(X_train.shape,Y_rss.shape)


    print('Done. Normalizing the data...')


    ## Normalize the data

    dims = X_train.shape
    for i in range(dims[0]):
        for j in range(dims[3]):
            X_train[i,:,:,j] = X_train[i,:,:,j]/((np.max(X_train[i,:,:,j])-np.min(X_train[i,:,:,j]))+1e-10)

    for i in range(dims[0]):
        Y_rss[i,:,:] = Y_rss[i,:,:]/((np.max(Y_rss[i,:,:])-np.min(Y_rss[i,:,:]))+1e-10)


    print('Done. Performing a datasplit...')


    ## Create a dataset split 90-10 training-validation

    x_train = X_train[0:int(X_train.shape[0]-X_train.shape[0]*0.1),:,:,:]
    y_train = Y_rss[0:int(X_train.shape[0]-X_train.shape[0]*0.1),:,:]
    x_test = X_train[int(X_train.shape[0]-X_train.shape[0]*0.1):,:,:,:]
    y_test = Y_rss[int(X_train.shape[0]-X_train.shape[0]*0.1):,:,:]
    y_test = np.reshape(y_test, (y_test.shape[0],crop_size[1],crop_size[2]))
    grappa_train_indx = np.array(range(0,int(X_train.shape[0]-X_train.shape[0]*0.1)),dtype=int)
    grappa_test_indx = np.array(range(int(X_train.shape[0]-X_train.shape[0]*0.1),X_train.shape[0]),dtype=int)


    print('Done. Saving results...')


    ## Save the results

    path_to_save_mri_data = '/usr/local/micapollo01/MIC/DATA/STUDENTS/mvhave7/Results/Preprocessing/mri/'
    path_to_save_grappa_data = '/usr/local/micapollo01/MIC/DATA/STUDENTS/mvhave7/Results/Preprocessing/grappa/'

    np.save(path_to_save_mri_data+"training_data_GrappaNet_16_coils_batch_{}.npy".format(batch_number), x_train)
    np.save(path_to_save_mri_data+"training_data_GT_GrappaNet_16_coils_batch_{}.npy".format(batch_number), y_train)
    np.save(path_to_save_mri_data+"validation_data_GrappaNet_16_coils_batch_{}.npy".format(batch_number), x_test)
    np.save(path_to_save_mri_data+"validation_data_GT_GrappaNet_16_coils_batch_{}.npy".format(batch_number), y_test)

    np.save(path_to_save_grappa_data+"grappa_train_indx_GrappaNet_16_coils_batch_{}.npy".format(batch_number), grappa_train_indx)
    np.save(path_to_save_grappa_data+"grappa_validation_indx_GrappaNet_16_coils_batch_{}.npy".format(batch_number), grappa_test_indx)

    with open(path_to_save_grappa_data+'grappa_wt_batch_{}.pickle'.format(batch_number), 'wb') as handle:
        pickle.dump(grappa_wt, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(path_to_save_grappa_data+'grappa_p_batch_{}.pickle'.format(batch_number), 'wb') as handle:
        pickle.dump(grappa_p, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

    print('Done. Clearing memory and preparing to load in a new batch...')


    ## Free up RAM memory

    time.sleep(1)
    del batch_files
    del cnt, last_mask
    del filename, k, sequence, nSL, nCh, nFE, nPE
    del mask, subsampled_k, target_img, sub_kspace, wt, ps
    del X_train, Y_train, grappa_wt, grappa_p, Y_rss, dims
    del x_train, y_train, x_test, y_test, grappa_train_indx, grappa_test_indx
    time.sleep(1)

    gc.collect()


    print('Done.')
    batch_number += 1



## Process remaining files in last batch

print('Preprocessing the last batch, batch '+str(batch_number)+'...')

if batch_size == 1:
    remaining_files = training_files[-1]
elif batch_size > 1:
    remaining_files = training_files[num_batches*batch_size:]


## Create training data pairs and estimate GRAPPA kernels

print('Starting dataset construction and GRAPPA kernel estimation...')

cnt = 1
last_mask = None
X_train = []
Y_train = []
grappa_wt = []
grappa_p = []

for mri_f in remaining_files:
    filename = os.path.basename(mri_f)
    filename = filename.replace(".h5","")
    with h5py.File(mri_f,'r') as f:

        k = f['kspace'][()]
        sequence = f.attrs['acquisition']
        nSL, nCh, nFE, nPE = k.shape
            
        # Create the subsampled training data. In the GrappaNet paper, they say they performed experiments for R=4 and R=8, but they never mention the number or fraction of ACS lines used...
        mask = calculate_mask(nFE,nPE,0.08,4)
        last_mask = mask
        subsampled_k = k*mask
        
        for slice in range(nSL): 
            target_img = np.zeros((nCh,nFE,nPE),dtype=np.float)
            sub_kspace = np.zeros((nCh*2,nFE,nPE),dtype=np.float)
            wt,ps = estimate_mdgrappa_kernel(kspace=subsampled_k[slice,:,:,:],calib=None,kernel_size=(5,5),coil_axis=0) 
            for iCh in range(nCh):
                    target_img[iCh,:,:] = abs(fft.fftshift(fft.ifft2(k[slice,iCh,:,:])))
                    sub_kspace[iCh,:,:] = subsampled_k[slice,iCh,:,:].real
                    sub_kspace[iCh+nCh,:,:] = subsampled_k[slice,iCh,:,:].imag
            X_train.append(list(comp_sub_kspace(sub_kspace,crop_size)))
            Y_train.append(list(comp_img(target_img,(crop_size[0]//2,crop_size[1],crop_size[2]))))
            grappa_wt.append(wt)
            grappa_p.append(ps)
          
        print(cnt,filename,sequence,nSL,nCh,nFE,nPE,sub_kspace.shape)
        cnt += 1

Y_train = np.array(Y_train).astype(np.float32)
X_train = np.array(X_train).astype(np.float32)


print('Done. Calculating RSS images as references for the loss function...')


## Calculate RSS images that will be used as references for the loss function

X_train = np.transpose(X_train,(0,2,3,1))
Y_rss = np.sqrt(np.sum(np.square(Y_train),axis=1))
Y_rss = Y_rss.astype(np.float32)
print(X_train.shape,Y_rss.shape)


print('Done. Normalizing the data...')


## Normalize the data

dims = X_train.shape
for i in range(dims[0]):
    for j in range(dims[3]):
        X_train[i,:,:,j] = X_train[i,:,:,j]/((np.max(X_train[i,:,:,j])-np.min(X_train[i,:,:,j]))+1e-10)

for i in range(dims[0]):
    Y_rss[i,:,:] = Y_rss[i,:,:]/((np.max(Y_rss[i,:,:])-np.min(Y_rss[i,:,:]))+1e-10)


print('Done. Performing a datasplit...')


## Create a dataset split 90-10 training-validation

x_train = X_train[0:int(X_train.shape[0]-X_train.shape[0]*0.1),:,:,:]
y_train = Y_rss[0:int(X_train.shape[0]-X_train.shape[0]*0.1),:,:]
x_test = X_train[int(X_train.shape[0]-X_train.shape[0]*0.1):,:,:,:]
y_test = Y_rss[int(X_train.shape[0]-X_train.shape[0]*0.1):,:,:]
y_test = np.reshape(y_test, (y_test.shape[0],crop_size[1],crop_size[2]))
grappa_train_indx = np.array(range(0,int(X_train.shape[0]-X_train.shape[0]*0.1)),dtype=int)
grappa_test_indx = np.array(range(int(X_train.shape[0]-X_train.shape[0]*0.1),X_train.shape[0]),dtype=int)


print('Done. Saving results...')


## Save the results

path_to_save_mri_data = '/usr/local/micapollo01/MIC/DATA/STUDENTS/mvhave7/Results/Preprocessing/mri/'
path_to_save_grappa_data = '/usr/local/micapollo01/MIC/DATA/STUDENTS/mvhave7/Results/Preprocessing/grappa/'

np.save(path_to_save_mri_data+"training_data_GrappaNet_16_coils_batch_{}.npy".format(batch_number), x_train)
np.save(path_to_save_mri_data+"training_data_GT_GrappaNet_16_coils_batch_{}.npy".format(batch_number), y_train)
np.save(path_to_save_mri_data+"validation_data_GrappaNet_16_coils_batch_{}.npy".format(batch_number), x_test)
np.save(path_to_save_mri_data+"validation_data_GT_GrappaNet_16_coils_batch_{}.npy".format(batch_number), y_test)

np.save(path_to_save_grappa_data+"grappa_train_indx_GrappaNet_16_coils_batch_{}.npy".format(batch_number), grappa_train_indx)
np.save(path_to_save_grappa_data+"grappa_validation_indx_GrappaNet_16_coils_batch_{}.npy".format(batch_number), grappa_test_indx)

with open(path_to_save_grappa_data+'grappa_wt_batch_{}.pickle'.format(batch_number), 'wb') as handle:
    pickle.dump(grappa_wt, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(path_to_save_grappa_data+'grappa_p_batch_{}.pickle'.format(batch_number), 'wb') as handle:
    pickle.dump(grappa_p, handle, protocol=pickle.HIGHEST_PROTOCOL)


print('Done. Visualizing an example of the processed data to check if everything is ok...')


## Visualize an example of the processed data

# Slice
indx = 7
ref_img = abs(fft.fftshift(fft.ifft2(x_train[indx,:,:,:])))

fix,ax = plt.subplots(nrows=1,ncols=2,figsize=(6,10))
ax[0].imshow(x_train[indx,:,:,0],cmap='gray')
ax[1].imshow(Y_rss[indx,:,:],cmap='gray')

plt.show()


print('Done.')


