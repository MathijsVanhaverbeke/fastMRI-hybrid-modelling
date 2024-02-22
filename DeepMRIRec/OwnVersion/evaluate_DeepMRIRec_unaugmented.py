### DeepMRIRec version 4 - Own Version

import resource

resource.setrlimit(resource.RLIMIT_AS, (40_000_000_000, 40_000_000_000))

print('Resource limit set. Importing libraries...')

import h5py
import random
import glob
from pygrappa import grappa
from numpy import fft 
import time
import gc
from typing import Optional
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Input,BatchNormalization, Conv2D, MaxPooling2D
from tensorflow.keras.layers import PReLU, LeakyReLU, add, Attention, Dropout
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.models import Model

print('Libraries imported. Starting to load the model architecture...')

crop_size = (12,640,320)

model = None
kernel_size = (3,3)
loss_weights = [1.0, 0.000001]

def ssim_tf(gt_tf, pred_tf, maxval=None):
    maxval = tf.math.reduce_max(gt_tf) if maxval is None else maxval
    ssim = tf.reduce_mean(tf.image.ssim(img1=gt_tf,img2=pred_tf,max_val=maxval))
    return ssim

def model_loss_all(y_true, y_pred):
    global loss_weights
    
    ssim_loss = 1 - ssim_tf(y_true, y_pred)
    l1_loss = tf.reduce_sum(tf.math.abs(y_true-y_pred))
    
    return loss_weights[0]*ssim_loss+loss_weights[1]*l1_loss

def conv_block(ip, nfilters, drop_rate):
    
    layer_top = Conv2D(nfilters, (3,3), padding = "same")(ip)
    layer_top = BatchNormalization()(layer_top)

    res_model = LeakyReLU()(layer_top)
    res_model = Dropout(drop_rate)(res_model)
    
    res_model = Conv2D(nfilters, (3,3), padding = "same")(res_model)
    res_model = BatchNormalization()(res_model)

    res_model = Dropout(drop_rate)(res_model)
    res_model = add([layer_top,res_model])
    res_model = LeakyReLU()(res_model)
    return res_model

def encoder(inp, nlayers, nbasefilters, drop_rate):
    
    skip_layers = []
    layers = inp
    for i in range(nlayers):
        layers = conv_block(layers,nbasefilters*2**i,drop_rate)
        skip_layers.append(layers)
        layers = MaxPooling2D((2,2))(layers)
    return layers, skip_layers

def decoder(inp, nlayers, nbasefilters,skip_layers, drop_rate):
    
    layers = inp
    for i in range(nlayers):
        layers = conv_block(layers,nbasefilters*(2**(nlayers-1-i)),drop_rate)
        layers=Conv2DTranspose(kernel_size=(2,2),filters=nbasefilters*(2**(nlayers-1-i)),strides=(2,2), padding='same')(layers)
        layers=add([layers,skip_layers.pop()])
    return layers

def create_gen(gen_ip, nlayers, nbasefilters, drop_rate):
    op,skip_layers = encoder(gen_ip,nlayers, nbasefilters,drop_rate)
    op = decoder(op,nlayers, nbasefilters,skip_layers,drop_rate)
    op = Conv2D(1, (3,3), padding = "same")(op)
    op = Activation('linear', dtype='float32')(op)
    return Model(inputs=gen_ip,outputs=op)

input_shape = (crop_size[1],crop_size[2],crop_size[0])
input_layer = Input(shape=input_shape)

print('Done. Starting to define necessary variables...')

path_to_checkpoints = '/usr/local/micapollo01/MIC/DATA/STUDENTS/mvhave7/Results/Models/UnaugmentedExtensiveSavesOwnVersion/'
model_checkpoints = file_paths_test_GT = sorted(glob.glob(path_to_checkpoints+"DeepMRIRec_*"))

evaluation_save_path = '/usr/local/micapollo01/MIC/DATA/STUDENTS/mvhave7/Results/Reconstructions/DeepMRIRec/UnaugmentedEvaluationsOwnVersion/'
path_to_selected_test_data = '/usr/local/micapollo01/MIC/DATA/STUDENTS/mvhave7/Results/Preprocessing/exploration/'
file_paths_test_GT = list(np.load(path_to_selected_test_data+'DeepMRIRec_adequate_test_scans.npy'))
test_scans = random.sample(file_paths_test_GT, 30)
np.save(evaluation_save_path+'evaluated_test_scans.npy', np.array(test_scans))

mse_scores = []
nmse_scores = []
psnr_scores = []
ssim_scores = []
epoch = 1

def calculate_mask(mask,start,end,nPE):
    '''
    The mask contains more information from the center of k-space. 
    We divide phase encoding space into nine areas/slots (see Figure 3 from paper, nine slots are separated by red dotted lines) 
    and select 10% of data points from the center area and 4%, 2%,1%, and 0.5% from the area adjacent to the centers
    In total, we thus retain 10% + (4+2+1+0.5)*2 % = 25% of the original k-space. This way, we obtain R=4.
    '''
    
    total_point = start
    max_len = nPE
    number_of_sample = (max_len/4)-(end-start)
    step = int((4*start)/number_of_sample)
    i = step
    f = 1
    indx = 1
    while i < total_point+1:
        offset = 10 + int(random.sample(range(0,3), 1)[0])
        if offset+i < total_point:
            mask[:,:,:,offset+i] = 1
        else:
            mask[:,:,:,i] = 1

        offset = 5 + int(random.sample(range(0,3), 1)[0])
        if i-offset > 0:
            mask[:,:,:,max_len-i-offset] = 1
        else:
            mask[:,:,:,max_len-i] = 1
        i = i+step
        indx = indx+1
        if i >= ((total_point*f)//2):
            step = int(step/3)
            f = f+1
    return mask

def Grappa_recon(kspace,start,end):
    calib = kspace[:,:,start:end].copy()
    res = grappa(kspace, calib, kernel_size=(5,5),coil_axis=0)
    return res

def comp_img(img,crop_size):
    ## Note: only functions correctly if the height and width of a slice of img are larger than the desired crop size
    s = img.shape
    start_height = s[1]//2 - (crop_size[1]//2)
    start_width = s[2]//2 - (crop_size[2]//2)
    return img[:,start_height:(start_height+crop_size[1]),start_width:(start_width+crop_size[2])]

def mse(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute Mean Squared Error (MSE)"""
    return np.mean((gt - pred) ** 2)

def nmse(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute Normalized Mean Squared Error (NMSE)"""
    return np.array(np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2)

def psnr(
    gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None
) -> np.ndarray:
    """Compute Peak Signal to Noise Ratio metric (PSNR)"""
    if maxval is None:
        maxval = gt.max()
    return peak_signal_noise_ratio(gt, pred, data_range=maxval)

def ssim(
    gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None
) -> np.ndarray:
    """Compute Structural Similarity Index Metric (SSIM)"""
    if not gt.ndim == 3:
        raise ValueError("Unexpected number of dimensions in ground truth.")
    if not gt.ndim == pred.ndim:
        raise ValueError("Ground truth dimensions does not match pred.")

    maxval = gt.max() if maxval is None else maxval

    ssim = np.array([0])
    for slice_num in range(gt.shape[0]):
        ssim = ssim + structural_similarity(
            gt[slice_num], pred[slice_num], data_range=maxval
        )

    return ssim / gt.shape[0]

print('Done. Starting to evaluate model checkpoints...')

for checkpoint in model_checkpoints:

    print('Epoch: '+str(epoch))

    model = create_gen(input_layer,5,32,0.01)
    model.load_weights(checkpoint)

    for test_scan in test_scans:

        last_mask = None
        X_test = []
        Y_test = []

        with h5py.File(test_scan,'r') as f:

            k = f['kspace'][()]
            sequence = f.attrs['acquisition']
            nSL, nCh, nFE, nPE = k.shape
            
            # Select ACS region
            mid = nPE//2
            start = mid-int(nPE*0.05)
            end = mid+int(nPE*0.05)
            
            mask = 0*k
            mask[:,:,:,start:end] = 1
            mask = calculate_mask(mask,start,end,nPE)  
            last_mask = mask
            subsampled_k = k*mask
            ts = time.time()
            if nCh > 18:
                channels_picked = [0,1,2,4,7,8,11,13,15,16,17,18]
            elif nCh > 12 and nCh < 18:
                channels_picked = [0,1,2,3,4,5,6,7,8,10,11,12]
            else:
                channels_picked = []
            if len(channels_picked)>0:
                for slices in range(nSL): 
                    chindx = 0
                    ref_img = np.zeros((len(channels_picked),nFE,nPE),dtype=np.float32)
                    sub_img = np.zeros((len(channels_picked),nFE,nPE),dtype=np.float32)
                    subsampled_tmp = Grappa_recon(subsampled_k[slices,:,:,:],start,end)    
                    for iCh in channels_picked:
                        sub_img[chindx,:,:] = abs(fft.fftshift(fft.ifft2(subsampled_tmp[iCh,:,:])))
                        ref_img[chindx,:,:] = abs(fft.fftshift(fft.ifft2(k[slices,iCh,:,:])))
                        chindx = chindx+1
                    X_test.append(list(comp_img(sub_img,crop_size)))
                    Y_test.append(list(comp_img(ref_img,crop_size)))
        X_test_arr = np.array(X_test).astype(np.float32)
        Y_test_arr = np.array(Y_test).astype(np.float32)

        X_test_arr = np.transpose(X_test_arr,(0,2,3,1))
        Y_test_arr = np.transpose(Y_test_arr,(0,2,3,1))

        Y_rss = np.sqrt(np.sum(np.square(Y_test_arr),axis=3))
        Y_rss = Y_rss.astype(np.float32)

        def normalize(data, mean, std, eps=0.0):
            return (data - mean) / (std + eps)

        def normalize_instance(data, eps=0.0):
            mean = np.mean(data)
            std = np.std(data)
            return normalize(data, mean, std, eps), mean, std

        dims = X_test_arr.shape

        for i in range(dims[0]):
            for j in range(dims[-1]):
                X_test_arr[i,:,:,j], _, _ = normalize_instance(X_test_arr[i,:,:,j])
                X_test_arr[i,:,:,j] = np.clip(X_test_arr[i,:,:,j], -6, 6)

        for i in range(dims[0]):
            Y_rss[i,:,:], _, _ = normalize_instance(Y_rss[i,:,:])
            Y_rss[i,:,:] = np.clip(Y_rss[i,:,:], -6, 6)

        # Make prediction with our model
        reconstructed_test_image = model.predict(X_test_arr)
        reconstructed_test_image = reconstructed_test_image[:,:,:,0]

        # Note: actually, here we should also look into cropping the images to the central 320x320 region to avoid background noise influencing the metrics
        calculated_mse = mse(Y_rss, reconstructed_test_image)
        calculated_nmse = nmse(Y_rss, reconstructed_test_image)
        calculated_psnr = psnr(Y_rss, reconstructed_test_image)
        calculated_ssim = ssim(Y_rss, reconstructed_test_image)

        mse_scores.append(calculated_mse)
        nmse_scores.append(calculated_nmse)
        psnr_scores.append(calculated_psnr)
        ssim_scores.append(calculated_ssim)
    
    mse_scores_array = np.array(mse_scores)
    nmse_scores_array = np.array(nmse_scores)
    psnr_scores_array = np.array(psnr_scores)
    ssim_scores_array = np.array(ssim_scores)
    np.save(evaluation_save_path+'mse_scores_epoch_{}.npy'.format(epoch), mse_scores_array)
    np.save(evaluation_save_path+'nmse_scores_epoch_{}.npy'.format(epoch), nmse_scores_array)
    np.save(evaluation_save_path+'psnr_scores_epoch_{}.npy'.format(epoch), psnr_scores_array)
    np.save(evaluation_save_path+'ssim_scores_epoch_{}.npy'.format(epoch), ssim_scores_array)

    print('MSE: '+str(mse_scores_array.mean())+' +- '+str(mse_scores_array.std()))
    print('NMSE: '+str(nmse_scores_array.mean())+' +- '+str(nmse_scores_array.std()))
    print('PSNR: '+str(psnr_scores_array.mean())+' +- '+str(psnr_scores_array.std()))
    print('SSIM: '+str(ssim_scores_array.mean())+' +- '+str(ssim_scores_array.std()))

    model = None

    time.sleep(1)
    del X_test
    del Y_test
    del X_test_arr
    del Y_test_arr
    del Y_rss
    del model
    del mse_scores, mse_scores_array
    del nmse_scores, nmse_scores_array
    del psnr_scores, psnr_scores_array
    del ssim_scores, ssim_scores_array
    time.sleep(1)

    gc.collect()

    mse_scores = []
    nmse_scores = []
    psnr_scores = []
    ssim_scores = []
    epoch += 1

print('Done.')
