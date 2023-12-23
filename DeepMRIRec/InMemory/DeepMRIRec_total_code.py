### VERY IMPORTANT CONTRAST WITH GRAPPANET: DEEPMRIREC IS AN IMAGE SPACE METHOD


## Set up a resource limiter, such that the script doesn't take up more than a certain amount of RAM (normally 40GB is the limit). In that case, an error will be thrown

import resource

resource.setrlimit(resource.RLIMIT_AS, (40_000_000_000, 40_000_000_000))


print('Resource limit set. Importing libraries...')


## Import libraries

import random, h5py
import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from numpy import fft 
import os
from pygrappa import grappa
import gc
import time 
from pathlib import Path
from itertools import chain


print('Libraries imported. Starting to prepare the dataset...')


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
training_files = files_16_640_320

# Select the first 5 scans
training_files = training_files[:5]

crop_size = (12,640,320)


print('All variables are loaded. Starting training dataset construction and initial GRAPPA image reconstructions of the undersampled data...')


## Create training data pairs and perform initial GRAPPA image reconstructions of the undersampled data

# First, we define helper functions
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

def Grappa_recon(kspace,start, end):
    calib = kspace[:,:,start:end].copy()
    res = grappa(kspace, calib, kernel_size=(5,5),coil_axis=0)
    return res

def comp_img(img,crop_size):
    s = img.shape
    start_height = s[1]//2 - (crop_size[1]//2)
    start_width = s[2]//2 - (crop_size[2]//2)
    return img[:,start_height:(start_height+crop_size[1]),start_width:(start_width+crop_size[2])]

# Now, we start the training data construction and GRAPPA image reconstruction estimation
cnt = 1
last_mask = None
X_train = []
Y_train = []
for mri_f in sorted(training_files):
    filename = os.path.basename(mri_f)
    filename = filename.replace(".h5","")
    with h5py.File(mri_f,'r') as f:

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
                X_train.append(list(comp_img(sub_img,crop_size)))
                Y_train.append(list(comp_img(ref_img,crop_size)))
            print(cnt,filename,sequence,nSL,nCh,nFE,nPE,sub_img.shape,"ACS region indices: ",start,end,"Processing time: ", time.time()-ts)
            cnt += 1
            gc.collect()
            time.sleep(1)
X_train_arr = np.array(X_train).astype(np.float32)
Y_train_arr = np.array(Y_train).astype(np.float32)
print("X_train and Y_train constructed with shapes: ",X_train_arr.shape,Y_train_arr.shape)
minimum_slices = X_train_arr.shape[0]


print('Done. Visualizing an example of the processed data to check if everything is ok...')


## Visualize an example of the processed data

# Slice
indx = 3
Y_rss = np.sqrt(np.sum(np.square(Y_train_arr),axis=1))

fix,ax = plt.subplots(nrows=1,ncols=2,figsize=(6,8))
ax[0].imshow(X_train_arr[indx,0,:,:],cmap='gray')
ax[1].imshow(Y_rss[indx,:,:],cmap='gray')
plt.show()


print('Done. Saving results for other runs in the future...')


## Save the results as the previous code can run for a long time

path_to_save_mri_data = '/usr/local/micapollo01/MIC/DATA/STUDENTS/mvhave7/Results/Preprocessing/Backlog/fully_processed_at_once_augmented/pre_augmentation/'

np.save(path_to_save_mri_data+"training_data_DeepMRIRec_16_coils.npy", X_train_arr)
np.save(path_to_save_mri_data+"training_data_GT_DeepMRIRec_16_coils.npy", Y_train_arr)


print('Done. Performing data augmentation for increased model performance...')


## Perform data augmentation

import numpy as np
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt
import random

# Define possible image augmentations
seq = iaa.Sequential([

    iaa.Fliplr(1),
    iaa.Flipud(1),
    iaa.Dropout([0.1, 0.2]),
    iaa.GaussianBlur(sigma=(1.5)),
    iaa.GaussianBlur(sigma=(0.8)),
    iaa.GaussianBlur(sigma=(3)),
    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.4, 3.5)),

    #PiecewiseAffine and elastic deformation
    iaa.PiecewiseAffine(scale=(0.01, 0.03)),
    iaa.PiecewiseAffine(scale=(0.04, 0.06)),
    iaa.ElasticTransformation(alpha=(2.0,4.0), sigma=1),
    iaa.ElasticTransformation(alpha=(14.0,17.0), sigma=5),
    iaa.Affine(rotate=(-20, -15),
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            scale={"x": (0.8), "y": (0.8)}, order=3,
            ),
    iaa.Affine(rotate=(-10, -5),
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            scale={"x": (1.1), "y": (1.1)}, order=3,
            ),
    iaa.Affine(rotate=(5, 10),
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            scale={"x": (0.8), "y": (0.8)}, order=3,
            ),
    iaa.Affine(rotate=(15, 20),
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            scale={"x": (1.1), "y": (1.1)}, order=3,
            )
              
], random_order=False)

def apply_augmentation(seq_picked,image,ref):
    augseq_det = seq_picked.to_deterministic()
    dist_image = augseq_det.augment_image(image)
    ref_image = augseq_det.augment_image(ref)
    return dist_image, ref_image

augmented_image_X = []
augmented_image_Y = []
X_train_arr = np.transpose(X_train_arr,(0,2,3,1))
Y_train_arr = np.transpose(Y_train_arr,(0,2,3,1))
for k in range(X_train_arr.shape[0]):
    image = X_train_arr[k,:,:,:].astype(np.float32)
    ref = Y_train_arr[k,:,:,:].astype(np.float32)
    augmented_image_X.append(image)
    augmented_image_Y.append(ref)
    for i in range(len(seq)):
        seq_picked = iaa.Sequential([seq[i]])
        dist_image,ref_image = apply_augmentation(seq_picked,image,ref)
        augmented_image_X.append(dist_image)
        if i in [2,3,4,5,6]:
            augmented_image_Y.append(ref)
        else:
            augmented_image_Y.append(ref_image)
        if dist_image.shape[0] != crop_size[1] and dist_image.shape[1] != crop_size[2] and dist_image.shape[1] != crop_size[0]:
            print("Warning: augmentation resulted in a different slice size: ",image.shape,dist_image.shape,ref_image.shape)
    
    #PiecewiseAffine and Affine
    seq_picked = iaa.Sequential([seq[7],seq[11]])
    dist_image,ref_image = apply_augmentation(seq_picked,image,ref)
    augmented_image_X.append(dist_image)
    augmented_image_Y.append(ref_image)
    
    #Elastic and Affine
    seq_picked = iaa.Sequential([seq[10],seq[11]])
    dist_image,ref_image = apply_augmentation(seq_picked,image,ref)
    augmented_image_X.append(dist_image)
    augmented_image_Y.append(ref_image)
    
    seq_picked = iaa.Sequential([seq[9],seq[14]])
    dist_image,ref_image = apply_augmentation(seq_picked,image,ref)
    augmented_image_X.append(dist_image)
    augmented_image_Y.append(ref_image)

X_train_arr = np.array(augmented_image_X)
Y_train_arr = np.array(augmented_image_Y)

def select_slices(array, slice_index, total_slices):
    # Ensure the array has 4 dimensions
    if array.ndim != 4:
        raise ValueError("Input array must be 4-dimensional")
        
    # Ensure total_slices is not greater than the size of the first dimension
    if total_slices > array.shape[0]:
        raise ValueError("Total slices must be less than or equal to the size of the first dimension of the input array")
    
    # All slices before slice_index are definitely kept
    slices_to_keep = array[:slice_index, :, :, :]
    
    # Only select slices beyond slice_index if total_slices is greater than slice_index
    if total_slices > slice_index:
        slices_to_select = array[slice_index:, :, :, :]
        
        # Randomly select the remaining slices to reach total_slices
        indices = np.random.choice(slices_to_select.shape[0], total_slices - slice_index, replace=False)
        slices_to_add = slices_to_select[indices, :, :, :]
        
        # Concatenate the slices to keep and the newly selected slices
        output_array = np.concatenate((slices_to_keep, slices_to_add), axis=0)
    else:
        output_array = slices_to_keep

    return output_array

X_train_arr = select_slices(X_train_arr, minimum_slices, 150)
Y_train_arr = select_slices(Y_train_arr, minimum_slices, 150)
print("New dimensions of X_train and Y_train: ",X_train_arr.shape,Y_train_arr.shape) 


print("Done. Saving results for other runs in the future...")


## Save the results as the previous code can run for a long time

path_to_save_mri_data = '/usr/local/micapollo01/MIC/DATA/STUDENTS/mvhave7/Results/Preprocessing/Backlog/fully_processed_at_once_augmented/post_augmentation/'

np.save(path_to_save_mri_data+"augmented_training_data_DeepMRIRec_16_coils.npy", X_train_arr)
np.save(path_to_save_mri_data+"augmented_training_data_GT_DeepMRIRec_16_coils.npy", Y_train_arr)


print('Done. Calculating RSS images as references for the loss function...')


## Calculate RSS images that will be used as references for the loss function

Y_rss = np.sqrt(np.sum(np.square(Y_train_arr),axis=3))
Y_rss = Y_rss.astype(np.float32)


print('Done. Normalizing the data...')


## Normalize the data

dims = X_train_arr.shape

for i in range(dims[0]):
    for j in range(dims[3]):
        X_train_arr[i,:,:,j] = X_train_arr[i,:,:,j]/((np.max(X_train_arr[i,:,:,j])-np.min(X_train_arr[i,:,:,j]))+1e-10)

for i in range(dims[0]):
    Y_rss[i,:,:] = Y_rss[i,:,:]/((np.max(Y_rss[i,:,:])-np.min(Y_rss[i,:,:]))+1e-10)


print('Done. Performing a datasplit...')


## Create a dataset split 90-10 training-validation

Y_rss = Y_rss.reshape((dims[0],dims[1],dims[2],1))
x_train = X_train_arr[0:int(X_train_arr.shape[0]-X_train_arr.shape[0]*0.1),:,:,:]
y_train = Y_rss[0:int(X_train_arr.shape[0]-X_train_arr.shape[0]*0.1),:,:,:]
x_test = X_train_arr[int(X_train_arr.shape[0]-X_train_arr.shape[0]*0.1):,:,:,:]
y_test = Y_rss[int(X_train_arr.shape[0]-X_train_arr.shape[0]*0.1):,:,:,:]

del X_train
del Y_train
del augmented_image_X
del augmented_image_Y
del X_train_arr
del Y_train_arr
del Y_rss
time.sleep(1)
gc.collect()
time.sleep(1)


print('Done. Building the DeepMRIRec model architecture...')


## Build the model

import tensorflow as tf
from tensorflow.keras.layers import Input,BatchNormalization, Conv2D, MaxPooling2D
from tensorflow.keras.layers import PReLU, add, Attention, Dropout
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input

model = None
kernel_size = (3,3)
loss_weights = [1.0, 0.0001, 0.000001]

selected_layers = ['block1_conv1', 'block2_conv2', 'block3_conv3' ,'block4_conv3']
selected_layer_weights_content = [0.001, 0.01, 2, 4]

vgg = VGG19(weights='imagenet', include_top=False, input_shape=(crop_size[1],crop_size[2],3))
vgg.trainable = False

outputs = [vgg.get_layer(l).output for l in selected_layers]
vgg_model = Model(vgg.input, outputs)
vgg_model.trainable = False

def vgg_RSS(tensor):
    Y_rss = tf.math.sqrt(tf.math.reduce_sum(tf.square(tensor),axis=-1))
    return Y_rss

def compute_loss(A, B):
    meanA = tf.reduce_mean(A, axis=(1,2), keepdims=True)
    meanB = tf.reduce_mean(B, axis=(1,2), keepdims=True)
    nA = tf.sqrt(tf.reduce_sum(tf.square(A), axis=(1,2), keepdims=True))
    nB = tf.sqrt(tf.reduce_sum(tf.square(B), axis=(1,2), keepdims=True))
    return 1-(tf.reduce_sum((A-meanA)/nA*(B-meanB)/nB))

def model_loss_all(y_true, y_pred):
    global vgg_model
    global loss_weights
    
    ssim_loss = 1- tf.math.abs(tf.reduce_mean(tf.image.ssim(img1=y_true,img2=y_pred,max_val=1.0,filter_size=3,filter_sigma=0.1)))
    pixel_loss = tf.reduce_mean(tf.math.abs(y_true-y_pred))
    
    content_loss = 0.0
    res_y_rss = tf.image.grayscale_to_rgb(y_true*255)
    res_y_rss = preprocess_input(res_y_rss)
    vgg_f_gt = vgg_model(res_y_rss)

    res_y_pred = tf.image.grayscale_to_rgb(y_pred*255)
    res_y_pred = preprocess_input(res_y_pred)
    vgg_f_pred = vgg_model(res_y_pred)

    for h1, h2, cw in zip(vgg_f_gt, vgg_f_pred, selected_layer_weights_content):
        content_loss = content_loss + cw *tf.reduce_mean(tf.square(tf.math.abs(h1 - h2)))
    
    return loss_weights[0]*ssim_loss+loss_weights[1]*pixel_loss +loss_weights[2]*content_loss

def model_loss_NMSE(y_true, y_pred):
    nmse = tf.sqrt(y_pred-y_true)/(tf.sqrt(y_true)+1e-8)
    return nmse

def conv_block(ip, nfilters, drop_rate):
    
    layer_top = Conv2D(nfilters, (3,3), padding = "same")(ip)
    layer_top = BatchNormalization()(layer_top)

    res_model = PReLU()(layer_top)
    res_model = Dropout(drop_rate)(res_model)
    
    res_model = Conv2D(nfilters, (3,3), padding = "same")(res_model)
    res_model = BatchNormalization()(res_model)

    res_model = Dropout(drop_rate)(res_model)
    res_model = add([layer_top,res_model])
    res_model = PReLU()(res_model)
    return res_model

def encoder(inp, nlayers, nbasefilters, drop_rate):
    
    skip_layers = []
    layers = inp
    for i in range(nlayers):
        layers = conv_block(layers,nbasefilters*2**i,drop_rate)
        
        #attention
        #layers = Attention()([layers,layers])
        
        skip_layers.append(layers)
        layers = MaxPooling2D((2,2))(layers)
    
    return layers, skip_layers

def decoder(inp, nlayers, nbasefilters,skip_layers, drop_rate):
    
    layers = inp
    for i in range(nlayers):
        layers = conv_block(layers,nbasefilters*(2**(nlayers-1-i)),drop_rate)
        
        #attention
        #layers=Attention()([layers,layers])
        #mul_layer = MultiHeadAttention(num_heads=2, key_dim=2, attention_axes=(1,2, 3))
        #layers=mul_layer(layers, layers)

        #layers=UpSampling2D((2,2))(layers)
        layers=Conv2DTranspose(kernel_size=(2,2),filters=nbasefilters*(2**(nlayers-1-i)),strides=(2,2), padding='same')(layers)
        layers=add([layers,skip_layers.pop()])
    return layers

def create_gen(gen_ip, nlayers, nbasefilters, drop_rate):
    op,skip_layers = encoder(gen_ip,nlayers, nbasefilters,drop_rate)
    op = decoder(op,nlayers, nbasefilters,skip_layers,drop_rate)
    op = Conv2D(1, (3,3), padding = "same")(op)
    return Model(inputs=gen_ip,outputs=op)

input_shape = (crop_size[1],crop_size[2],crop_size[0])
input_layer = Input(shape=input_shape)
model = create_gen(input_layer,5,32,0.01)


print('Done. Training the model...')


## Train the model

import math
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

model_name = "/usr/local/micapollo01/MIC/DATA/STUDENTS/mvhave7/Results/Models/Backlog/best_model_DeepMRIRec.h5"

def step_decay(epoch, initial_lrate, drop, epochs_drop):
    return initial_lrate * math.pow(drop, math.floor((1+epoch)/float(epochs_drop)))

def get_callbacks(model_file, learning_rate_drop=0.7,learning_rate_patience=7, verbosity=1):
    callbacks = list()
    callbacks.append(ModelCheckpoint(model_file, save_best_only=True))
    callbacks.append(ReduceLROnPlateau(factor=learning_rate_drop, patience=learning_rate_patience,verbose=verbosity))
    callbacks.append(EarlyStopping(verbose=verbosity, patience=30))
    return callbacks

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = create_gen(input_layer,5,32,0.01)
    metrics = tf.keras.metrics.RootMeanSquaredError()
    model.compile(loss=model_loss_all, optimizer= Adam(learning_rate=0.0001),metrics=[metrics] )
    history = model.fit(x_train, y_train,
                    epochs=20,  # In their paper, they use 100 epochs
                    batch_size=128,
                    shuffle=True,
                    validation_data=(x_test, y_test),
                    callbacks=get_callbacks(model_name,0.6,10,1))

model.save_weights("/usr/local/micapollo01/MIC/DATA/STUDENTS/mvhave7/Results/Models/Backlog/final_model_DeepMRIRec.h5")


print("Done. Saved model to disk.")


print('Plotting loss function training curve')


#import pandas as pd
#pd.DataFrame(history.history).plot(figsize=(8,5))
#plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


