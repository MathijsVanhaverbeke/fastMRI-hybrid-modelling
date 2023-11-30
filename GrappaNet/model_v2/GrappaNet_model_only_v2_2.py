## Set up a resource limiter, such that the script doesn't take up more than a certain amount of of RAM (normally 40GB is the limit). In that case, an error will be thrown

import resource

# Because micsd01 has very jobs running currently, we can increase the RAM limit to a higher number than 40GB
memory_limit = 100_000_000_000
resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))


print('Resource limit set. Importing libraries...')


## Import libraries

import numpy as np
import matplotlib.pyplot as plt
from utils import apply_kernel_weight
import math
import glob
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gc
import time
import tensorflow as tf
from tensorflow.keras import backend as k
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D
from tensorflow.keras.layers import add, Dropout, Lambda, ReLU
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop
import tensorflow_addons as tfa
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

model = None
crop_size = (32,640,320)


print('Libraries imported. Starting to prepare the dataset...')


## Prepare dataset

clustered_data_2 = np.load("/usr/local/micapollo01/MIC/DATA/STUDENTS/mvhave7/Results/Preprocessing/exploration/16coil_slice_size_clustered_fastmri_data.npy", allow_pickle=True)
clustered_data_2 = clustered_data_2.item()
files_16_640_320 = clustered_data_2[(640,320)]
training_files = sorted(files_16_640_320)
batch_size_files = 5
num_batches = len(training_files) // batch_size_files
batch_number = 1

path_to_save_mri_data = '/usr/local/micapollo01/MIC/DATA/STUDENTS/mvhave7/Results/Preprocessing/mri/'
path_to_save_grappa_data = '/usr/local/micapollo01/MIC/DATA/STUDENTS/mvhave7/Results/Preprocessing/grappa/'


file_paths_train = sorted(glob.glob(path_to_save_mri_data+"training_data_GrappaNet_16_coils_batch_*.npy"))
file_paths_train_GT = sorted(glob.glob(path_to_save_mri_data+"training_data_GT_GrappaNet_16_coils_batch_*.npy"))
file_paths_val = sorted(glob.glob(path_to_save_mri_data+"validation_data_GrappaNet_16_coils_batch_*.npy"))
file_paths_val_GT = sorted(glob.glob(path_to_save_mri_data+"validation_data_GT_GrappaNet_16_coils_batch_*.npy"))
file_paths_grappa_indx_train = sorted(glob.glob(path_to_save_grappa_data+"grappa_train_indx_GrappaNet_16_coils_batch_*.npy"))
file_paths_grappa_indx_val = sorted(glob.glob(path_to_save_grappa_data+"grappa_validation_indx_GrappaNet_16_coils_batch_*.npy"))
file_paths_grappa_wt = sorted(glob.glob(path_to_save_grappa_data+"grappa_wt_batch_*.pickle"))
file_paths_grappa_p = sorted(glob.glob(path_to_save_grappa_data+"grappa_p_batch_*.pickle"))


print('Done. Building the GrappaNet model architecture...')


## Build the model

lamda = 0.001

@tf.function
def model_loss_ssim(y_true, y_pred):
    global lamda
    ssim_loss = 0
    max_val = 1.0
    if tf.reduce_max(y_pred)>1.0:
        max_val = tf.reduce_max(y_pred)
    ssim_loss = tf.math.abs(tf.reduce_mean(tf.image.ssim(img1=y_true,img2=y_pred,max_val=max_val,filter_size=3,filter_sigma=0.1)))
    l1_loss = lamda*tf.reduce_mean(tf.math.abs(y_true-y_pred))
    return 1-ssim_loss+l1_loss


def conv_block(ip, nfilters, drop_rate):
    
    layer_top = Conv2D(nfilters,(3,3),padding="same")(ip)

    #layer_top = BatchNormalization()(layer_top)
    layer_top = tfa.layers.InstanceNormalization(axis=3,center=True, 
                                                 scale=True,beta_initializer="random_uniform",
                                                 gamma_initializer="random_uniform")(layer_top)
    res_model = ReLU()(layer_top)
    res_model = Dropout(drop_rate)(res_model)
    
    res_model = Conv2D(nfilters,(3,3),padding="same")(res_model)
    res_model = tfa.layers.InstanceNormalization(axis=3, center=True, 
                                                 scale=True,beta_initializer="random_uniform",
                                                 gamma_initializer="random_uniform")(res_model)
    #res_model = BatchNormalization()(res_model)
    res_model = Dropout(drop_rate)(res_model)
    res_model = add([layer_top,res_model])
    res_model = ReLU()(res_model)
    return res_model


def encoder(inp,nlayers, nbasefilters, drop_rate):
    
    skip_layers = []
    layers = inp
    for i in range(nlayers):
        layers = conv_block(layers,nbasefilters*2**i,drop_rate)
        skip_layers.append(layers)
        layers = MaxPooling2D((2,2))(layers)
    return layers, skip_layers


def decoder(inp,nlayers, nbasefilters, skip_layers, drop_rate):
    
    layers = inp
    for i in range(nlayers):
        layers = conv_block(layers,nbasefilters*(2**(nlayers-1-i)),drop_rate)
        layers = UpSampling2D(size=(2,2),interpolation='bilinear')(layers)
        layers = add([layers,skip_layers.pop()])
    return layers


def create_gen(gen_ip, nlayers, nbasefilters, drop_rate):
    op,skip_layers = encoder(gen_ip,nlayers,nbasefilters,drop_rate)
    op = decoder(op,nlayers,nbasefilters,skip_layers,drop_rate)
    op = Conv2D(crop_size[0],(3,3),padding="same")(op)
    return op


def custom_data_consistency(tensors):
    output = tf.where(tf.greater_equal(tensors[0], 1), tensors[0], tensors[1])
    out_cmplx = tf.complex(output[:,:,:,0:(crop_size[0]//2)], output[:,:,:,(crop_size[0]//2):(crop_size[0])])
    ift_sig = tf.signal.fftshift(tf.signal.ifft2d(out_cmplx, name=None))
    real_p = tf.math.real(ift_sig)
    imag_p = tf.math.imag(ift_sig)
    comb = tf.concat(axis=-1,values=[real_p, imag_p])
    return comb


def custom_data_consistency_2(tensors):
    out_cmplx = tf.complex(tensors[1][:,:,:,0:(crop_size[0]//2)], tensors[1][:,:,:,(crop_size[0]//2):(crop_size[0])])
    ft_sig = tf.signal.fftshift(tf.signal.fft2d(out_cmplx, name=None))
    real_p = tf.math.real(ft_sig)
    imag_p = tf.math.imag(ft_sig)
    comb = tf.concat(axis=-1,values=[real_p, imag_p])
    output = tf.where(tf.greater_equal(tensors[0], 1), tensors[0], comb)
    return output


def aux_Grappa_layer(tensor1, tensor2):
    global grappa_wt
    global grappa_p
    t1 = tensor1.numpy()
    t2 = tensor2.numpy()

    x_train_cmplx_target = t2[:,:,:,0:(crop_size[0]//2)]+1j*t2[:,:,:,(crop_size[0]//2):(crop_size[0])]
    x_train_cmplx_target = np.transpose(x_train_cmplx_target,(0,3,1,2))
    l_grappa = []
    for i in range(x_train_cmplx_target.shape[0]):
        res = apply_kernel_weight(kspace=x_train_cmplx_target[i],calib=None,
                                 kernel_size=(5,5),coil_axis=0,
                                 weights=grappa_wt[int(t1[i][0])],P=grappa_p[int(t1[i][0])])
        res = np.transpose(res,(1,2,0))
        out_cmplx_real = tf.convert_to_tensor(res.real)
        out_cmplx_imag = tf.convert_to_tensor(res.imag)
        comb = tf.concat(axis=2,values=[out_cmplx_real, out_cmplx_imag])
        l_grappa.append(comb)
    b_grappa = tf.stack(l_grappa)

    return b_grappa


def Grappa_layer(tensor):
    out_tensor = tf.py_function(func=aux_Grappa_layer, inp=tensor, Tout=tf.float32)
    out_tensor.set_shape(tensor[1].get_shape())
    return out_tensor


def ift_RSS(tensor):
    cmplx_tensor = tf.complex(tensor[:,:,:,0:(crop_size[0]//2)], tensor[:,:,:,(crop_size[0]//2):(crop_size[0])])
    ift_sig = tf.signal.fftshift(tf.signal.ifft2d(cmplx_tensor, name=None))
    Y_rss = tf.math.sqrt(tf.math.reduce_sum(tf.square(tf.math.abs(ift_sig)),axis=3))
    return Y_rss


def build_model(input_shape, n_filter=32, n_depth=4, dropout_rate=0.05):

    #first pass
    input_layer = Input(shape=input_shape)
    input_layer_grappa_wt_indx = Input(shape=(1))
    kspace_u1 = create_gen(input_layer,n_depth,n_filter,dropout_rate)
    data_con_layer = Lambda(custom_data_consistency, name="data_const_K_u1")([input_layer, kspace_u1])
    img_space_u1 = create_gen(data_con_layer,n_depth,n_filter,dropout_rate)
    data_con_layer = Lambda(custom_data_consistency_2, name="data_const_K_u1_2")([input_layer, img_space_u1])
    grappa_recon_k = Lambda(Grappa_layer, name="data_const_K_2")([input_layer_grappa_wt_indx, data_con_layer])
    
    #second Pass
    kspace_u2 = create_gen(grappa_recon_k,n_depth,n_filter,dropout_rate)
    data_con_layer = Lambda(custom_data_consistency, name="data_const_K_u2")([input_layer, kspace_u2])
    img_space_u2 = create_gen(data_con_layer,n_depth,n_filter,dropout_rate)
    data_con_layer = Lambda(custom_data_consistency_2, name="data_const_K_u2_2")([input_layer, img_space_u2])
    
    #IFT+RSS
    data_con_layer = Lambda(ift_RSS, name="IFT_RSS")(data_con_layer)

    return Model(inputs=[input_layer,input_layer_grappa_wt_indx],outputs=data_con_layer)


print('Done. Training the model in batches...')


## Train the model

input_shape = (crop_size[1],crop_size[2],crop_size[0])
epochs = 1
model = build_model(input_shape)


def train_model(model, train, target, val, val_target, optimizer, metrics, val_metrics, n_epochs, batch_number, verbose=True, plot=True):
    
    loss_history = []
    val_loss_history = []
    metrics_history = []
    val_metrics_history = []
    
    for epoch in range(n_epochs):
    
        with tf.GradientTape() as tape:
            outputs = model(train, training=True)
            current_loss = model_loss_ssim(target, outputs)
        
        gradients = tape.gradient(current_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        loss_history.append(current_loss.numpy())
        metrics.update_state(target, outputs)
        current_metrics = metrics.result().numpy()
        metrics_history.append(current_metrics)
    
        metrics.reset_state()

        val_outputs = model(val, training=False)
        val_current_loss = model_loss_ssim(val_target, val_outputs)

        val_loss_history.append(val_current_loss.numpy())
        val_metrics.update_state(val_target, val_outputs)
        val_current_metrics = val_metrics.result().numpy()
        val_metrics_history.append(val_current_metrics)

        val_metrics.reset_state()

        if verbose:            
            print(str(epoch+1) + '.\tTraining Loss: ' + str(current_loss.numpy()) + '.\tValidation Loss: ' + str(val_current_loss.numpy()) + ',\tTraining RMSE: ' + str(current_metrics) + ',\tValidation RMSE: ' + str(val_current_metrics))
        
    
    print('\nTraining complete for batch {}.').format(batch_number)
    
    if plot:
        plt.figure(figsize = (15, 4))
        plt.subplots_adjust(wspace=0.2)  # this distances the plots

        plt.subplot(1, 2, 1)
        plt.plot(loss_history)
        plt.plot(val_loss_history)
        plt.legend('Training','Validation')
        plt.title('Loss History')
        plt.xlabel('epochs')
        plt.ylabel('Binary Cross-Entropy')

        plt.subplot(1, 2, 2)
        plt.plot(metrics_history)
        plt.plot(val_metrics_history)
        plt.legend('Training','Validation')
        plt.title('Metrics History')
        plt.xlabel('epochs')
        plt.ylabel('Metrics')
        
        plt.show()
    
    return loss_history, metrics_history


# Define an evaluation metric of the model's goodness
rmse = tf.keras.metrics.RootMeanSquaredError()
val_rmse = tf.keras.metrics.RootMeanSquaredError()

# Define optimizer
optimizer = tf.optimizers.Adam(learning_rate=0.0003)

# Write a custom training loop which trains the model on batches which are loaded in on the fly
for batch in range(num_batches):

    x_train = np.load(file_paths_train[batch_number-1])
    y_train = np.load(file_paths_train_GT[batch_number-1])
    grappa_train_indx = np.load(file_paths_grappa_indx_train[batch_number-1])
    with open(file_paths_grappa_wt[batch_number-1], 'rb') as handle:
        grappa_wt = pickle.load(handle)
    with open(file_paths_grappa_p[batch_number-1], 'rb') as handle:
        grappa_p = pickle.load(handle)
    
    x_test = np.load(file_paths_val[batch_number-1])
    y_test = np.load(file_paths_val_GT[batch_number-1])
    grappa_test_indx = np.load(file_paths_grappa_indx_val[batch_number-1])

    train_inputs = [x_train, grappa_train_indx]
    test_inputs = [x_test, grappa_test_indx]
    
    loss_history, accuracy_history = train_model(model, train_inputs, y_train, test_inputs, y_test, optimizer, rmse, val_rmse, n_epochs=epochs, batch_number=batch_number)

    del x_train, y_train, grappa_train_indx, x_test, y_test, grappa_test_indx, grappa_wt, grappa_p
    time.sleep(1)
    gc.collect()
    time.sleep(1)

    batch_number += 1


model.save_weights("/usr/local/micapollo01/MIC/DATA/STUDENTS/mvhave7/Results/Models/final_model_GrappaNet.h5")


print("Done. Saved model to disk.")

