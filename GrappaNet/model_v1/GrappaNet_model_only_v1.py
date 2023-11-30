## Set up a resource limiter, such that the script doesn't take up more than a certain amount of RAM (normally 40GB is the limit). In that case, an error will be thrown

import resource

# Because micsd01 has very jobs running currently, we can increase the RAM limit to a higher number than 40GB
resource.setrlimit(resource.RLIMIT_AS, (80_000_000_000, 80_000_000_000))


print('Resource limit set. Importing libraries...')


## Import libraries

import numpy as np
import matplotlib.pyplot as plt
from numpy import fft 
from utils import apply_kernel_weight
import math
import gc

crop_size = (32,640,320)


print('Libraries imported. Starting to load the already preprocessed dataset...')


## Load the already preprocessed dataset

import pickle

path_of_saved_mri_data = '/usr/local/micapollo01/MIC/DATA/STUDENTS/mvhave7/Results/Preprocessing/mri/'
path_of_saved_grappa_data = '/usr/local/micapollo01/MIC/DATA/STUDENTS/mvhave7/Results/Preprocessing/grappa/'

X_train = np.load(path_of_saved_mri_data+"training_data_GrappaNet_16_coils.npy")
Y_train = np.load(path_of_saved_mri_data+"training_data_GT_GrappaNet_16_coils.npy")

with open(path_of_saved_grappa_data+'grappa_wt.pickle', 'rb') as handle:
    grappa_wt = pickle.load(handle)

with open(path_of_saved_grappa_data+'grappa_p.pickle', 'rb') as handle:
    grappa_p = pickle.load(handle)


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


print('Done. Visualizing an example of the processed data to check if everything is ok...')


## Visualize an example of the processed data

# Slice
indx = 10
ref_img = abs(fft.fftshift(fft.ifft2(x_train[indx,:,:,:])))

fix,ax = plt.subplots(nrows=1,ncols=2,figsize=(6,10))
ax[0].imshow(x_train[indx,:,:,0],cmap='gray')
ax[1].imshow(Y_rss[indx,:,:],cmap='gray')

plt.show()


print('Done. Building the GrappaNet model architecture...')


## Build the model

import gc
model = None
gc.collect()

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D
from tensorflow.keras.layers import add, Dropout, Lambda, ReLU
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop
import tensorflow_addons as tfa
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

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


print('Done. Training the model...')


## Train the model

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

model_name = "/usr/local/micapollo01/MIC/DATA/STUDENTS/mvhave7/Results/Models/best_model_GrappaNet.h5"


def step_decay(epoch, initial_lrate, drop, epochs_drop):
    return initial_lrate * math.pow(drop, math.floor((1+epoch)/float(epochs_drop)))

def get_callbacks(model_file, learning_rate_drop=0.7, learning_rate_patience=7, verbosity=1):
    callbacks = list()
    callbacks.append(ModelCheckpoint(model_file, save_best_only=True))
    callbacks.append(ReduceLROnPlateau(factor=learning_rate_drop, patience=learning_rate_patience, verbose=verbosity))
    callbacks.append(EarlyStopping(verbose=verbosity, patience=200))
    return callbacks

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    input_shape = (crop_size[1],crop_size[2],crop_size[0])
    epochs = 1
    batch_size = 8
    model = build_model(input_shape)
    metrics = tf.keras.metrics.RootMeanSquaredError()
    model.compile(loss=model_loss_ssim, optimizer=Adam(learning_rate=0.0003), metrics=[metrics])
    #model.compile(loss=model_loss_ssim, optimizer=RMSprop(learning_rate=0.0003), metrics=[metrics])


history = model.fit([x_train, grappa_train_indx], y_train,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=False,
            validation_data=([x_test, grappa_test_indx], y_test),
            callbacks=get_callbacks(model_name,0.6,10,1),
            max_queue_size=32,
            workers=100,
            use_multiprocessing=False)


model.save_weights("/usr/local/micapollo01/MIC/DATA/STUDENTS/mvhave7/Results/Models/final_model_GrappaNet.h5")


print("Done. Saved model to disk.")


print('Plotting loss function training curve')


print(history.history)

import pandas as pd

pd.DataFrame(history.history).plot(figsize=(8,5))
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
