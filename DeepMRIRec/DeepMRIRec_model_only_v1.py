### VERY IMPORTANT CONTRAST WITH GRAPPANET: DEEPMRIREC IS AN IMAGE SPACE METHOD


## Set up a resource limiter, such that the script doesn't take up more than a certain amount of RAM (normally 40GB is the limit). In that case, an error will be thrown

import resource

# Because micsd01 has very few jobs running currently, we can increase the RAM limit to a higher number than 40GB
resource.setrlimit(resource.RLIMIT_AS, (150_000_000_000, 150_000_000_000))


print('Resource limit set. Importing libraries...')


## Import libraries

import numpy as np
import matplotlib.pyplot as plt
from pygrappa import grappa, mdgrappa
import gc
import time

crop_size = (12,640,320)


print('Libraries imported. Starting to load the already preprocessed the dataset...')


## Load the already preprocessed dataset

path_to_save_mri_data = '/usr/local/micapollo01/MIC/DATA/STUDENTS/mvhave7/Results/Preprocessing/fully_processed_at_once_augmented/post_augmentation/'

X_train_arr = np.load(path_to_save_mri_data+"augmented_training_data_DeepMRIRec_16_coils.npy")
Y_train_arr = np.load(path_to_save_mri_data+"augmented_training_data_GT_DeepMRIRec_16_coils.npy")


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
loss_weights = [1.0, 0.0001, 0.000001, 0]

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
    
    ssim_loss = 1- tf.math.abs(tf.reduce_mean(tf.image.ssim(img1=y_true,img2=y_pred,max_val=1.0,filter_size=3,filter_sigma=0.1)))
    pixel_loss = tf.reduce_mean(tf.math.abs(y_true-y_pred))
    
    content_loss = 0.0
    res_y_rss = tf.image.grayscale_to_rgb(y_true*255)
    res_y_rss = preprocess_input(res_y_rss)
    vgg_f_gt = vgg_model(res_y_rss)

    res_y_pred = tf.image.grayscale_to_rgb(y_pred*255)
    res_y_pred = preprocess_input(res_y_pred)
    vgg_f_pred = vgg_model( res_y_pred)

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

model_name = "/usr/local/micapollo01/MIC/DATA/STUDENTS/mvhave7/Results/Models/best_model_DeepMRIRec.h5"

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

model.save_weights("/usr/local/micapollo01/MIC/DATA/STUDENTS/mvhave7/Results/Models/final_model_DeepMRIRec.h5")


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

