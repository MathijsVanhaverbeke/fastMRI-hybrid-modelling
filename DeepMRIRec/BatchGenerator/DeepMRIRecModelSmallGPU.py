### VERY IMPORTANT CONTRAST WITH DeepMRIRec: DEEPMRIREC IS AN IMAGE SPACE METHOD


## Set up a resource limiter, such that the script doesn't take up more than a certain amount of RAM (normally 40GB is the limit). In that case, an error will be thrown

import resource

resource.setrlimit(resource.RLIMIT_AS, (40_000_000_000, 40_000_000_000))


print('Resource limit set. Importing libraries...')


## Import libraries

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import tensorflow as tf
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Input, BatchNormalization, Conv2D, MaxPooling2D
from tensorflow.keras.layers import PReLU, add, Attention, Dropout
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
import threading
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

#tf.keras.backend.set_floatx('float16')
#tf.keras.mixed_precision.experimental.set_policy('mixed_float16')

tf.keras.backend.set_floatx('float32')
tf.keras.mixed_precision.experimental.set_policy('float32')


crop_size = (12,640,320)


print('Libraries imported. Starting to prepare the dataset...')


## Prepare dataset

path_to_save_mri_data = '/usr/local/micapollo01/MIC/DATA/STUDENTS/mvhave7/Results/Preprocessing/mri_augmented_float32_small_batch/'


file_paths_train = sorted(glob.glob(path_to_save_mri_data+"training_data_DeepMRIRec_16_coils_batch_*.npy"))
file_paths_train_GT = sorted(glob.glob(path_to_save_mri_data+"training_data_GT_DeepMRIRec_16_coils_batch_*.npy"))
file_paths_val = sorted(glob.glob(path_to_save_mri_data+"validation_data_DeepMRIRec_16_coils_batch_*.npy"))
file_paths_val_GT = sorted(glob.glob(path_to_save_mri_data+"validation_data_GT_DeepMRIRec_16_coils_batch_*.npy"))

lock = threading.Lock()

def train_generator(file_paths_train, file_paths_train_GT):
    for file_path_train, file_path_train_GT in zip (file_paths_train, file_paths_train_GT):
        with lock:
            x_train = np.load(file_path_train)
            y_train = np.load(file_path_train_GT)
             
            yield (x_train, y_train)


def validation_generator(file_paths_val, file_paths_val_GT):
    for file_path_val, file_path_val_GT in zip(file_paths_val, file_paths_val_GT):
        with lock:
            x_test = np.load(file_path_val)
            y_test = np.load(file_path_val_GT)

            yield (x_test, y_test)


print('Done. Setting up tensorflow structure to process in batches...')


## Create a .from_generator() object

training_dataset = tf.data.Dataset.from_generator(generator=lambda: train_generator(file_paths_train, file_paths_train_GT), output_shapes=((None, None, None, None), (None, None, None, 1)), output_types=(tf.float32, tf.float32))
validation_dataset = tf.data.Dataset.from_generator(generator=lambda: validation_generator(file_paths_val, file_paths_val_GT), output_shapes=((None, None, None, None), (None, None, None, 1)), output_types=(tf.float32, tf.float32))


print('Done. Building the DeepMRIRec model architecture...')


## Build the model

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

def model_loss_all(y_true, y_pred):
    global vgg_model
    global loss_weights
    global selected_layer_weights_content
    
    # Remove abs() from the ssim loss
    ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(img1=y_true,img2=y_pred,max_val=1.0,filter_size=3,filter_sigma=0.1))
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
        
        # Attention layer
        #layers = Attention()([layers,layers])
        
        skip_layers.append(layers)
        layers = MaxPooling2D((2,2))(layers)
    return layers, skip_layers

def decoder(inp, nlayers, nbasefilters,skip_layers, drop_rate):
    
    layers = inp
    for i in range(nlayers):
        layers = conv_block(layers,nbasefilters*(2**(nlayers-1-i)),drop_rate)
        
        # Attention layer
        #layers=Attention()([layers,layers])
        
        layers=Conv2DTranspose(kernel_size=(2,2),filters=nbasefilters*(2**(nlayers-1-i)),strides=(2,2), padding='same')(layers)
        layers=add([layers,skip_layers.pop()])
    return layers

def create_gen(gen_ip, nlayers, nbasefilters, drop_rate):
    op,skip_layers = encoder(gen_ip,nlayers, nbasefilters,drop_rate)
    op = decoder(op,nlayers, nbasefilters,skip_layers,drop_rate)
    op = Conv2D(1, (3,3), padding = "same")(op)
    # Add sigmoid activation layer to force output images to have pixel values [0,1]
    #op = Activation('sigmoid', dtype='float32')(op)
    op = Activation('sigmoid')(op)
    return Model(inputs=gen_ip,outputs=op)

input_shape = (crop_size[1],crop_size[2],crop_size[0])
input_layer = Input(shape=input_shape)
model = create_gen(input_layer,5,32,0.01)


print('Done. Training the model...')


## Train the model

import math
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

model_name = "/usr/local/micapollo01/MIC/DATA/STUDENTS/mvhave7/Results/Models/ExtensiveSaves/DeepMRIRec_trained_in_small_batches_GPU_epoch{epoch:02d}.h5"

def step_decay(epoch, initial_lrate, drop, epochs_drop):
    return initial_lrate * math.pow(drop, math.floor((1+epoch)/float(epochs_drop)))

def get_callbacks(model_file, learning_rate_drop=0.7,learning_rate_patience=7, verbosity=1):
    callbacks = list()
    #callbacks.append(ModelCheckpoint(model_file, save_best_only=True))
    callbacks.append(ModelCheckpoint(filepath=model_file, save_weights_only=True, save_freq=1))
    callbacks.append(ReduceLROnPlateau(factor=learning_rate_drop, patience=learning_rate_patience,verbose=verbosity))
    callbacks.append(EarlyStopping(verbose=verbosity, patience=30))
    return callbacks

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = create_gen(input_layer,5,32,0.01)
    metrics = tf.keras.metrics.RootMeanSquaredError()
    model.compile(loss=model_loss_all, optimizer= Adam(learning_rate=0.0001),metrics=[metrics] )


history = model.fit(training_dataset,
            epochs=20,  # In their paper, they use 100 epochs
            shuffle=True,
            validation_data=validation_dataset,
            callbacks=get_callbacks(model_name,0.6,10,1))


model.save_weights("/usr/local/micapollo01/MIC/DATA/STUDENTS/mvhave7/Results/Models/ExtensiveSaves/final_model_DeepMRIRec_trained_in_small_batches_GPU.h5")


print("Done. Saved model to disk.")


print('Plotting loss function training curve')


#pd.DataFrame(history.history).plot(figsize=(8,5))
#plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


