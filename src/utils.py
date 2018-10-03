# (c) Copyright 2017 Marc Assens. All Rights Reserved.

__author__ = "Marc Assens"
__version__ = "1.0"



"""
    Utilities with dealing with the dataset
    of 360 Salient Challenge
    API to the 360 salient dataset
"""

import numpy as np
import matplotlib.image as mpimg
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
import glob


def load_image(img_path, input_size):
    img =  mpimg.imread(img_path)
    img_size = (img.shape[0], img.shape[1])

    model_input_sizes ={
        'small_images': (224, 224),
        '360': (448, 896)
    }
    
    img = image.load_img(img_path, target_size=model_input_sizes[input_size])
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x, img_size

    
def file_paths_for_images(path):
    paths = sorted(glob.glob(path+"/*.jpg"))[:200]
    return paths
    

def training_output(mini_batch_size):
    """

    """

    # Data for training
    y_decoder = np.ones((2*mini_batch_size,63,1))
    y_decoder[:mini_batch_size,:,:] = 0.                        #[00000000...11111] [false, true]
    y_gen_dec = np.ones((mini_batch_size,63,1))                 # [11111...111] [true], in fact it is false, 
                                                                # but puting it into true it inverts the gradient
    return y_decoder, y_gen_dec


                                                            
def save_history(G, history, encoder_loss, encoder_acc, encoder_loss_mse, encoder_mae):
    print history

    if G > 1 : 
        encoder_loss.append(history['concatenate_5_loss'][0])
        encoder_acc.append(history['concatenate_5_acc'][0])
        encoder_loss_mse.append(history['concatenate_6_loss'][0])
        encoder_mae.append(history['concatenate_6_mean_absolute_error'][0])
        # encoder_loss.append(history['discriminator_loss'][0])
        # encoder_acc.append(history['discriminator_acc'][0])
        # encoder_loss_mse.append(history['generator_loss'][0])
        # encoder_mae.append(history['generator_mean_absolute_error'][0])
    else:
        encoder_loss.append(history['discriminator_loss'][0])
        encoder_acc.append(history['discriminator_acc'][0])
        encoder_loss_mse.append(history['generator_loss'][0])
        encoder_mae.append(history['generator_mean_absolute_error'][0])

    return encoder_loss, encoder_acc, encoder_loss_mse, encoder_mae
