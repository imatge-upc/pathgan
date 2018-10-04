import models
from keras.models import *
from keras.optimizers import SGD, RMSprop
from keras.layers import *
from keras.applications.vgg16 import VGG16
from time import gmtime, strftime
from keras.utils.training_utils import multi_gpu_model
import keras
import h5py
import numpy as np
import scipy.io as io
import tensorflow as tf
import argparse
import os
import utils

def accumulate_times(times):
    new_times = np.copy(times)
    for i in range(len(times)):
        new_times[i] = np.sum(times[:i+1])
    new_times = new_times - new_times[0]
    return new_times
    

def prepare_sp(scanpath, user_id):
    # scanpath => [x, y, EOS, t]
    # Outputs a matrix with
    # amat => [user_id, fix_id, time, x, y]
    indices = scanpath[:, 2] < 0.5 
    x = scanpath[indices, :3]
    amat = np.zeros((x.shape[0], 5))
    amat[:,0] = user_id
    amat[:,3:] = x[:, :2]
    amat[:,1] = np.arange(x.shape[0]) + 1 # 1,2,3,4... indexes
    a = scanpath[indices, 3]
    a *= 500        # Restore the normlaization. Sigmoid is 0..1, but values should be 0..500
    a = accumulate_times(a)
    a[a < 0] = 0
    amat[:,2] = a
    # amat = np.round(amat)
    return amat

# [10, 63, 1]
def prepare_image(scanpaths):
    a = []
    for i in range(scanpaths.shape[0]):
      if i == 0:
        a = prepare_sp(scanpaths[i,:,:], i+1)
      else:
        a = np.vstack([a, prepare_sp(scanpaths[i,:,:], i+1)])
    return a

def predict(img_path):
    # example pathgan.predict('/root/sharedfolder/Images/P1.jpg')

    loss_weights            = [1., 0.05] #0.05
    adversarial_iteration   = 2
    batch_size              = 40 #100
    mini_batch_size         = 800 #4000
    G                       = 1
    epochs                  = 200
    n_hidden_gen            = 1000
    lr                      = 1e-4
    content_loss            = 'mse'
    lstm_activation         = 'tanh'
    dropout                 = 0.1
    dataset_path            = '/root/sharedfolder/predict_scanpaths/finetune_saltinet_isun/input/salient360_EVAL_noTime.hdf5'
    model360                = 'false'
    weights_generator       = '../weights/generator_single_weights.h5'
    opt = RMSprop(lr=lr, rho=0.9, epsilon=1e-08, decay=0.0)


    # Load image 
    img, img_size = utils.load_image(img_path, 'small_images')
    
    # Get the model
    params = {
        'n_hidden_gen':n_hidden_gen,
        'lstm_activation':lstm_activation,
        'dropout':dropout,
        'optimizer':opt,
        'loss':content_loss,
        'weights':weights_generator,
        'G':G
    }
    _, generator_parallel = models.generator(**params)

    # Predict with a model
    n_sp_per_image = 1 

    #provisional
    output = np.zeros((n_sp_per_image, 63, 4 ))
    for i in range(n_sp_per_image):
        print("Calculating observer %d" % i)
        noise  = np.random.normal(0,3,img.shape)
        noisy_img = img + noise
        prediction = generator_parallel.predict(noisy_img)
        output[i] = prediction

    # Prepare the predictions for matlab and save it on individual files
    output = prepare_image(output[:,:,:])

    return output


def save_scanpath_as_csv(scanpaths, out_path, in_path):

    #name = "_".join(in_path.split(os.sep)[-1].split("_")[:2]).split(".")[0]
    name = in_path.split('/')[-1].split('.')[0]
    out_path = out_path + '%s.csv' % name

    with open(out_path, "w") as saveFile:
        saveFile.write("Idx, longitude, latitude, start timestamp\n")
        for i in range(scanpaths.shape[0]):
            idx = scanpaths[i, 1]
            lon = scanpaths[i, 3]
            lat = scanpaths[i, 4]
            tim = scanpaths[i, 2]
            saveFile.write("{}, {}, {}, {},\n".format(
                int(idx), lon, lat, tim
                )
            )
        print('Saved scanpaths from image %s in file %s' % (in_path, out_path))

def predict_and_save(imgs_path, out_path):
    """ 
        Predicts multiple images and saves them in .mat format
        on an output path

        Param:
            img_path : path where the images are
            out_path: path where the .mat files will be saved

        i.e.:
            img_path = '/root/sharedfolder/360Salient/'
            out_path =  '/root/sharedfolder/360Salient/results/'
    """

    # Preproces and load images

    paths = utils.file_paths_for_images(imgs_path)

    for i, path in enumerate(paths):
        print('Working on image %d of %d' % (i+1, len(paths)))

        # Predict the scanpaths
        scanpaths = predict(path)

        # Turn into a float np.array
        scanpaths = np.array(scanpaths, dtype=np.float32)

        # Save in output folder
        save_scanpath_as_csv(scanpaths, out_path, path)
        # To save it in .mat format
        # name = "_".join(path.split(os.sep)[-1].split("_")[:2]).split(".")[0]
        # io.savemat(out_path + '/%s.mat' % name, {name: scanpaths})


    print('Done!')
    return True
