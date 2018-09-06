from keras.models import *
from keras.optimizers import SGD, RMSprop
from keras.layers import *
from keras.applications.vgg16 import VGG16
from keras.utils.training_utils import multi_gpu_model
import keras
import h5py
import numpy as np
import scipy.io as io
import tensorflow as tf
import argparse

def decoder(lstm_activation=None, optimizer=None, weights=None):
    print "Setting up decoder"
    # Decoder -------------------------------------------
    # 1. Scanpath input
    main_input = Input(shape=(63,4))
    x = LSTM(500, input_shape=(63,4), activation=lstm_activation, return_sequences=True)(main_input)
    x = BatchNormalization()(x)

    # 2. Image input
    aux_input = Input(shape=(224, 224, 3))
    vgg = VGG16(weights='imagenet', include_top=False, input_tensor=aux_input)
    vgg.trainable=False
    z = vgg.output
    z = Conv2D(100, (3,3), activation='linear')(z)
    z = LeakyReLU(alpha=0.3)(z)
    z = Flatten()(z) #1600
    z = RepeatVector(63)(z)
    z = Reshape((63,2500))(z)

    # 3. Merge
    x = keras.layers.concatenate([x,z])
    x = LSTM(100, activation=lstm_activation, return_sequences=True)(x)
    x = BatchNormalization()(x)
    x = LSTM(100, activation=lstm_activation, return_sequences=True)(x)
    x = BatchNormalization()(x)
    x = LSTM(100, activation=lstm_activation, return_sequences=True)(x)
    x = BatchNormalization()(x)
    output = LSTM(1, activation='sigmoid', return_sequences=True)(x)

    decoder = Model(inputs=[main_input, aux_input], outputs=output)

    # Don't train VGG layers
    for i in range(19):
        #print decoder.layers[i]
        decoder.layers[i].trainable = False

    decoder.compile(optimizer=optimizer, loss='binary_crossentropy', sample_weight_mode='temporal', metrics=['accuracy'])


    if weights != "-":
        print("Loading discriminator weights")
        decoder.load_weights(weights)

    return decoder




def generator(n_hidden_gen=None, lstm_activation=None, dropout=None, optimizer=None, loss=None, weights=None, G=None, loss_weights=None):
    # Encoder -------------------------------------------
    print "Setting up generator"

    generator = Sequential()
    main_input = Input(shape=(224, 224, 3)) 
    vgg = VGG16(weights='imagenet', include_top=False, input_tensor=main_input)
    vgg.trainable=False

    generator.add(vgg)
    generator.add(BatchNormalization())
    generator.add(Flatten())
    generator.add(Dense(n_hidden_gen, activation='linear'))
    generator.add(LeakyReLU(alpha=0.3))
    generator.add(RepeatVector(63))
    generator.add(Reshape((63, n_hidden_gen)))
    generator.add(LSTM(n_hidden_gen, activation=lstm_activation, return_sequences=True, dropout=dropout, recurrent_dropout=dropout,))
    generator.add(BatchNormalization())
    generator.add(LSTM(n_hidden_gen, activation=lstm_activation, return_sequences=True, dropout=dropout, recurrent_dropout=dropout,))
    generator.add(BatchNormalization())
    generator.add(LSTM(n_hidden_gen, activation=lstm_activation, return_sequences=True, dropout=dropout, recurrent_dropout=dropout,))
    generator.add(BatchNormalization())
    generator.add(Dense(4, activation='sigmoid'))
    
    # Compile
    generator.compile(loss=loss, optimizer=optimizer, sample_weight_mode='temporal', metrics=['mae'], loss_weights=loss_weights)

    # Load weights
    if weights != "-":
        print("Loading generator weights")
        generator.load_weights(weights)


    if G > 1 : 
        generator_parallel = multi_gpu_model(generator, gpus=G)
        generator_parallel.compile(loss=loss, optimizer=optimizer, sample_weight_mode='temporal', metrics=['mae'])
    else: 
        generator_parallel = generator 

    return generator, generator_parallel


def gen_dec(content_loss=None, optimizer=None, loss_weights=None, generator=None, decoder=None, G=None, shape=(224, 224, 3)):

    print "Setting up combined net"
    generator_input = Input(shape=shape)
    dec_img_input = Input(shape=shape)

    generator.name = 'generator'
    x = generator(generator_input)

    decoder.name = 'discriminator'
    decoder.trainable=False

    output = decoder([x, dec_img_input])

    gen_dec = Model(inputs=[generator_input, dec_img_input], outputs=[output, x])

    gen_dec.compile(
                    loss=['binary_crossentropy', content_loss], 
                    optimizer=optimizer, 
                    sample_weight_mode='temporal',
                    metrics=['accuracy', 'mae'],
                    loss_weights=loss_weights
                   )


    if G > 1:
        gen_dec_parallel = multi_gpu_model(gen_dec, gpus=G)
        gen_dec_parallel.compile(
                                loss=['binary_crossentropy', content_loss], 
                                optimizer=optimizer, 
                                sample_weight_mode='temporal',
                                metrics=['accuracy', 'mae'],
                                loss_weights=loss_weights
                               )
    else:
        gen_dec_parallel = gen_dec

    return gen_dec, gen_dec_parallel
