import os
import sys
import time
import numpy as np
import pandas as pd
from keras.models import Model, Sequential
from keras.layers import Convolution1D, AtrousConvolution1D, Dense, Input, Flatten, Lambda, merge, Activation
import keras


def wavenetBlock(n_atrous_filters, atrous_filter_size, atrous_rate):
    """

    @param input_size:
    @return:
    """
    def f(input_):
        residual = input_
        tanh_out = AtrousConvolution1D(n_atrous_filters, atrous_filter_size,
                                       atrous_rate=atrous_rate,
                                       border_mode='same',
                                       activation='tanh')(input_)
        sigmoid_out = AtrousConvolution1D(n_atrous_filters, atrous_filter_size,
                                          atrous_rate=atrous_rate,
                                          border_mode='same',
                                          activation='sigmoid')(input_)
        merged = keras.layers.Multiply()([tanh_out, sigmoid_out])
        skip_out = Convolution1D(1, 1, activation='relu', border_mode='same')(merged)
        out = keras.layers.Add()([skip_out, residual])
        return out, skip_out
    return f


def get_basic_generative_model(input_size):
    """

    @param input_size:
    @return:
    """
    input_ = Input(shape=(input_size, 1))
    A, B = wavenetBlock(64, 2, 2)(input_)
    skip_connections = [B]
    # the hidden layers of generative model
    for i in range(10):
        A, B = wavenetBlock(64, 2, 2**((i+2) % 9))(A)
        skip_connections.append(B)

    net = keras.layers.Add()(skip_connections)
    net = Activation('relu')(net)
    net = Convolution1D(1, 1, activation='relu')(net)
    net = Convolution1D(1, 1)(net)
    net = Flatten()(net)
    net = Dense(256, activation='softmax')(net)
    model = Model(input=input_, output=net)
    model.compile(loss='categorical_crossentropy', optimizer='sgd',
                  metrics=['accuracy'])
    model.summary()
    return model


def frame_generator(audio, frame_size, frame_shift, minibatch_size=32):
    """

    @param audio:
    @param frame_size:
    @param frame_shift:
    @param minibatch_size:
    """
    audio_len = len(audio)
    X = []
    y = []
    while 1:
        for i in range(0, audio_len - frame_size - 1, frame_shift):
            frame = audio[i:i+frame_size]
            if len(frame) < frame_size:
                break
            if i + frame_size >= audio_len:
                break
            temp = audio[i + frame_size]
            target_val = int((np.sign(temp) * (np.log(1 + 256*abs(temp)) / (
                np.log(1+256))) + 1)/2.0 * 255)
            X.append(frame.reshape(frame_size, 1))
            y.append((np.eye(256)[target_val]))
            if len(X) == minibatch_size:
                yield np.array(X), np.array(y)
                X = []
                y = []


def get_load_from_model(model, length, seed_audio):
    """

    @param model:
    @param length:
    @param seed_audio:
    @return:
    """
    print('Generating audio...')
    new_audio = np.zeros((length))
    curr_sample_idx = 0
    while curr_sample_idx < new_audio.shape[0]:
        distribution = np.array(model.predict(seed_audio.reshape(1, frame_size, 1)), dtype=float).reshape(256)
        distribution /= distribution.sum().astype(float)
        predicted_val = np.random.choice(range(256), p=distribution)
        ampl_val_8 = (((predicted_val / 255.0) - 0.5) * 2.0)
        ampl_val_16 = (np.sign(ampl_val_8) * (1/256.0) * ((1 + 256.0)**abs(
            ampl_val_8) - 1)) * 2**15
        new_audio[curr_sample_idx] = ampl_val_16
        seed_audio[:-1] = seed_audio[1:]
        seed_audio[-1] = ampl_val_16
        pc_str = str(round(100*curr_sample_idx/float(new_audio.shape[0]), 2))
        curr_sample_idx += 1
    print('Audio generated.')
    return new_audio.astype(np.int16)


if __name__ == '__main__':

    n_epochs = 2000
    frame_size = 96*7
    frame_shift = 1
    mini_batches =32

    # -------------------------
    # read the data
    # -------------------------

    path_training = './cleaned_data/accumulate_frame/Austin_from_Jan_to_June.csv'
    path_validation = './cleaned_data/accumulate_frame/Boulder_from_Jan_to_June.csv'

    df_train = pd.read_csv(path_training)
    df_validation = pd.read_csv(path_validation)
    df_train = df_train.drop(columns=['local_15min', 'avg'])
    df_validation = df_validation.drop(columns=['local_15min', 'avg'])
    np_train_1 = df_train.transpose().to_numpy().flatten()
    np_train = (np_train_1 - np_train_1.min())/(np_train_1.max()-np_train_1.min())*2 - 1
    np_validation_1 = df_validation.transpose().to_numpy().flatten()
    np_validation = (np_validation_1 - np_validation_1.min())/(np_validation_1.max()-np_validation_1.min())*2 - 1

    # -------------------------
    # building the model
    # -------------------------

    model = get_basic_generative_model(frame_size)

    # -------------------------
    # training
    # -------------------------

    train_data_gen = frame_generator(np_train, frame_size, frame_shift, mini_batches)
    model.fit_generator(train_data_gen, steps_per_epoch=3000, epochs=n_epochs)

    model.save('./models/model_{}.h5'.format(n_epochs))

    # -------------------------
    # generate the data
    # -------------------------
    generate_context = np_validation[0:frame_size]
    new_load = get_load_from_model(model, np_validation.shape[0], generate_context)
