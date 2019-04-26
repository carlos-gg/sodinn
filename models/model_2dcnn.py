"""
Discriminative models
"""
from __future__ import absolute_import
from __future__ import print_function

import livelossplot
import numpy as np
import tensorflow as tf
from keras.backend import set_session, clear_session
from keras.callbacks import EarlyStopping, TensorBoard
from keras.layers import (Dense, Dropout, Activation, Flatten,
                          Conv2D, MaxPooling2D)
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
from sklearn.model_selection import train_test_split
# from keras.regularizers import l2, l1, l1_l2
# from keras.layers.advanced_activations import PReLU
from vip_hci.conf import time_ini, timing, time_fin


def train_2dconvnet(X, Y, test_size=0.1, validation_split=0.1, random_state=0,
                    pseudo3d=False, nconvlayers=2, conv_nfilters=(40, 80),
                    kernel_sizes=((3, 3), (3, 3)),
                    conv_strides=((1, 1), (1, 1)),
                    pool_layers=2, pool_sizes=((2, 2), (2, 2)),
                    pool_strides=((2, 2), (2, 2)), dense_units=128,
                    activation='relu', learnrate=0.003, batchsize=64, epochs=20,
                    patience=2, min_delta=0.01, retrain=None, verb=1,
                    summary=True, gpu_id='0', full_output=False,
                    plot='tb', tb_path='./logs'):
    """ 2D Convolutional network for pairwise subtracted patches.

    Parameters
    ----------
    ...

    Notes
    -----
    Multi-GPU with Keras:
    https://keras.io/utils/#multi_gpu_model

    """
    clear_session()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = gpu_id
    set_session(tf.Session(config=config))
    ngpus = len(gpu_id.split(','))

    if not nconvlayers == len(conv_nfilters):
        raise ValueError('`conv_nfilters` has a wrong length')
    if not nconvlayers == len(kernel_sizes):
        raise ValueError('`kernel_sizes` has a wrong length')
    if not nconvlayers == len(conv_strides):
        raise ValueError('`conv_strides` has a wrong length')

    if pool_layers > 0:
        if not pool_layers == len(pool_sizes):
            raise ValueError('`pool_sizes` has a wrong length')
        if pool_strides is not None:
            if not pool_layers == len(pool_strides):
                raise ValueError('`pool_strides` has a wrong length')
        else:
            pool_strides = [None] * pool_layers
    if pseudo3d:
        if not X.ndim == 4:
            raise ValueError('X must contain 4D samples')

    starttime = time_ini()
    patch_size = X.shape[-1]

    # Mixed train/test sets with Sklearn split
    resplit = train_test_split(X, Y, test_size=test_size,
                               random_state=random_state)
    X_train, X_test, y_train, y_test = resplit
    msg = 'Zeros in train: {} |  Ones in train: {}'
    print(msg.format(y_train.tolist().count(0), y_train.tolist().count(1)))
    msg = 'Zeros in test: {} |  Ones in test: {}'
    print(msg.format(y_test.tolist().count(0), y_test.tolist().count(1)))

    if pseudo3d:
        if not X.ndim == 4:
            raise ValueError('`X` has wrong number of dimensions')
        # moving the temporal dimension to the channels dimension (last)
        X_train = np.moveaxis(X_train, 1, -1)
        X_test = np.moveaxis(X_test, 1, -1)
        input_shape = (patch_size, patch_size, X_train.shape[-1])
    else:
        # adding the channels dimension
        X_train = X_train.reshape(X_train.shape[0], patch_size, patch_size, 1)
        X_test = X_test.reshape(X_test.shape[0], patch_size, patch_size, 1)
        input_shape = (patch_size, patch_size, 1)

    print("\nShapes of train and test sets:")
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape, '\n')

    # --------------------------------------------------------------------------
    if retrain is not None:
        M = retrain
        # re-training the network
        M.compile(loss='binary_crossentropy', metrics=['accuracy'],
                  optimizer=Adam(lr=learnrate, decay=1e-2))
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience,
                                       min_delta=min_delta, verbose=verb)

        hist = M.fit(X_train, y_train, batch_size=batchsize, epochs=epochs,
                     initial_epoch=0, verbose=verb, validation_split=0.1,
                     callbacks=[early_stopping], shuffle=True)

        score = M.evaluate(X_test, y_test, verbose=verb)
        print('\nTest score/loss:', score[0], '\n',
              'Test accuracy:', score[1])

        timing(starttime)
        fintime = time_fin(starttime)

        if full_output:
            return M, hist.history, score, fintime
        else:
            return M
    # --------------------------------------------------------------------------

    # Creating the NN model
    if ngpus > 1:
        with tf.device('/cpu:0'):
            M = Sequential()
    else:
        M = Sequential()

    # Stack of 2d convolutional layers
    kernel_init = 'glorot_uniform'
    bias_init = 'random_normal'

    for i in range(nconvlayers):
        if i == 0:
            M.add(Conv2D(filters=conv_nfilters[i], kernel_size=kernel_sizes[i],
                         strides=conv_strides[i], padding='same',
                         kernel_initializer=kernel_init,
                         bias_initializer=bias_init,
                         name='conv2d_layer1', data_format='channels_last',
                         input_shape=input_shape))
        else:
            M.add(Conv2D(filters=conv_nfilters[i], kernel_size=kernel_sizes[i],
                         strides=conv_strides[i], padding='same',
                         kernel_initializer=kernel_init,
                         bias_initializer=bias_init,
                         name='conv2d_layer' + str(i+1)))

        M.add(Activation(activation, name='activ_layer' + str(i+1)))

        if pool_layers != 0:
            M.add(MaxPooling2D(pool_size=pool_sizes[i], strides=pool_strides[i],
                               padding='valid'))
            pool_layers -= 1

        M.add(Dropout(rate=0.25, name='dropout_layer' + str(i+1)))

    M.add(Flatten(name='flatten'))

    # Dense or fully-connected layer
    M.add(Dense(units=dense_units, name='dense_128units'))
    M.add(Activation(activation, name='activ_dense'))
    # M.add(BatchNormalization())
    M.add(Dropout(rate=0.5, name='dropout_dense'))

    M.add(Dense(units=1, name='dense_1unit'))
    M.add(Activation('sigmoid', name='activ_out'))

    if summary:
        M.summary()

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience,
                                   min_delta=min_delta, verbose=verb)
    if plot is not None:
        if plot == 'tb':
            tensorboard = TensorBoard(log_dir=tb_path, histogram_freq=1,
                                      write_graph=True, write_images=True)
            callbacks = [early_stopping, tensorboard]
        elif plot == 'llp':
            plotlosses = livelossplot.PlotLossesKeras()
            callbacks = [early_stopping, plotlosses]
        else:
            raise ValueError("`plot` method not recognized")
    else:
        callbacks = [early_stopping]

    # Multi-GPUs
    if ngpus > 1:
        Mpar = multi_gpu_model(M, gpus=ngpus)
        # Training the network
        Mpar.compile(loss='binary_crossentropy', metrics=['accuracy'],
                     optimizer=Adam(lr=learnrate, decay=1e-2))
        hist = Mpar.fit(X_train, y_train, batch_size=batchsize * ngpus,
                        epochs=epochs, initial_epoch=0, verbose=verb,
                        validation_split=validation_split,
                        callbacks=callbacks, shuffle=True)
        score = Mpar.evaluate(X_test, y_test, verbose=verb)
    else:
        # Training the network
        M.compile(loss='binary_crossentropy', metrics=['accuracy'],
                  optimizer=Adam(lr=learnrate, decay=1e-2))
        hist = M.fit(X_train, y_train, batch_size=batchsize * ngpus,
                     epochs=epochs, initial_epoch=0, verbose=verb,
                     validation_split=validation_split,
                     callbacks=callbacks, shuffle=True)
        score = M.evaluate(X_test, y_test, verbose=verb)

    print('\nTest score/loss:', score[0], '\n',
          'Test accuracy:', score[1])

    timing(starttime)
    fintime = time_fin(starttime)

    if full_output:
        return M, hist.history, score, fintime
    else:
        return M
