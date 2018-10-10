"""
Discriminative models
"""
from __future__ import print_function
from __future__ import absolute_import
import numpy as np
import tensorflow as tf
import livelossplot
from tensorflow import get_default_graph, Session
from tensorflow.keras.backend import set_session, get_session, clear_session
from tensorflow.keras.models import Sequential, Model as KerasModel
from tensorflow.keras.layers import (Dense, Dropout, Activation, Flatten,
                                     Conv2D, ConvLSTM2D, MaxPooling2D, Conv3D,
                                     Input, MaxPooling3D, ZeroPadding3D,
                                     CuDNNLSTM, LSTM, CuDNNGRU, concatenate,
                                     SpatialDropout3D, SpatialDropout2D,
                                     AveragePooling2D, AveragePooling3D,
                                     TimeDistributed, Bidirectional,
                                     BatchNormalization)
from tensorflow.keras.optimizers import Adam, RMSprop, Adadelta, SGD
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.utils import multi_gpu_model
# from keras.regularizers import l2, l1, l1_l2
# from keras.layers.advanced_activations import PReLU
from vip_hci.conf import time_ini, timing, time_fin
from sklearn.model_selection import train_test_split


def train_4dnet(X, Y, test_size=0.1, validation_split=0.1, random_state=0,
                layer_type=('conv3d', 'conv3d'), nconvlayers=2,
                conv_nfilters=(40, 80), kernel_sizes=((3, 3, 3), (2, 2, 2)),
                conv_strides=((1, 1, 1), (1, 1, 1)), conv_padding='same',
                dilation_rate=((1, 1, 1), (1, 1, 1)), pool_layers=2,
                pool_func='ave', pool_sizes=((2, 2, 2), (2, 2, 2)),
                pool_strides=((2, 2, 2), (2, 2, 2)), dense_units=128,
                rec_hidden_states=64, activation='relu', learnrate=0.003,
                batchsize=64, epochs=20, patience=2, min_delta=0.01,
                retrain=None, verb=1, summary=True, gpu_id='0',
                plot='tb', tb_path='./logs', full_output=False):
    """
    """
    clear_session()
    graph = get_default_graph()
    config = tf.ConfigProto()
    # Don't pre-allocate memory; allocate as-needed
    config.gpu_options.allow_growth = True
    # Only allow a total of half the GPU memory to be allocated
    # config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.gpu_options.visible_device_list = gpu_id
    session = tf.Session(graph=graph, config=config)
    set_session(session)
    ngpus = len(gpu_id.split(','))
    batchsize *= ngpus

    if not nconvlayers == len(conv_nfilters):
        raise ValueError('`conv_nfilters` has a wrong length')
    if not nconvlayers == len(kernel_sizes):
        raise ValueError('`kernel_sizes` has a wrong length')
    if not nconvlayers == len(conv_strides):
        raise ValueError('`conv_strides` has a wrong length')
    if not nconvlayers == len(dilation_rate):
        raise ValueError('`dilation_rate` has a wrong length')

    if pool_layers > 0:
        if not pool_layers == len(pool_sizes):
            raise ValueError('`pool_sizes` has a wrong length')
        if pool_strides is not None:
            if not pool_layers == len(pool_strides):
                raise ValueError('`pool_strides` has a wrong length')
        else:
            pool_strides = [None] * pool_layers

    if isinstance(layer_type, str):
        layer_type = [layer_type for _ in range(nconvlayers)]

    starttime = time_ini()

    # Mixed train/test sets with Sklearn split
    res = train_test_split(X, Y, test_size=test_size, random_state=random_state)
    X_train, X_test, y_train, y_test = res
    msg = 'Zeros in train: {} |  Ones in train: {}'
    print(msg.format(y_train.tolist().count(0), y_train.tolist().count(1)))
    msg = 'Zeros in test: {} |  Ones in test: {}'
    print(msg.format(y_test.tolist().count(0), y_test.tolist().count(1)))

    # adding the "channels" dimension (1)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1],
                              X_train.shape[2], X_train.shape[3],
                              X_train.shape[4], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2],
                            X_test.shape[3], X_train.shape[4], 1)

    print("\nShapes of train and test:")
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape, '\n')
    # --------------------------------------------------------------------------

    kernel_init = 'glorot_uniform'
    bias_init = 'random_normal'
    rec_act = 'hard_sigmoid'
    rec_init = 'orthogonal'
    temp_dim = X_train.shape[1]
    k_dim = X_train.shape[2]
    patch_size = X_train.shape[3]
    input_shape_3d = (temp_dim, patch_size, patch_size, 1)

    if pool_func == 'ave':
        pooling_func = AveragePooling3D
    elif pool_func == 'max':
        pooling_func = MaxPooling3D

    # --------------------------------------------------------------------------
    # Per branch model
    # --------------------------------------------------------------------------
    input_layer = Input(shape=input_shape_3d, name='input_layer',
                        dtype='float32')

    for i in range(nconvlayers):
        # Stack of Conv3d, (B)CLSTM, (B)LRCN or (B)GRCN layers
        if layer_type[i] in ('conv3d', 'clstm', 'bclstm'):
            if pool_func == 'ave':
                pooling_func = AveragePooling3D
            elif pool_func == 'max':
                pooling_func = MaxPooling3D
        elif layer_type[i] in ('lrcn', 'blrcn', 'grcn', 'bgrcn'):
            if pool_func == 'ave':
                pooling_func = AveragePooling2D
            elif pool_func == 'max':
                pooling_func = MaxPooling2D
        else:
            raise ValueError('pool_func is not recognized')

        if layer_type[i] == 'conv3d':
            if not len(kernel_sizes[i]) == 3:
                raise ValueError(
                    'Kernel sizes for Conv3d are tuples of 3 values')
            if not len(conv_strides[i]) == 3:
                raise ValueError('Strides for Conv3d are tuples of 3 values')
            if not len(dilation_rate[i]) == 3:
                raise ValueError('Dilation for Conv3d is a tuple of 3 values')

            if i == 0:
                x = Conv3D(filters=conv_nfilters[i],
                           kernel_size=kernel_sizes[i],
                           strides=conv_strides[i],
                           padding=conv_padding,
                           kernel_initializer=kernel_init,
                           bias_initializer=bias_init,
                           name='conv3d_layer1',
                           dilation_rate=dilation_rate[i],
                           data_format='channels_last',
                           input_shape=input_shape_3d)(input_layer)

                x = SpatialDropout3D(0.5)(x)
                x = Activation(activation, name='activ_layer1')(x)
                x = pooling_func(pool_size=pool_sizes[i],
                                 strides=pool_strides[i], padding='valid')(x)

            else:
                x = Conv3D(filters=conv_nfilters[i],
                           kernel_size=kernel_sizes[i],
                           strides=conv_strides[i],
                           padding=conv_padding,
                           kernel_initializer=kernel_init,
                           bias_initializer=bias_init,
                           name='conv3d_layer' + str(i+1),
                           dilation_rate=dilation_rate[i])(x)

                x = SpatialDropout3D(0.25)(x)
                x = Activation(activation, name='activ_layer' + str(i+1))(x)
                x = pooling_func(pool_size=pool_sizes[i],
                                 strides=pool_strides[i], padding='valid')(x)

        elif layer_type[i] == 'clstm':
            msg = 'are tuples of 2 integers'
            if not len(kernel_sizes[0]) == 2:
                raise ValueError('Kernel sizes for ConvLSTM' + msg)
            if not len(conv_strides[0]) == 2:
                raise ValueError('Strides for ConvLSTM')
            if not len(dilation_rate[0]) == 2:
                raise ValueError('Dilation rates for ConvLSTM')

            if i == 0:
                x = ConvLSTM2D(filters=conv_nfilters[i],
                               kernel_size=kernel_sizes[i],
                               strides=conv_strides[i], padding=conv_padding,
                               kernel_initializer=kernel_init,
                               input_shape=input_shape_3d,
                               name='convlstm_layer1',
                               return_sequences=True,
                               dilation_rate=dilation_rate[i],
                               activation='tanh',
                               recurrent_activation=rec_act,
                               use_bias=True, recurrent_initializer=rec_init,
                               bias_initializer='zeros',
                               unit_forget_bias=True, dropout=0.0,
                               recurrent_dropout=0.0)(input_layer)

                x = SpatialDropout3D(0.5)(x)

                x = pooling_func(pool_size=pool_sizes[i],
                                 strides=pool_strides[i], padding='valid')(x)

            else:
                x = ConvLSTM2D(filters=conv_nfilters[i],
                               kernel_size=kernel_sizes[i],
                               strides=conv_strides[i], padding=conv_padding,
                               kernel_initializer=kernel_init,
                               name='convlstm_layer' + str(i+1),
                               return_sequences=True,
                               dilation_rate=dilation_rate[i],
                               activation='tanh',
                               recurrent_activation=rec_act,
                               use_bias=True, recurrent_initializer=rec_init,
                               bias_initializer='zeros',
                               unit_forget_bias=True,
                               dropout=0.0, recurrent_dropout=0.0)(x)

                x = SpatialDropout3D(0.25)(x)
                x = pooling_func(pool_size=pool_sizes[1],
                                 strides=pool_strides[1], padding='valid')(x)

        elif layer_type[i] in ('lrcn', 'blrcn', 'grcn', 'bgrcn'):
            if not len(kernel_sizes[i]) == 2:
                raise ValueError(
                    'Kernel sizes for LRCN are tuples of 2 values')
            if not len(conv_strides[i]) == 2:
                raise ValueError('Strides for LRCN are tuples of 2 values')
            if not len(dilation_rate[i]) == 2:
                raise ValueError('Dilation for LRCN is a tuple of 2 values')

            # TimeDistributed wrapper applies a layer to every temporal
            # slice of an input. The input should be at least 3D and the
            # dimension of index one will be considered to be the temporal
            # dimension.
            if i == 0:
                x = TimeDistributed(Conv2D(filters=conv_nfilters[i],
                                           kernel_size=kernel_sizes[i],
                                           strides=conv_strides[i],
                                           padding=conv_padding,
                                           name='lrcn_layer1',
                                           activation=activation),
                                    input_shape=input_shape_3d)(input_layer)

                # This version performs the same function as Dropout, however it
                # drops entire 2D feature maps instead of individual elements.
                x = TimeDistributed(SpatialDropout2D(0.5))(x)

                x = TimeDistributed(pooling_func(pool_size=pool_sizes[i],
                                                 strides=pool_strides[i],
                                                 padding='valid'))(x)

            else:
                x = TimeDistributed(Conv2D(filters=conv_nfilters[1],
                                           kernel_size=kernel_sizes[1],
                                           strides=conv_strides[1],
                                           padding=conv_padding,
                                           name='lrcn_layer' + str(i+1),
                                           activation=activation))(x)

                x = TimeDistributed(SpatialDropout2D(0.25))(x)

                x = TimeDistributed(pooling_func(pool_size=pool_sizes[1],
                                                 strides=pool_strides[1],
                                                 padding='valid'))(x)

    # Final layer
    if layer_type[-1] in ('conv3d', 'clstm'):
        flatten_layer = Flatten(name='flatten')(x)

        # Fully-connected or dense layer
        output = Dense(units=dense_units, name='dense_layer')(flatten_layer)
        output = Activation(activation, name='activ_dense')(output)
        output = Dropout(rate=0.5, name='dropout_dense')(output)

    elif layer_type[-1] in ('lrcn', 'blrcn', 'grcn', 'bgrcn'):
        output = TimeDistributed(Flatten(name='flatten'))(x)
        output = Dropout(0.5, name='dropout_flatten')(output)

    model_branch = KerasModel(inputs=input_layer, outputs=output)
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    # Multi-input model
    # --------------------------------------------------------------------------
    inputs = []
    outputs = []
    for i in range(k_dim):
        input_ = Input(shape=input_shape_3d, name='input_' + str(i+1),
                       dtype='float32')
        output_ = model_branch(input_)
        inputs.append(input_)
        outputs.append(output_)

    # Concatenating the branches. Shape [samples, time steps, features*k_dim]
    concatenated = concatenate(outputs)

    if layer_type[1] == 'lrcn':
        lstm = CuDNNLSTM(rec_hidden_states, kernel_initializer=kernel_init,
                         return_sequences=False)(concatenated)
        concatenated = Dropout(0.5, name='dropout_lstm')(lstm)
    elif layer_type[1] == 'blrcn':
        blstm = Bidirectional(LSTM(rec_hidden_states,  # TODO: bug CuDNNLSTM?
                                   kernel_initializer=kernel_init,
                                   return_sequences=False))(concatenated)
        concatenated = Dropout(0.5, name='dropout_lstm')(blstm)
    elif layer_type[1] == 'grcn':
        gru = CuDNNGRU(rec_hidden_states, kernel_initializer=kernel_init,
                       return_sequences=False)(concatenated)
        concatenated = Dropout(0.5, name='dropout_gru')(gru)
    elif layer_type[1] == 'bgrcn':
        bgru = Bidirectional(CuDNNGRU(rec_hidden_states,
                                      kernel_initializer=kernel_init,
                                      return_sequences=False))(concatenated)
        concatenated = Dropout(0.5, name='dropout_gru')(bgru)

    # Sigmoid unit
    prediction = Dense(units=1, name='sigmoid_output_unit',
                       activation='sigmoid')(concatenated)

    model_final = KerasModel(inputs=inputs, outputs=prediction)
    # --------------------------------------------------------------------------

    if summary:
        model_branch.summary()
        # model_final.summary()

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

    X_train = list(np.moveaxis(X_train, 2, 0))
    X_test = list(np.moveaxis(X_test, 2, 0))

    # Training the network
    if ngpus > 1:
        # Multi-GPUs
        Mpar = multi_gpu_model(model_final, gpus=ngpus)
        # Training the network
        Mpar.compile(loss='binary_crossentropy', metrics=['accuracy'],
                     optimizer=Adam(lr=learnrate, decay=1e-2))
        hist = Mpar.fit(X_train, y_train, batch_size=batchsize, shuffle=True,
                        epochs=epochs, initial_epoch=0, verbose=verb,
                        validation_split=validation_split, callbacks=callbacks)
        score = Mpar.evaluate(X_test, y_test, verbose=verb)

    else:
        # Single GPU
        model_final.compile(loss='binary_crossentropy', metrics=['accuracy'],
                            optimizer=Adam(lr=learnrate, decay=1e-2))
        hist = model_final.fit(X_train, y_train, batch_size=batchsize,
                               epochs=epochs, initial_epoch=0, verbose=verb,
                               validation_split=validation_split,
                               callbacks=callbacks, shuffle=True)
        score = model_final.evaluate(X_test, y_test, verbose=verb)

    print('\nTest score/loss:', score[0], '\nTest accuracy:', score[1])

    timing(starttime)
    fintime = time_fin(starttime)

    if full_output:
        return model_final, hist.history, score, fintime
    else:
        return model_final


