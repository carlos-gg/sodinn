"""
Discriminative models
"""
from __future__ import absolute_import
from __future__ import print_function

import livelossplot
import tensorflow as tf
from keras.backend import set_session, clear_session
from keras.callbacks import EarlyStopping, TensorBoard
from keras.layers import (Dense, Dropout, Activation, Flatten,
                          Conv2D, ConvLSTM2D, MaxPooling2D, Conv3D,
                          MaxPooling3D, CuDNNLSTM, LSTM, CuDNNGRU,
                          SpatialDropout3D, SpatialDropout2D,
                          AveragePooling2D, AveragePooling3D,
                          TimeDistributed, Bidirectional)
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
from sklearn.model_selection import train_test_split
from tensorflow import get_default_graph
# from keras.regularizers import l2, l1, l1_l2
# from keras.layers.advanced_activations import PReLU
from vip_hci.conf import time_ini, timing, time_fin


# TODO: convert the model to functional Keras API
def train_3dnet(X, Y, test_size=0.1, validation_split=0.1, random_state=0,
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
    """ 3D Convolutional network or Convolutional LSTM network for SODINN-PW
    and SODINN-SVD.

    Parameters
    ----------
    ...
    layer_type : {'conv3d', 'clstm'} str optional
    batchsize :
        Batch size per GPU (no need to increase it when ``ngpus`` > 1).
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
                              X_train.shape[2], X_train.shape[3], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2],
                            X_test.shape[3], 1)

    print("\nShapes of train and test:")
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape, '\n')

    # --------------------------------------------------------------------------
    if retrain is not None:
        M = retrain
        # Training the network
        M.compile(loss='binary_crossentropy', metrics=['accuracy'],
                  optimizer=Adam(lr=learnrate, decay=1e-2))
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience,
                                       min_delta=min_delta, verbose=verb)

        hist = M.fit(X_train, y_train, batch_size=batchsize, epochs=epochs,
                     initial_epoch=0, verbose=verb, validation_split=0.1,
                     callbacks=[early_stopping], shuffle=True)

        score = M.evaluate(X_test, y_test, verbose=verb)
        print('\n Test score/loss:', score[0])
        print(' Test accuracy:', score[1])

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

    kernel_init = 'glorot_uniform'
    bias_init = 'random_normal'
    rec_act = 'hard_sigmoid'
    rec_init = 'orthogonal'
    temp_dim = X_train.shape[1]  # npcs or pw slices
    patch_size = X_train.shape[2]
    input_shape = (temp_dim, patch_size, patch_size, 1)

    # Stack of Conv3d, (B)CLSTM, (B)LRCN or (B)GRCN layers
    for i in range(nconvlayers):
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
                M.add(Conv3D(filters=conv_nfilters[i],
                             kernel_size=kernel_sizes[i],
                             strides=conv_strides[i], padding=conv_padding,
                             kernel_initializer=kernel_init,
                             bias_initializer=bias_init, name='conv3d_layer1',
                             dilation_rate=dilation_rate[i],
                             data_format='channels_last',
                             input_shape=input_shape))
                M.add(SpatialDropout3D(0.5))
            else:
                M.add(Conv3D(filters=conv_nfilters[i],
                             kernel_size=kernel_sizes[i],
                             strides=conv_strides[i], padding=conv_padding,
                             kernel_initializer=kernel_init,
                             dilation_rate=dilation_rate[i],
                             name='conv3d_layer' + str(i + 1)))
                M.add(SpatialDropout3D(0.25))

            M.add(Activation(activation, name='activ_layer' + str(i + 1)))

            if pool_layers != 0:
                M.add(pooling_func(pool_size=pool_sizes[i],
                                   strides=pool_strides[i], padding='valid'))
                pool_layers -= 1

            M.add(Dropout(rate=0.25, name='dropout_layer' + str(i + 1)))

        elif layer_type[i] == 'clstm':
            msg = 'are tuples of 2 integers'
            if not len(kernel_sizes[i]) == 2:
                raise ValueError('Kernel sizes for ConvLSTM' + msg)
            if not len(conv_strides[i]) == 2:
                raise ValueError('Strides for ConvLSTM')
            if not len(dilation_rate[i]) == 2:
                raise ValueError('Dilation rates for ConvLSTM')

            if i == 0:
                M.add(ConvLSTM2D(filters=conv_nfilters[i],
                                 kernel_size=kernel_sizes[i],
                                 strides=conv_strides[i], padding=conv_padding,
                                 kernel_initializer=kernel_init,
                                 input_shape=input_shape,
                                 name='convlstm_layer1',
                                 return_sequences=True,
                                 dilation_rate=dilation_rate[i],
                                 activation='tanh',
                                 recurrent_activation=rec_act,
                                 use_bias=True, recurrent_initializer=rec_init,
                                 bias_initializer='zeros',
                                 unit_forget_bias=True,
                                 # TODO: Errors when using dropout, Keras bug?
                                 dropout=0.0, recurrent_dropout=0.0))
                M.add(SpatialDropout3D(0.5))
            else:
                M.add(ConvLSTM2D(filters=conv_nfilters[i],
                                 kernel_size=kernel_sizes[i],
                                 strides=conv_strides[i], padding=conv_padding,
                                 kernel_initializer=kernel_init,
                                 name='convlstm_layer' + str(i + 1),
                                 return_sequences=True,
                                 dilation_rate=dilation_rate[i],
                                 activation='tanh',
                                 recurrent_activation=rec_act,
                                 use_bias=True, recurrent_initializer=rec_init,
                                 bias_initializer='zeros',
                                 unit_forget_bias=True,
                                 # TODO: Errors when using dropout, Keras bug?
                                 dropout=0.0, recurrent_dropout=0.0))
                M.add(SpatialDropout3D(0.25))

            if pool_layers != 0:
                M.add(pooling_func(pool_size=pool_sizes[i],
                                   strides=pool_strides[i], padding='valid'))
                pool_layers -= 1

        elif layer_type[i] == 'bclstm':
            msg = 'are tuples of 2 integers'
            if not len(kernel_sizes[i]) == 2:
                raise ValueError('Kernel sizes for ConvLSTM' + msg)
            if not len(conv_strides[i]) == 2:
                raise ValueError('Strides for ConvLSTM')
            if not len(dilation_rate[i]) == 2:
                raise ValueError('Dilation rates for ConvLSTM')

            if i == 0:
                M.add(Bidirectional(ConvLSTM2D(filters=conv_nfilters[i],
                                               kernel_size=kernel_sizes[i],
                                               strides=conv_strides[i],
                                               padding=conv_padding,
                                               kernel_initializer=kernel_init,
                                               name='convlstm_layer1',
                                               return_sequences=True,
                                               dilation_rate=dilation_rate[i],
                                               activation='tanh',
                                               recurrent_activation=rec_act,
                                               use_bias=True,
                                               recurrent_initializer=rec_init,
                                               bias_initializer='zeros',
                                               unit_forget_bias=True),
                                    input_shape=input_shape))
                M.add(SpatialDropout3D(0.5))
            else:
                M.add(Bidirectional(ConvLSTM2D(filters=conv_nfilters[i],
                                               kernel_size=kernel_sizes[i],
                                               strides=conv_strides[i],
                                               padding=conv_padding,
                                               kernel_initializer=kernel_init,
                                               name='convlstm_layer' + str(i+1),
                                               return_sequences=True,
                                               dilation_rate=dilation_rate[i],
                                               activation='tanh',
                                               recurrent_activation=rec_act,
                                               use_bias=True,
                                               recurrent_initializer=rec_init,
                                               bias_initializer='zeros',
                                               unit_forget_bias=True)))
                M.add(SpatialDropout3D(0.25))

            if pool_layers != 0:
                M.add(pooling_func(pool_size=pool_sizes[i],
                                   strides=pool_strides[i], padding='valid'))
                pool_layers -= 1

        elif layer_type[i] in ('lrcn', 'blrcn', 'grcn', 'bgrcn'):
            if not len(kernel_sizes[i]) == 2:
                raise ValueError(
                    'Kernel sizes for LRCN are tuples of 2 values')
            if not len(conv_strides[i]) == 2:
                raise ValueError('Strides for LRCN are tuples of 2 values')
            if not len(dilation_rate[i]) == 2:
                raise ValueError('Dilation for LRCN is a tuple of 2 values')

            if i == 0:
                # TimeDistributed wrapper applies a layer to every temporal
                # slice of an input. The input should be at least 3D and the
                # dimension of index one will be considered to be the temporal
                # dimension.
                M.add(TimeDistributed(Conv2D(filters=conv_nfilters[i],
                                             kernel_size=kernel_sizes[i],
                                             strides=conv_strides[i],
                                             padding=conv_padding,
                                             name='lrcn_layer1',
                                             activation=activation),
                                      input_shape=input_shape))
                # This version performs the same function as Dropout, however it
                # drops entire 2D feature maps instead of individual elements.
                M.add(TimeDistributed(SpatialDropout2D(0.5)))
            else:
                M.add(TimeDistributed(Conv2D(filters=conv_nfilters[i],
                                             kernel_size=kernel_sizes[i],
                                             strides=conv_strides[i],
                                             padding=conv_padding,
                                             name='lrcn_layer' + str(i + 1),
                                             activation=activation)))
                M.add(TimeDistributed(SpatialDropout2D(0.25)))

            if pool_layers != 0:
                M.add(TimeDistributed(pooling_func(pool_size=pool_sizes[i],
                                                   strides=pool_strides[i],
                                                   padding='valid')))
                pool_layers -= 1

    # (B)LRCN or (B)GRCN on Conv2d extracted features
    if layer_type[-1] == 'lrcn':
        M.add(TimeDistributed(Flatten(name='flatten')))
        M.add(Dropout(0.5, name='dropout_flatten'))
        M.add(CuDNNLSTM(rec_hidden_states, kernel_initializer=kernel_init,
                        return_sequences=False))
        M.add(Dropout(0.5, name='dropout_lstm'))
    elif layer_type[-1] == 'blrcn':
        M.add(TimeDistributed(Flatten(name='flatten')))
        M.add(Dropout(0.5, name='dropout_flatten'))
        M.add(Bidirectional(LSTM(rec_hidden_states,  # TODO: bug CuDNNLSTM?
                                 kernel_initializer=kernel_init,
                                 return_sequences=False)))
        M.add(Dropout(0.5, name='dropout_lstm'))
    elif layer_type[-1] == 'grcn':
        M.add(TimeDistributed(Flatten(name='flatten')))
        M.add(Dropout(0.5, name='dropout_flatten'))
        M.add(CuDNNGRU(rec_hidden_states, kernel_initializer=kernel_init,
                       return_sequences=False))
        M.add(Dropout(0.5, name='dropout_lstm'))
    elif layer_type[-1] == 'bgrcn':
        M.add(TimeDistributed(Flatten(name='flatten')))
        M.add(Dropout(0.5, name='dropout_flatten'))
        M.add(Bidirectional(CuDNNGRU(rec_hidden_states,
                                     kernel_initializer=kernel_init,
                                     return_sequences=False)))
        M.add(Dropout(0.5, name='dropout_lstm'))
    # Otherwise we just flatten and go to dense layers
    else:
        M.add(Flatten(name='flatten'))

    # Fully-connected or dense layer
    M.add(Dense(units=dense_units, name='dense_layer'))
    M.add(Activation(activation, name='activ_dense'))
    M.add(Dropout(rate=0.5, name='dropout_dense'))

    # Sigmoid unit
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

    # Training the network
    if ngpus > 1:
        # Multi-GPUs
        Mpar = multi_gpu_model(M, gpus=ngpus)
        # Training the network
        Mpar.compile(loss='binary_crossentropy', metrics=['accuracy'],
                     optimizer=Adam(lr=learnrate, decay=1e-2))
        hist = Mpar.fit(X_train, y_train, batch_size=batchsize, epochs=epochs,
                        initial_epoch=0, verbose=verb, shuffle=True,
                        validation_split=validation_split, callbacks=callbacks)
        score = Mpar.evaluate(X_test, y_test, verbose=verb)

    else:
        # Single GPU
        M.compile(loss='binary_crossentropy', metrics=['accuracy'],
                  optimizer=Adam(lr=learnrate, decay=1e-2))
        hist = M.fit(X_train, y_train, batch_size=batchsize, epochs=epochs,
                     initial_epoch=0, verbose=verb, shuffle=True,
                     validation_split=validation_split, callbacks=callbacks)
        score = M.evaluate(X_test, y_test, verbose=verb)

    print('\nTest score/loss:', score[0], '\nTest accuracy:', score[1])

    timing(starttime)
    fintime = time_fin(starttime)

    if full_output:
        return M, hist.history, score, fintime
    else:
        return M
