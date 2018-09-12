"""
Discriminative models
"""
from __future__ import print_function
from __future__ import absolute_import

__all__ = ['Model']

import tables
import numpy as np
import tensorflow as tf
import livelossplot
from tensorflow.keras import models
from tensorflow import get_default_graph, Session
# from keras import initializers
# from keras import activations
# from keras import regularizers
# from keras import constraints
# from keras.legacy import interfaces
# from keras import backend as K
# from keras.layers import ConvRecurrent2D
# from keras.engine import InputSpec
# from keras.utils import conv_utils
# from keras.layers.recurrent import _generate_dropout_mask
# from keras.layers import ConvRNN2D
# from keras.engine import Layer
# from tensorflow.keras.layers.wrappers import TimeDistributed, Bidirectional
# from tensorflow.keras.backend.tensorflow_backend import set_session
from tensorflow.keras.backend import set_session, get_session, clear_session
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, Bidirectional
from tensorflow.keras.layers import (Dense, Dropout, Activation, Flatten,
                                          Conv2D, MaxPooling2D, Conv3D,
                                          MaxPooling3D, ZeroPadding3D,
                                          CuDNNLSTM, LSTM, CuDNNGRU,
                                          SpatialDropout3D, SpatialDropout2D,
                                          AveragePooling2D, AveragePooling3D)
from tensorflow.keras.optimizers import Adam, RMSprop, Adadelta, SGD
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.layers import ConvLSTM2D
# from tensorflow.keras.layers.convolutional_recurrent import ConvLSTM2D
# from keras.layers.normalization import BatchNormalization
# from keras.regularizers import l2, l1, l1_l2
# from keras.layers.advanced_activations import PReLU
from vip_hci.conf import time_ini, timing, time_fin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from .data_labeling.labeling import DataLabeler


class Model:
    """

    """
    def __init__(self, labeled_data, layer_type=('conv3d', 'conv3d'),
                 conv_nfilters=(40, 80), kernel_sizes=((3, 3, 3), (2, 2, 2)),
                 conv_strides=((1, 1, 1), (1, 1, 1)), conv_padding='same',
                 dilation_rate=((1, 1, 1), (1, 1, 1)), pool_layers=2,
                 pool_func='ave', pool_sizes=((2, 2, 2), (2, 2, 2)),
                 pool_strides=((2, 2, 2), (2, 2, 2)),
                 rec_hidden_states=128, dense_units=128, activation='relu',
                 conv2d_pseudo3d=False, save=None):
        """
        """
        if not hasattr(labeled_data, 'x_minus'):
            raise ValueError('labeled_data must be a sodinn.DataLabeler object')

        self.sample_type = labeled_data.sample_type
        self.sample_dim = labeled_data.sample_dim
        self.fwhm = labeled_data.fwhm
        self.plsc = labeled_data.plsc
        self.patch_size = labeled_data.patch_size
        self.slice3d = labeled_data.slice3d
        self.high_pass = labeled_data.high_pass
        self.kernel_size = labeled_data.kernel_size
        self.normalization = labeled_data.normalization
        self.imlib = labeled_data.imlib
        self.interpolation = labeled_data.interpolation
        self.min_adi_snr = labeled_data.min_adi_snr
        self.max_adi_snr = labeled_data.max_adi_snr
        self.sampling_sep = labeled_data.sampling_sep
        self.nsamp_sep = labeled_data.nsamp_sep
        self.min_n_slices = labeled_data.min_n_slices
        self.cevr_thresh = labeled_data.cevr_thresh
        self.n_ks = labeled_data.n_ks
        self.n_proc = labeled_data.n_proc
        self.radius_int = labeled_data.radius_int
        self.delta_rot = labeled_data.delta_rot
        self.flo = labeled_data.flo
        self.fhi = labeled_data.fhi
        self.distances = labeled_data.distances
        self.x_minus = labeled_data.x_minus
        self.x_plus = labeled_data.x_plus
        self.y_minus = labeled_data.y_minus
        self.y_plus = labeled_data.y_plus
        self.augmented = labeled_data.augmented
        self.n_aug_inj = labeled_data.n_aug_inj
        self.n_aug_aver = labeled_data.n_aug_aver
        self.n_aug_rotshi = labeled_data.n_aug_rotshi
        self.n_aug_mupcu = labeled_data.n_aug_mupcu
        self.save_filename_labdata = labeled_data.save_filename_labdata

        self.save_filename_model = save
        self.conv2d_pseudo3d = conv2d_pseudo3d
        if self.sample_dim == 2 and self.conv2d_pseudo3d:
            raise ValueError('Conv2d_pseudo3d only works with 3d samples')
        self.layer_type = layer_type
        self.nconvlayers = len(self.layer_type)
        self.conv_nfilters = conv_nfilters
        self.kernel_sizes = kernel_sizes
        self.conv_strides = conv_strides
        self.conv_padding = conv_padding
        self.dilation_rate = dilation_rate
        self.pool_func = pool_func
        self.pool_layers = pool_layers
        self.pool_sizes = pool_sizes
        self.pool_strides = pool_strides
        self.dense_units = dense_units
        self.activation = activation
        self.rec_hidden_states = rec_hidden_states
        self.history = None
        self.score = None
        self.runtime = None
        self.learning_rate = None
        self.batch_size = None
        self.test_split = None
        self.validation_split = None
        self.random_state = None
        self.epochs = None
        self.patience = None
        self.min_delta = None
        self.gpu_id = None
        self.model = None
        self.epochs_trained = None

    def save(self, filename):
        """
        """
        def fix_tup_of_tup_len(tup):
            max_len_tup = max([len(i) for i in tup])
            min_len_tup = min([len(i) for i in tup])
            if min_len_tup < max_len_tup:
                tup2li = list(list(i) for i in tup)
                for inntu in tup2li:
                    while len(inntu) < max_len_tup:
                        inntu.append(0)
                li2tup = tuple(tuple(i) for i in tup2li)
                return li2tup
            else:
                return tup

        if self.model is None:
            raise RuntimeError('The model has not been trained (.train())')

        self.model.save(filename + '.h5')

        # Creating HDF5 file
        if not filename.endswith('.hdf5'):
            filename += '.hdf5'

        with tables.open_file(filename, mode='w') as fh5:

            # Writing to HDF5 file
            for key in self.__dict__.keys():
                if key not in ['model', 'x_minus', 'x_plus', 'y_minus',
                               'y_plus']:
                    attr = self.__dict__[key]
                    f32atom = tables.Float32Atom()
                    if isinstance(attr, str):
                        attr = np.char.array(attr)
                        _ = fh5.create_array('/', key, obj=attr)
                    elif attr is None:
                        attr = np.char.array('None')
                        _ = fh5.create_array('/', key, obj=attr)
                    elif isinstance(attr, (np.ndarray, list)):
                        if isinstance(attr, list):
                            attr = np.array(attr, dtype='float32')
                        else:
                            if attr.dtype == 'float64':
                                attr = attr.astype('float32')
                        _ = fh5.create_array('/', key, obj=attr, atom=f32atom)
                    elif isinstance(attr, tuple):
                        if isinstance(attr[0], np.ndarray):
                            attr = np.array(attr, dtype='float32')
                            _ = fh5.create_array('/', key, obj=attr,
                                                 atom=f32atom)
                        elif isinstance(attr[0], int):
                            attr = np.array(attr, dtype='int')
                            _ = fh5.create_array('/', key, obj=attr)
                        elif isinstance(attr[0], tuple):
                            attr = fix_tup_of_tup_len(attr)
                            _ = fh5.create_array('/', key, obj=attr)
                        elif isinstance(attr[0], str):
                            attr = np.char.array(attr)
                            _ = fh5.create_array('/', key, obj=attr)
                    elif isinstance(attr, dict):
                        # for the history dictionary -> np.ndarray
                        # array rows : val_loss, val_acc, loss, acc
                        attr = np.array(tuple(item for _, item in attr.items()))
                        _ = fh5.create_array('/', key, obj=attr)
                    else:
                        _ = fh5.create_array('/', key, obj=attr)

            fh5.flush()

    @classmethod
    def load(cls, filename):
        """
        # TODO: find out why model trains in a single GPU after re-loading
        """
        def array_to_tup_of_tup(array):
            arr2li = list(list(i) for i in array)
            for elem in arr2li:
                if elem[-1] == 0:
                    elem.pop()
            return tuple(tuple(i) for i in arr2li)

        # config = tf.ConfigProto()
        # # Don't pre-allocate memory; allocate as-needed
        # config.gpu_options.allow_growth = True
        # # Only allow a fraction the GPU memory to be allocated
        # config.gpu_options.per_process_gpu_memory_fraction = gpu_memfract
        # config.gpu_options.visible_device_list = gpu_id
        set_session(get_session())

        # Opening HDF5 file
        if not filename.endswith('.hdf5'):
            filename += '.hdf5'
        fh5 = tables.open_file(filename, mode='r')
        filen_labdata = fh5.root.save_filename_labdata[0].decode() + '.hdf5'
        labeled_data = DataLabeler.load(filen_labdata)
        # model = models.load_model(fh5.root.save_filename_model[0].decode() +
        #                           '.h5')
        with tf.device('/cpu:0'):
            model = models.load_model(fh5.root.save_filename_model[0].decode() +
                                      '.h5')
        obj = cls(labeled_data)
        obj.model = model
        obj.layer_type = tuple(np.char.decode(fh5.root.layer_type))
        obj.nconvlayers = fh5.root.nconvlayers.read()
        obj.conv_nfilters = tuple(fh5.root.conv_nfilters)
        obj.kernel_sizes = array_to_tup_of_tup(fh5.root.kernel_sizes)
        obj.conv_strides = array_to_tup_of_tup(fh5.root.conv_strides)
        obj.dilation_rate = array_to_tup_of_tup(fh5.root.dilation_rate)
        obj.pool_layers = fh5.root.pool_layers.read()
        obj.pool_sizes = array_to_tup_of_tup(fh5.root.pool_sizes)
        obj.pool_strides = array_to_tup_of_tup(fh5.root.pool_strides)
        obj.pool_func = str(fh5.root.pool_func[0].decode())
        obj.dense_units = fh5.root.dense_units.read()
        obj.activation = str(fh5.root.activation[0].decode())
        obj.conv2d_pseudo3d = fh5.root.conv2d_pseudo3d.read()
        obj.save_filename_model = filename
        obj.save_filename_labdata = filen_labdata
        obj.learning_rate = fh5.root.learning_rate.read()
        obj.batch_size = fh5.root.batch_size.read()
        obj.test_split = fh5.root.test_split.read()
        obj.validation_split = fh5.root.validation_split.read()
        obj.random_state = fh5.root.random_state.read()
        obj.epochs = fh5.root.epochs.read()
        obj.epochs_trained = fh5.root.epochs_trained.read()
        obj.patience = fh5.root.patience.read()
        obj.min_delta = fh5.root.min_delta.read()
        obj.gpu_id = np.char.decode(fh5.root.gpu_id).tolist()[0]
        obj.runtime = str(fh5.root.runtime[0].decode())
        obj.score = fh5.root.score.read()

        fh5.close()
        return obj

    def train(self, test_split=0.1, validation_split=0.1, random_state=0,
              learning_rate=0.003, batch_size=64, epochs=20, patience=2,
              min_delta=0.001, retrain=False, verbose=1, summary=True,
              gpu_id='0', plot='tb', tblog_path='./logs'):
        """

        Parameters
        ----------
        min_delta:
            Minimum change in the monitored quantity to qualify as an
            improvement, i.e. an absolute change of less than min_delta, will
            count as no improvement.
        patience:
            Number of epochs with no improvement after which training will be
            stopped.
        """
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.test_split = test_split
        self.validation_split = validation_split
        self.random_state = random_state
        self.epochs = epochs
        self.patience = patience
        self.min_delta = min_delta
        self.gpu_id = gpu_id
        if retrain:
            retrain = self.model

        x = np.concatenate((self.x_plus, self.x_minus), axis=0)
        y = np.concatenate((self.y_plus, self.y_minus), axis=0)

        if self.sample_dim == 2:
            nn = train_2dconvnet
            res = nn(x, y, test_size=self.test_split,
                     validation_split=self.validation_split, pseudo3d=False,
                     random_state=self.random_state,
                     nconvlayers=self.nconvlayers,
                     conv_nfilters=self.conv_nfilters,
                     kernel_sizes=self.kernel_sizes,
                     conv_strides=self.conv_strides,
                     pool_layers=self.pool_layers, pool_sizes=self.pool_sizes,
                     pool_strides=self.pool_strides,
                     dense_units=self.dense_units, activation=self.activation,
                     learnrate=self.learning_rate, batchsize=self.batch_size,
                     epochs=self.epochs, patience=self.patience,
                     gpu_id=self.gpu_id, min_delta=self.min_delta,
                     retrain=retrain, verb=verbose, summary=summary,
                     full_output=True, plot=plot, tb_path=tblog_path)

        elif self.sample_dim == 3 and self.conv2d_pseudo3d:
            nn = train_2dconvnet
            res = nn(x, y, test_size=self.test_split,
                     validation_split=self.validation_split, pseudo3d=True,
                     random_state=self.random_state,
                     nconvlayers=self.nconvlayers,
                     conv_nfilters=self.conv_nfilters,
                     kernel_sizes=self.kernel_sizes,
                     conv_strides=self.conv_strides,
                     pool_layers=self.pool_layers,
                     pool_sizes=self.pool_sizes, pool_strides=self.pool_strides,
                     dense_units=self.dense_units, activation=self.activation,
                     learnrate=self.learning_rate, batchsize=self.batch_size,
                     epochs=self.epochs, patience=self.patience,
                     gpu_id=self.gpu_id, min_delta=self.min_delta,
                     retrain=retrain, verb=verbose, summary=summary,
                     full_output=True, plot=plot, tb_path=tblog_path)

        elif self.sample_dim == 3 and not self.conv2d_pseudo3d:
            nn = train_3dnet
            res = nn(x, y, test_size=self.test_split,
                     validation_split=self.validation_split,
                     random_state=self.random_state, layer_type=self.layer_type,
                     nconvlayers=self.nconvlayers,
                     conv_nfilters=self.conv_nfilters,
                     kernel_sizes=self.kernel_sizes,
                     conv_strides=self.conv_strides,
                     conv_padding=self.conv_padding,
                     dilation_rate=self.dilation_rate,
                     pool_layers=self.pool_layers, pool_func=self.pool_func,
                     pool_sizes=self.pool_sizes, pool_strides=self.pool_strides,
                     dense_units=self.dense_units,
                     rec_hidden_states=self.rec_hidden_states,
                     activation=self.activation, learnrate=self.learning_rate,
                     batchsize=self.batch_size, epochs=self.epochs,
                     patience=self.patience, gpu_id=self.gpu_id,
                     min_delta=self.min_delta, retrain=retrain, verb=verbose,
                     summary=summary, plot=plot, tb_path=tblog_path,
                     full_output=True)

        self.model, self.history, self.score, self.runtime = res
        self.epochs_trained = len(self.history['val_loss'])

        if self.save_filename_model is not None:
            self.save(self.save_filename_model)


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
                raise ValueError('Srides for Conv3d are tuples of 3 values')
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
                # If adjacent pixels within feature maps are strongly correlated
                # (as is normally the case in early convolution layers) then
                # regular dropout will not regularize the activations and will
                # otherwise just result in an effective learning rate decrease.
                # In this case, SpatialDropout2D will help promote independence
                # between feature maps and should be used instead.
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
        hist = Mpar.fit(X_train, y_train, batch_size=batchsize * ngpus,
                        epochs=epochs, initial_epoch=0, verbose=verb,
                        validation_split=validation_split, callbacks=callbacks,
                        shuffle=True)
        score = Mpar.evaluate(X_test, y_test, verbose=verb)

    else:
        # Single GPU
        M.compile(loss='binary_crossentropy', metrics=['accuracy'],
                  optimizer=Adam(lr=learnrate, decay=1e-2))
        hist = M.fit(X_train, y_train, batch_size=batchsize * ngpus,
                     epochs=epochs, initial_epoch=0, verbose=verb,
                     validation_split=validation_split, callbacks=callbacks,
                     shuffle=True)
        score = M.evaluate(X_test, y_test, verbose=verb)

    print('\nTest score/loss:', score[0], '\nTest accuracy:', score[1])

    timing(starttime)
    fintime = time_fin(starttime)

    if full_output:
        return M, hist.history, score, fintime
    else:
        return M


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
