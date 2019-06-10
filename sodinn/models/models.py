"""
Discriminative models
"""
from __future__ import absolute_import
from __future__ import print_function

__all__ = ['Model']

import tables
import numpy as np
import tensorflow as tf
from keras import models
from keras.backend import set_session, get_session
from ..data_labeling.labeling import DataLabeler
from .model_2dcnn import train_2dconvnet
from .model_3dnet import train_3dnet
from .model_4dnet import train_4dnet


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
                 conv2d_pseudo3d=False, identifier=1, dir_path=None):
        """
        Model generation for a given DataLabeler

        Parameters
        ----------
        labeled_data : DataLabeler
            DataLabeler from SODINN. Makes sure to run() it before
        layer_type : tuple of str {'conv3d', 'clstm', 'bclstm', 'lrcn',
        'blrcn', 'grcn', 'bgrcn'}, optional
            [labeled_data.sample_dim > 3 & conv2d_pseudo3d=False] The type of
            layers in the neural net, the len of the tuple is number of layers
        conv_nfilters : tuple of int, optional
            Filters used in ``keras.layers`` module. The dimensionality of the
            output space (i.e. the number of output filters in the convolution).
        kernel_sizes : tuple of tuples of int, optional
            specifying the depth, height and width of the convolution window
        conv_strides : tuple of tuples of int, optional
            specifying the strides of the convolution along each spatial
            dimension. Specifying any stride value != 1 is incompatible with
            specifying any `dilation_rate` value != 1.
        conv_padding : {'valid' or 'same'}, optional
        dilation_rate : tuple of tuples of int, optinal
            specifying the dilation rate to use for dilated convolution.
            Currently, specifying any `dilation_rate` value != 1 is incompatible
            with specifying any stride value != 1.
        pool_layers : int, optional
            The number of pool layers int the net. It can't be greater than the
            number of convolution layers
        pool_func : {'ave' or 'max'}, optional
            Type if the pool layers, AveragePooling or MaxPooling
        pool_sizes : tuple of tuples of int, optional
            factors by which to downscale in the pool layers
        pool_strides : tuple of tuples of int, optional
            Strides value for the pool layers
        rec_hidden_states : int, optional
            [layer_type = lrcn', 'blrcn', 'grcn' or 'bgrcn'] dimensionality of
            the output space for CuDNN layers
        dense_units : int, optional
            dimensionality of the output space for Dense layer
        activation : str, optional
            Type of the Activation layer. See ''keras.layers.Activation''
        conv2d_pseudo3d : bool, optional
            [sample_dim==3] if True, force to train a 2D net with a 3D input by
            changing the shape of the input cube
        identifier : int, optional
            Id key of the Model object
        dir_path : path or None, optional
            Path were the model will be saved
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
        self.labda_identifier = labeled_data.labda_identifier

        # save_filename_model: eg. dir_path/model_mlar_v1_clstm_v1
        self.model_identifier = 'v' + str(identifier)
        type_layer1st = layer_type[0]
        if type_layer1st == 'conv2d' and self.sample_dim == 3:
            type_layer1st = 'ps3d'
        self.labda_name = self.sample_type + '_' + self.labda_identifier
        self.model_name = type_layer1st + '_' + self.model_identifier
        if dir_path is not None:
            self.save_filename_model = dir_path + 'model_' + self.labda_name \
                                       + '_' + self.model_name
        else:
            self.save_filename_model = None

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
              gpu_id='0', plot='tb', tblog_path='./logs/', tblog_name=None):
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
        else:
            retrain = None

        x = np.concatenate((self.x_plus, self.x_minus), axis=0)
        y = np.concatenate((self.y_plus, self.y_minus), axis=0)

        if tblog_name is None:
            tblog_name = ''
        tblog_fullpath = tblog_path + self.labda_name + '_' + self.model_name \
                         + '_' + tblog_name

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
                     full_output=True, plot=plot, tb_path=tblog_fullpath)

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
                     full_output=True, plot=plot, tb_path=tblog_fullpath)

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
                     summary=summary, plot=plot, tb_path=tblog_fullpath,
                     full_output=True)

        elif self.sample_dim == 4:
            nn = train_4dnet
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
                     summary=summary, plot=plot, tb_path=tblog_fullpath,
                     full_output=True)

        self.model, self.history, self.score, self.runtime = res
        self.epochs_trained = len(self.history['val_loss'])

        if self.save_filename_model is not None:
            self.save(self.save_filename_model)


