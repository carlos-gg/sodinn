"""
Prediction class.
"""

__all__ = ['Predictor']

from copy import deepcopy

import numpy as np
import tables
from hciplot import plot_frames

from .prediction_mlar import predict_mlar
from .prediction_pw import predict_pairwise
from ..data_labeling.labeling import DataLabeler
from ..models.models import Model


class Predictor:
    """
    """
    def __init__(self, labeled_data, model, radius_int=None, radius_out=None,
                 identifier=1, dir_path=None):
        """
        """
        if not hasattr(labeled_data, 'x_minus'):
            raise ValueError('labeled_data must be a sodinn.DataLabeler object')
        if not hasattr(model, 'model'):
            raise ValueError('model must be a sodinn.Model object')

        self.model = model.model
        self.save_filename_labdata = deepcopy(model.save_filename_labdata)
        self.save_filename_model = deepcopy(model.save_filename_model)
        self.layer_type = deepcopy(model.layer_type)
        self.nconvlayers = deepcopy(model.nconvlayers)
        self.conv_nfilters = deepcopy(model.conv_nfilters)
        self.kernel_sizes = deepcopy(model.kernel_sizes)
        self.conv_strides = deepcopy(model.conv_strides)
        self.conv_padding = deepcopy(model.conv_padding)
        self.dilation_rate = deepcopy(model.dilation_rate)
        self.pool_func = deepcopy(model.pool_func)
        self.pool_layers = deepcopy(model.pool_layers)
        self.pool_sizes = deepcopy(model.pool_sizes)
        self.pool_strides = deepcopy(model.pool_strides)
        self.dense_units = deepcopy(model.dense_units)
        self.activation = deepcopy(model.activation)
        self.rec_hidden_states = deepcopy(model.rec_hidden_states)
        self.learning_rate = deepcopy(model.learning_rate)
        self.batch_size = deepcopy(model.batch_size)
        self.test_split = deepcopy(model.test_split)
        self.validation_split = deepcopy(model.validation_split)
        self.epochs = deepcopy(model.epochs)
        self.epochs_trained = deepcopy(model.epochs_trained)
        self.score = deepcopy(model.score)
        self.patience = deepcopy(model.patience)
        self.min_delta = deepcopy(model.min_delta)
        self.gpu_id = deepcopy(model.gpu_id)
        self.runtime_train = deepcopy(model.runtime)

        self.cube = deepcopy(labeled_data.cube)
        self.n_cubes = deepcopy(labeled_data.n_cubes)
        self.pa = deepcopy(labeled_data.pa)
        self.fwhm = deepcopy(labeled_data.fwhm)
        self.plsc = deepcopy(labeled_data.plsc)
        self.patch_size = deepcopy(labeled_data.patch_size)
        self.patch_size_px = deepcopy(labeled_data.patch_size_px)
        self.delta_rot = deepcopy(labeled_data.delta_rot)
        if radius_int is None:
            self.radius_int = deepcopy(labeled_data.radius_int)
        else:
            self.radius_int = radius_int
        self.radius_out = radius_out
        self.high_pass = deepcopy(labeled_data.high_pass)
        self.kernel_size = deepcopy(labeled_data.kernel_size)
        self.normalization = deepcopy(labeled_data.normalization)
        self.min_n_slices = deepcopy(labeled_data.min_n_slices)
        self.cevr_thresh = deepcopy(labeled_data.cevr_thresh)
        self.n_ks = deepcopy(labeled_data.n_ks)
        self.kss_window = deepcopy(labeled_data.kss_window)
        self.tss_window = deepcopy(labeled_data.tss_window)
        self.lr_mode = deepcopy(labeled_data.lr_mode)
        self.imlib = deepcopy(labeled_data.imlib)
        self.interpolation = deepcopy(labeled_data.interpolation)
        self.sample_type = deepcopy(labeled_data.sample_type)
        self.sample_dim = deepcopy(labeled_data.sample_dim)
        self.min_adi_snr = deepcopy(labeled_data.min_adi_snr)
        self.max_adi_snr = deepcopy(labeled_data.max_adi_snr)
        self.sampling_sep = deepcopy(labeled_data.sampling_sep)
        self.augmented = deepcopy(labeled_data.augmented)
        self.n_aug_inj = deepcopy(labeled_data.n_aug_inj)
        self.n_aug_aver = deepcopy(labeled_data.n_aug_aver)
        self.n_aug_rotshi = deepcopy(labeled_data.n_aug_rotshi)
        self.n_aug_mupcu = deepcopy(labeled_data.n_aug_mupcu)
        self.nsamp_sep = deepcopy(labeled_data.nsamp_sep)
        self.min_n_slices = deepcopy(labeled_data.min_n_slices)
        self.flo = deepcopy(labeled_data.flo)
        self.fhi = deepcopy(labeled_data.fhi)
        self.distances = deepcopy(labeled_data.distances)
        self.n_init_samples = deepcopy(labeled_data.n_init_samples)
        self.n_total_samples = deepcopy(labeled_data.n_total_samples)
        self.runtime_labda = deepcopy(labeled_data.runtime)
        self.labda_identifier = deepcopy(labeled_data.labda_identifier)
        del labeled_data

        # save_filename_prediction: eg. dir_path/pred_v1_mlar_v1_clstm_v1
        self.pred_identifier = 'v' + str(identifier)
        self.model_identifier = deepcopy(model.model_identifier)
        type_layer1st = self.layer_type[0]
        if type_layer1st == 'conv2d' and self.sample_dim == 3:
            type_layer1st = 'ps3d'
        self.model_name = type_layer1st + '_' + self.model_identifier
        self.labda_name = self.sample_type + '_' + self.labda_identifier
        if dir_path is not None:
            self.save_filename_prediction = dir_path + 'pred_' \
                                            + self.pred_identifier + '_' \
                                            + self.labda_name + '_' \
                                            + self.model_name
        else:
            self.save_filename_prediction = None

        self.pmap = None
        self.cube_index = None
        self.patches = None
        self.coords = None
        self.probas = None
        self.cube_pred = None
        self.pa_pred = None

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

        if self.pmap is None:
            raise RuntimeError('The predictor has not been executed (.run())')

        # Creating HDF5 file
        if not filename.endswith('.hdf5'):
            filename += '.hdf5'

        with tables.open_file(filename, mode='w') as fh5:
            # Writing to HDF5 file
            for key in self.__dict__.keys():
                if key not in ['model']:
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
                            _ = fh5.create_array('/', key, obj=attr,
                                                 atom=f32atom)
                        elif isinstance(attr, np.ndarray):
                            if attr.dtype == 'float64':
                                attr = attr.astype('float32')
                                _ = fh5.create_array('/', key, obj=attr,
                                                     atom=f32atom)
                            else:
                                _ = fh5.create_array('/', key, obj=attr)
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
        """
        if not filename.endswith('.hdf5'):
            filename += '.hdf5'

        # Reading HDF5 file
        with tables.open_file(filename, mode='r') as fh5:
            filen_labdata = fh5.root.save_filename_labdata[0].decode()
            filen_model = fh5.root.save_filename_model[0].decode()
            labeled_data = DataLabeler.load(filen_labdata)
            model = Model.load(filen_model)
            obj = cls(labeled_data, model)
            obj.pmap = fh5.root.pmap.read()
            obj.save_filename_prediction = filename
        return obj

    def print_info(self):
        """
        """
        print('LabData info :')
        print('\tSaved file : {}'.format(self.save_filename_labdata))
        print('\tN cubes : {}'.format(self.n_cubes))
        print('\tInit samples : {}'.format(self.n_init_samples))
        if self.n_cubes == 0:
            dists = len(self.distances)
        else:
            dists = len(self.distances[0])
        print('\tN distances : {}'.format(dists))
        print('\tAugmented : {}'.format(self.augmented))
        print('\tTotal samples : {}'.format(self.n_total_samples))
        print('\tSample type : {}'.format(self.sample_type))
        if self.sample_type in ('pw2d', 'pw3d'):
            print('\tSample dim : {}'.format(self.sample_dim))
            print('\tHP filter : {}'.format(self.high_pass))
        elif self.sample_type in ('mlar', 'tmlar', 'tmlar4d'):
            print('\tCEVR thresh : {}'.format(self.cevr_thresh))
            print('\tN k slices : {}'.format(self.n_ks))
        print('\tNormalization : {}'.format(self.normalization))
        print('\tPatch size : {} ({} pxs)'.format(self.patch_size,
                                                  self.patch_size_px))
        print('\tMin ADI S/N : {}'.format(self.min_adi_snr))
        print('\tMax ADI S/N : {}'.format(self.max_adi_snr))
        print('\tRuntime (labda generation) : {}'.format(self.runtime_labda))

        print('Model info :')
        print('\tSaved file : {}'.format(self.save_filename_model))
        print('\tLayer type : {}'.format(self.layer_type))
        print('\tConv nfilters : {}'.format(self.conv_nfilters))
        print('\tConv kernels : {}'.format(self.kernel_size))
        print('\tPooling function : {}'.format(self.pool_func))
        print('\tMax Epochs : {}'.format(self.epochs))
        print('\tTrained Epochs : {}'.format(self.epochs_trained))
        print('\tTest loss : {}'.format(self.score[0]))
        print('\tTest accuracy : {}'.format(self.score[1]))
        print('\tRuntime (model training) : {}'.format(self.runtime_train))

        print('Predictor info :')
        if self.cube_index is not None:
            print('\tCube index : {}'.format(self.cube_index))
        else:
            print('\tExternal cube (stored in self.cube_pred)')

    def inspect_patch(self, xy, cmap='bone', dpi=40):
        """
        """
        if self.probas is None:
            raise RuntimeError("You must run the predictor first")

        x_input, y_input = xy
        if not isinstance(xy, tuple):
            raise TypeError("`xy` must be a tuple")

        for i, coord in enumerate(self.coords):
            if coord[0] == y_input and coord[1] == x_input:
                index = i

        if index is None:
            raise ValueError("Input coordinates not found")

        prob = self.probas[index]
        print("Proba : " + str(prob))
        sample = np.squeeze(self.patches[index])
        max_slices = sample.shape[0]
        if self.sample_type == 'tmlar4d':
            for i in range(sample.shape[1]):
                plot_frames(tuple(sample[:, i]), axis=False,
                            colorbar=False, cmap=cmap, dpi=dpi, horsp=0.05)
        else:
            plot_frames(tuple(sample), axis=False, colorbar=False,
                        cmap=cmap, dpi=dpi, horsp=0.05)

    def inspect_probmap(self, vmin_log=1e-10, labelsize=10, circlerad=10,
                        circlecolor='white', circlealpha=0.6, grid=True,
                        gridspacing=10, gridalpha=0.2, showcent=True,
                        print_info=True, **kwargs):
        """
        from matplotlib.pyplot import hist, figure

        vec = pred_svd.pmap.flatten()
        _ = hist(vec[vec > 0], bins=np.sqrt(vec.shape[0]).astype(int))
        figure()
        _ = hist(np.log(vec[vec > 0]), bins=np.sqrt(vec.shape[0]).astype(int))

        """
        if print_info:
            self.print_info()

        plot_frames((tuple(self.pmap), tuple(self.pmap)), log=(False, True),
                    vmin=(0, vmin_log), vmax=(1, 1),
                    label=('Probmap', 'Probmap (logscale)'),
                    labelsize=labelsize, circlerad=circlerad,
                    circlecolor=circlecolor, circlealpha=circlealpha,
                    showcent=showcent, grid=grid, gridalpha=gridalpha,
                    gridspacing=gridspacing, **kwargs)

    def run(self, n_proc=30, chunks_per_proc=2, cube=None, pa=None,
            cube_index=0, verbose=True):
        """
        """
        # predicting on the selected cube (cube_index or provided cube)
        if cube is None:
            if cube_index >= self.n_cubes:
                msg = 'cube index is too large (with only {} cubes)'
                raise ValueError(msg.format(self.n_cubes))
            self.cube_index = cube_index
            cube = self.cube[cube_index]
            pa = self.pa[cube_index]
        else:
            if pa is None:
                raise ValueError('pa was not provided')
            else:
                pa = pa
            cube = cube
            self.cube_pred = cube
            self.pa_pred = pa

        if self.sample_type in ('pw2d', 'pw3d'):
            pmap = predict_pairwise(model=self.model, cube=cube,
                                    angle_list=pa, fwhm=self.fwhm,
                                    patch_size_px=self.patch_size_px,
                                    delta_rot=self.delta_rot,
                                    radius_int=self.radius_int,
                                    high_pass=self.high_pass,
                                    kernel_size=self.kernel_size,
                                    normalization=self.normalization,
                                    imlib=self.imlib,
                                    interpolation=self.interpolation,
                                    nproc=n_proc, verbose=verbose,
                                    chunks_per_proc=chunks_per_proc)
            patches = None
            coords = None
            probas = None

        elif self.sample_type in ('mlar', 'tmlar', 'tmlar4d'):
            if self.radius_out is None:
                # ann_width = fwhm + size_patch + 2 (for the MLAR samples)
                halfsize = int(cube.shape[1] / 2)
                self.radius_out = halfsize - self.patch_size_px

            radint_fwhm = int(self.radius_int / self.fwhm)
            radout_fwhm = int(self.radius_out / self.fwhm)
            res = predict_mlar(mode=self.sample_type, model=self.model,
                               cube=cube, angle_list=pa, fwhm=self.fwhm,
                               in_ann=radint_fwhm, out_ann=radout_fwhm,
                               patch_size=self.patch_size_px,
                               cevr_thresh=self.cevr_thresh, n_ks=self.n_ks,
                               kss_window=self.kss_window,
                               tss_window=self.tss_window,
                               normalize=self.normalization, n_proc=n_proc,
                               verbose=verbose, lr_mode=self.lr_mode)
            pmap = res[0]
            probas = res[1]
            patches = res[2]
            coords = res[3]

        self.pmap = pmap
        self.probas = probas
        self.patches = patches
        self.coords = coords

        if self.save_filename_prediction is not None:
            self.save(self.save_filename_prediction)

