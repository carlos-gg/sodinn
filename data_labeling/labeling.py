"""
Generation of labeled data for supervised learning. To be used to train the
discriminative models. 
"""
from __future__ import print_function, division, absolute_import

__all__ = ['DataLabeler']

import os
import tables
import copy
import numpy as np
from vip_hci.conf import time_ini, timing, time_fin
from vip_hci.var import frame_center, prepare_matrix
from vip_hci.conf.utils_conf import (pool_map, fixed, make_chunks)
from vip_hci.var import cube_filter_highpass, pp_subplots, get_annulus_segments
from vip_hci.metrics import cube_inject_companions
from vip_hci.preproc import (check_pa_vector, cube_derotate, cube_crop_frames,
                             frame_rotate, frame_shift, frame_px_resampling,
                             frame_crop, cube_collapse)
from vip_hci.preproc.derotation import _compute_pa_thresh, _find_indices_adi
from vip_hci.metrics import frame_quick_report
from vip_hci.medsub import median_sub
from vip_hci.pca import pca, svd_wrapper
from .mlar_samples import (make_mlar_samples_ann_noise,
                           make_mlar_samples_ann_signal)
from ..utils import (normalize_01_pw, cube_shift, create_synt_cube,
                     close_hdf5_files)
from .flux_estimation import FluxEstimator


class DataLabeler:
    """ Data labeling for SODINN.
    """
    def __init__(self, sample_type, cube, pa, psf, radius_int=None,
                 fwhm=4, plsc=0.01, delta_rot=0.5, patch_size=4, slice3d=True,
                 high_pass='laplacian', kernel_size=5, normalization='slice',
                 min_snr=1, max_snr=3, cevr_thresh=0.99, n_ks=20,
                 kss_window=None, tss_window=None, lr_mode='eigen',
                 imlib='opencv', interpolation='bilinear', n_proc=1,
                 random_seed=42, identifier=1, dir_path=None, reload=False):
        """

        cube : 3d ndarray or tuple of 3d ndarrays

        Labeled data generation for a given dataset (ADI).

        Parameters
        ----------
        n_samples : int
            For ``mlar`` it is the total number of samples (signal+noise) per
            annulus. For ``pairwise-XXX`` it is the number of iterations done per
            annulus, so the total number of samples for the annulus is ~
            2 * 3 * n_annuli * n_samples * n_temporal_pairwise_slices. The later
            term has a variable value.
        sample_type : {'mlar', 'tmlar', 'tmlar4d', 'pw2d', 'pw3d}
            Type of labeled data depending on the model to be use (SODINN-svd or
            SODINN-pairwise).
        cube
        pa
        psf
        sample_dim : int, ``2`` or ``3``
            Whether to return 2d or 3d samples, in the case of ``sample_type`` ==
            ``pairwise-XXX``.
        radius_int : int or None
            The initial separation in pixels at which the samples will be created.
            The default initial distance is ``1.5*fwhm``.
        sampling_sep : int or None
            Radial distances in pixels at which the samples are created.
        fwhm
        plsc
        delta_rot
        patch_size : int
            Patch size in terms of the FWHM.
        rint_flux_scaling
        rout_flux_scaling
        rint_flux_spread
        rout_flux_spread
        slice3d : bool
            Slicing the 3d samples wrt the shortest sequence.
        high_pass
        normalization
        imlib
        interpolation
        nproc

        Returns
        -------
        x : 3d np.ndarray
        y : 1d np.ndarray


        """
        if isinstance(cube, np.ndarray):
            if cube.ndim == 3:
                self.cube = [cube]
            elif cube.ndim == 4:
                self.cube = cube
        elif isinstance(cube, tuple):
            self.cube = list(cube)
        else:
            msg = 'cube must be a 3d ndarray, a tuple of 3d ndarrays or a 4d '
            msg += 'ndarray'
            raise ValueError()
        self.n_cubes = len(self.cube)
        for i in range(self.n_cubes):
            self.cube[i] = self.cube[i].astype('float32')

        if sample_type not in ('mlar', 'tmlar', 'tmlar4d', 'pw2d', 'pw3d'):
            raise ValueError("`Sample_type` not recognized")
        self.sample_type = sample_type
        if self.sample_type == 'pw2d':
            self.sample_dim = 2
        elif self.sample_type in ('pw3d', 'mlar', 'tmlar'):
            self.sample_dim = 3
        elif self.sample_type == 'tmlar4d':
            self.sample_dim = 4

        if isinstance(pa, np.ndarray):
            if cube.ndim == 3:
                self.pa = [pa]
            elif cube.ndim == 4:
                self.pa = pa
        elif isinstance(pa, tuple):
            self.pa = list(pa)
        else:
            raise ValueError('pa must be a 1 or several (tuple) 1d ndarrays')
        if isinstance(psf, np.ndarray):
            if cube.ndim == 3:
                self.psf = [psf]
            elif cube.ndim == 4:
                self.psf = psf
        elif isinstance(pa, tuple):
            self.psf = list(psf)
        else:
            raise ValueError('psf must be a 1 or several (tuple) 2d ndarrays')

        self.fwhm = fwhm
        self.plsc = plsc
        self.patch_size = patch_size
        self.patch_size_px = int(np.ceil(self.patch_size * self.fwhm))
        if self.patch_size_px % 2 == 0:
            self.patch_size_px += 1
        print('Patch size [pixels] = {}\n'.format(self.patch_size_px))
        self.slice3d = slice3d

        self.high_pass = high_pass
        self.kernel_size = kernel_size
        if self.high_pass is not None and not reload and \
                self.sample_type in ('pw2d', 'pw3d'):
            cubehp = []
            for i in range(self.n_cubes):
                tempcu = cube_filter_highpass(self.cube[i], self.high_pass,
                                              kernel_size=self.kernel_size,
                                              verbose=False)
                tempcu = tempcu.astype('float32')
                cubehp.append(tempcu)
            self.cubehp = cubehp

        self.delta_rot = delta_rot
        self.normalization = normalization
        self.imlib = imlib
        self.interpolation = interpolation
        self.cevr_thresh = cevr_thresh
        self.n_ks = n_ks
        self.kss_window = kss_window
        self.tss_window = tss_window
        self.lr_mode = lr_mode
        self.min_adi_snr = min_snr
        self.max_adi_snr = max_snr
        self.random_seed = random_seed
        self.runtime = None
        self.augmented = False
        self.nsamp_sep = None
        self.min_n_slices = -1
        self.n_proc = int(n_proc)

        # save_filename_labdata: eg. dir_path/labda_mlar_v1
        self.labda_identifier = 'v' + str(identifier)
        self.dir_path = dir_path
        if dir_path is not None:
            self.save_filename_labdata = self.dir_path + 'labda_' + \
                                         self.sample_type + '_' + \
                                         self.labda_identifier
        else:
            self.save_filename_labdata = None

        if radius_int is None:
            self.radius_int = int(round(2 * fwhm))
        else:
            self.radius_int = radius_int

        if self.sample_type in ('mlar', 'tmlar', 'tmlar4d'):
            self.sampling_sep = int(round(fwhm))
        elif self.sample_type in ('pw2d', 'pw3d'):
            self.sampling_sep = 1
        else:
            raise ValueError("`Sample_type` not recognized")

        self.x_minus = None
        self.x_plus = None
        self.y_minus = None
        self.y_plus = None
        self.n_init_samples = -1
        self.n_total_samples = -1
        self.flo = []
        self.fhi = []
        self.distances = []
        if not reload:
            for i in range(self.n_cubes):
                print('-------')
                print('Cube {} :'.format(i + 1))
                print('-------')
                dists = self._estimate_distances(self.cube[i])
                self.distances.append(dists)
                print('')
        self.n_aug_inj = -1
        self.n_aug_aver = -1
        self.n_aug_rotshi = -1
        self.n_aug_mupcu = -1
        self.k_list = []
        self.fluxes_list = []
        self.snrs_list = []
        self.radprof = []

    def _estimate_distances(self, cube):
        """ Distances at which we grab the samples, depending on the mode.
        """
        cy, cx = frame_center(cube[0])
        if self.sample_type in ('pw2d', 'pw3d'):
            max_rad = cy - self.patch_size_px - 4 - self.radius_int
            dist = [int(d) for d in range(self.radius_int, int(max_rad))]
        elif self.sample_type in ('mlar', 'tmlar', 'tmlar4d'):
            max_rad = cy - (self.patch_size_px * 2 + self.sampling_sep) \
                      - self.radius_int
            n_annuli = int(max_rad / self.sampling_sep)
            dist = [int(self.radius_int + i * self.sampling_sep) for i
                    in range(n_annuli)]
            nsamp_init = 2
            self.nsamp_sep = [nsamp_init + i for i in range(n_annuli)]

        print('{} distances: '.format(len(dist)))
        print(dist)

        return dist

    def _make_andro_samples_ann_noise(self, cube_index, dist, n_iter=None,
                                      cube=None, pa=None):
        """ Grabbing the pw patches in a 1 px width annulus.

        Returns a list of 2d or 3d ndarrays.

        """
        if cube is None and pa is None:
            if self.high_pass is not None:
                array = self.cubehp[cube_index]
            else:
                array = self.cube[cube_index]
            angle_list = check_pa_vector(self.pa[cube_index])
        else:
            array = cube
            angle_list = check_pa_vector(pa)

        random_state = np.random.RandomState(self.random_seed)

        # pair-wise subtraction with flipped sign rotation
        res = _pairwise_diff_residuals(array, angle_list, dist, self.fwhm,
                                       self.delta_rot, debug=False)

        res_der_flipsign = cube_derotate(res, -angle_list, imlib=self.imlib,
                                         interpolation=self.interpolation)

        # grabbing patches in a 1px wide annulus
        yy, xx = get_annulus_segments(array[0].shape, dist, 1, nsegm=1)[0]
        if n_iter is None:
            n_pats = yy.shape[0]
        else:
            n_pats = n_iter
        negpat = []
        # grabbing noise pattern patches
        for i in range(n_pats):
            if n_iter is None:
                posy_neg1 = int(yy[i])
                posx_neg1 = int(xx[i])
            else:
                ind = random_state.randint(0, yy.shape[0], 1)[0]
                posy_neg1 = int(yy[ind])
                posx_neg1 = int(xx[ind])
            res_der_crop_neg = cube_crop_frames(res_der_flipsign,
                                                size=self.patch_size_px,
                                                xy=(posx_neg1, posy_neg1),
                                                verbose=False)
            res_der_crop_neg = normalize_01_pw(res_der_crop_neg,
                                               self.normalization)
            negpat.append(res_der_crop_neg)

        if self.sample_dim == 2:
            negpat = np.vstack(negpat)
        return negpat

    def _make_andro_samples_ann_signal(self, cube_index, dist, flux_low,
                                       flux_high, debug=False, n_iter=None):
        """

        Returns a list of 2d or 3d ndarrays.

        """
        array = self.cube[cube_index]

        angle_list = check_pa_vector(self.pa[cube_index])
        random_state = np.random.RandomState(self.random_seed)

        # grabbing patches in a 1px wide annulus
        yy, xx = get_annulus_segments(array[0].shape, dist, 1, nsegm=1)[0]
        if n_iter is None:
            n_pats = yy.shape[0]
        else:
            n_pats = n_iter
        flux = random_state.uniform(flux_low, flux_high, n_pats)

        pospat = []
        for i in range(n_pats):
            centy_fr, centx_fr = frame_center(array[0])
            np.random.seed()
            theta = random_state.randint(0, 360)  # a random angle is chosen

            # injection of companions
            cubefc = cube_inject_companions(array, self.psf[cube_index],
                                            -angle_list, flux[i], self.plsc,
                                            rad_dists=dist, theta=theta,
                                            n_branches=1, imlib=self.imlib,
                                            interpolation=self.interpolation,
                                            verbose=debug)

            # high-pass filtering
            if self.high_pass is not None:
                cubefc = cube_filter_highpass(cubefc, self.high_pass,
                                              kernel_size=self.kernel_size,
                                              verbose=False)

            # subtraction + de-rotation
            res = _pairwise_diff_residuals(cubefc, angle_list, dist, self.fwhm,
                                           self.delta_rot, False)
            res_der = cube_derotate(res, -angle_list, imlib=self.imlib,
                                    interpolation=self.interpolation)

            # grabbing the patch
            posy1 = dist * np.sin(np.deg2rad(theta)) + centy_fr
            posx1 = dist * np.cos(np.deg2rad(theta)) + centx_fr
            pat = cube_crop_frames(res_der, self.patch_size_px,
                                   xy=(int(posx1), int(posy1)), verbose=False)
            pat = normalize_01_pw(pat, self.normalization)
            pospat.append(pat)

        if self.sample_dim == 2:
            pospat = np.vstack(pospat)

        return pospat

    def _do_slice_3d(self, arr):
        """ Handles a list of lists of 3d ndarrays.

        # TODO: handle variable length sequences
        """
        if self.min_n_slices == -1:
            ns = []
            for i in range(len(arr)):
                for j in range(len(arr[i])):
                    ns.append(len(arr[i][j]))
            self.min_n_slices = min(ns)

        for i in range(len(arr)):
            for j in range(len(arr[i])):
                ind = np.linspace(0, len(arr[i][j]), num=self.min_n_slices,
                                  endpoint=False, dtype=int)
                arr[i][j] = arr[i][j][ind]

    def _basic_augmentation(self, n_samp_annulus, fraction_averages,
                            fraction_rotshifts, shift_amplitude,
                            fraction_mupcu):
        """ Data augmentation for creating more labeled data using simple
        strategies (mean combinations, rotations, shifting of existing samples

        Assumes X contains first the C+ samples followed by the C- ones and
        that they are balanced.

        Parameters
        ----------
        fraction_averages :
            Fraction of new negative samples wrt to the negative samples in X.

        shift_amplitude : float or None
            Between 0 and 2 px is recommended.

        """
        starttime = time_ini()
        random_state = np.random.RandomState(self.random_seed)

        news = np.round(fraction_averages + fraction_rotshifts + fraction_mupcu)
        if not news == 1:
            ms = 'Fractions of averaged samples, rotated/shifted samples and '
            ms += 'samples from the `messed-up cube` must sum up to one'
            raise ValueError(ms)

        half_initsamples = self.x_plus.shape[0]
        print('Number of input samples: {}'.format(self.n_init_samples))
        samp_dim = self.x_minus[0].ndim

        close_hdf5_files()
        fileh = tables.open_file('temp.hdf5', mode='w')
        atom = tables.Atom.from_dtype(self.x_minus.dtype)
        if self.sample_type == 'pw2d':
            dshape = (0, self.x_minus.shape[1], self.x_minus.shape[2])
        elif self.sample_type in ('pw3d', 'mlar', 'tmlar'):
            dshape = (0, self.x_minus.shape[1], self.x_minus.shape[2],
                      self.x_minus.shape[3])
        elif self.sample_type == 'tmlar4d':
            dshape = (0, self.x_minus.shape[1], self.x_minus.shape[2],
                      self.x_minus.shape[3], self.x_minus.shape[4])

        data_cpl = fileh.create_earray(where=fileh.root, name='c_plus',
                                       atom=atom, shape=dshape)
        data_cpl.append(self.x_plus)
        labe_cpl = fileh.create_earray(where=fileh.root, name='c_plus_labels',
                                       atom=atom, shape=(0,))
        labe_cpl.append(self.y_plus)

        data_cmin = fileh.create_earray(where=fileh.root, name='c_minus',
                                        atom=atom, shape=dshape)
        data_cmin.append(self.x_minus)
        labe_cmin = fileh.create_earray(where=fileh.root, name='c_minus_labels',
                                        atom=atom, shape=(0,))
        labe_cmin.append(self.y_minus)

        self.x_minus = None
        self.x_plus = None
        self.y_minus = None
        self.y_plus = None

        for i in range(self.n_cubes):
            print('\n-------')
            print("Cube {}:".format(i + 1))
            print('-------')

            # ------------------------------------------------------------------
            # More C+ samples by injecting more companions
            if self.sample_type in ('pw2d', 'pw3d'):
                print("Making new C+ samples")
                res = pool_map(self.n_proc, self._make_andro_samples_ann_signal,
                               i, fixed(self.distances[i]), fixed(self.flo[i]),
                               fixed(self.fhi[i]), False, n_samp_annulus)

                if self.sample_dim == 3 and self.slice3d:
                    self._do_slice_3d(res)
                res = np.vstack(res)
                ncplus_samples = res.shape[0]
                data_cpl.append(res)
                print('{} new C+ PW samples'.format(ncplus_samples))

            elif self.sample_type in ('mlar', 'tmlar', 'tmlar4d'):
                print("Making new C+ MLAR samples:")
                width = 1 * self.fwhm
                distances = self.distances[i]

                ncplus_samples = 0
                for d in range(len(distances)):
                    inrad = distances[d]
                    outrad = inrad + width
                    force_klen = True  # we enforce the same number of k slices
                    f = make_mlar_samples_ann_signal
                    res = f(self.cube[i], self.pa[i], self.psf[i],
                            n_samp_annulus, self.cevr_thresh, self.n_ks,
                            force_klen, inrad, outrad, self.patch_size_px,
                            self.flo[i][d], self.fhi[i][d], self.plsc,
                            self.normalization, self.n_proc, 1,
                            self.interpolation, self.lr_mode, self.sample_type,
                            self.kss_window, self.tss_window, self.random_seed)
                    data_cpl.append(res)
                    ncplus_samples += res.shape[0]
                    print('+', end='')
                print('')
                print('{} new C+ MLAR samples'.format(ncplus_samples))

            del res
            timing(starttime)

            # ------------------------------------------------------------------
            # More C- samples by mean combinations of existing patches
            ave_nsamples = int(ncplus_samples * fraction_averages)
            print("{} C- random averages".format(ave_nsamples))

            # taking 2 lists of random negative samples
            ind_ave1 = random_state.randint(0, half_initsamples, ave_nsamples)
            ind_ave2 = random_state.randint(0, half_initsamples, ave_nsamples)
            new_neg_ave = np.mean((np.array(fileh.root.c_minus)[ind_ave1],
                                   np.array(fileh.root.c_minus)[ind_ave2]),
                                  axis=0)
            data_cmin.append(new_neg_ave)
            del new_neg_ave
            timing(starttime)

            # ------------------------------------------------------------------
            # Random rotations (and shifts)
            border_mode = 'reflect'
            roshi_nsamples = int(ncplus_samples * fraction_rotshifts)
            msg = "{} C- rotations/shifts:"
            print(msg.format(roshi_nsamples))
            if self.sample_type == 'pw2d':
                new_rotshi_samples = np.empty((roshi_nsamples, dshape[1],
                                               dshape[2]))
            elif self.sample_type in ('pw3d', 'mlar', 'tmlar'):
                new_rotshi_samples = np.empty((roshi_nsamples, dshape[1],
                                               dshape[2], dshape[3]))
            elif self.sample_type == 'tmlar4d':
                new_rotshi_samples = np.empty((roshi_nsamples, dshape[1],
                                               dshape[2], dshape[3], dshape[4]))

            for j in range(roshi_nsamples):
                neg_ang = random_state.uniform(0, 359, 1)
                neg_ind = random_state.randint(0, half_initsamples, 1)[0]
                trotshi = _rotations_and_shifts(fileh.root.c_minus[neg_ind],
                                                neg_ang, samp_dim, border_mode,
                                                shift_amplitude, self.imlib,
                                                self.interpolation,
                                                self.random_seed)
                new_rotshi_samples[j] = trotshi
                if j % 100 == 0:
                    print('.', end='')
            print('')
            data_cmin.append(new_rotshi_samples)
            del new_rotshi_samples

            # ------------------------------------------------------------------
            # Samples from the messed-up cube
            resampling_factor = 1.1
            mupcu_nsamples = int(ncplus_samples - ave_nsamples - roshi_nsamples)
            print("{} C- samples from messed-up cube:".format(mupcu_nsamples))

            mupcube = _create_mupcu(self.cube[i], shift_amplitude,
                                    resampling_factor, self.imlib,
                                    self.interpolation, border_mode,
                                    self.random_seed)
            mupcu_per_annulus = int(mupcu_nsamples / len(self.distances))
            if mupcu_per_annulus < 1:
                mupcu_per_annulus = 1

            width = 1 * self.fwhm
            distances = self.distances[i]
            # mupcube = np.flip(mupcube, axis=0)
            # pa_mupcube = np.flip(self.pa[i], axis=0)
            pa_mupcube = self.pa[i]

            if self.sample_type in ('mlar', 'tmlar', 'tmlar4d'):
                for d in range(len(distances)):
                    inrad = distances[d]
                    outrad = inrad + width
                    force_klen = True  # we enforce the same number of k slices
                    f0 = make_mlar_samples_ann_noise
                    res0 = f0(mupcube, pa_mupcube, self.cevr_thresh, self.n_ks,
                              force_klen, inrad, outrad, self.patch_size_px,
                              self.fwhm, self.normalization, 1,
                              self.interpolation, self.lr_mode,
                              self.sample_type, self.kss_window,
                              self.tss_window, self.random_seed,
                              nsamp_sep=mupcu_per_annulus)
                    res0 = res0[0]
                    print('.', end='')
                    if d == 0:
                        new_mupcu_samps = res0
                    else:
                        new_mupcu_samps = np.concatenate((new_mupcu_samps, res0),
                                                         axis=0)
                print('')

            elif self.sample_type in ('pw2d', 'pw3d'):
                negs = pool_map(self.n_proc, self._make_andro_samples_ann_noise,
                                i, fixed(self.distances[i]), mupcu_per_annulus,
                                mupcube, pa_mupcube)

                # list of lists of 3d ndarrays
                if self.sample_dim == 3 and self.slice3d:
                    self._do_slice_3d(negs)
                new_mupcu_samps = np.vstack(negs)

            data_cmin.append(new_mupcu_samps[:mupcu_nsamples])
            labe_cmin.append(np.zeros(ncplus_samples))
            labe_cpl.append(np.ones(ncplus_samples))
            timing(starttime)

        n_total_aug = ncplus_samples * 2 * self.n_cubes
        n_total_samples = self.n_init_samples + n_total_aug
        self.n_total_samples = n_total_samples
        self.x_minus = np.array(fileh.root.c_minus, dtype='float32')
        self.x_plus = np.array(fileh.root.c_plus, dtype='float32')
        self.y_minus = np.array(fileh.root.c_minus_labels, dtype='float32')
        self.y_plus = np.array(fileh.root.c_plus_labels, dtype='float32')
        fileh.close()
        os.remove('./temp.hdf5')

        print('Total number of samples: {}'.format(self.n_total_samples))

        self.augmented = True
        self.n_aug_inj = ncplus_samples * self.n_cubes
        self.n_aug_aver = ave_nsamples * self.n_cubes
        self.n_aug_rotshi = roshi_nsamples * self.n_cubes
        self.n_aug_mupcu = mupcu_nsamples * self.n_cubes

        self.runtime = time_fin(starttime)
        timing(starttime)

    def augment(self, mode='basic', n_samp_annulus=10, fraction_averages=0.6,
                fraction_rotshifts=0.2, shift_amplitude=0.5,
                fraction_mupcu=0.2):
        """
        """
        if mode == 'basic':
            if self.x_minus is None:
                raise RuntimeError("You must first run the DataLabeler")

            if self.sample_type in ('pw2d', 'pw3d'):
                if self.sample_type == 'pw2d' and n_samp_annulus > 500:
                    raise ValueError("`N_samp_annulus` is too high")
                if self.sample_type == 'pw3d' and n_samp_annulus > 10000:
                    raise ValueError("`N_samp_annulus` is too high")
            self._basic_augmentation(n_samp_annulus, fraction_averages,
                                     fraction_rotshifts, shift_amplitude,
                                     fraction_mupcu)
        else:
            print("Data augmentation mode not recognized")

        if self.save_filename_labdata is not None:
            self.save(self.save_filename_labdata)

    def inspect_samples(self, index=None, max_slices=None, n_samples=5,
                        init_sample=None, cmap='bone', dpi=10, **kwargs):
        """
        """
        if self.sample_type in ('pw3d', 'mlar', 'tmlar', 'tmlar4d'):
            if index is None:
                if init_sample is None:
                    init_sample = np.random.randint(50, size=1)[0]
                index = np.linspace(init_sample, self.x_plus.shape[0],
                                    n_samples, endpoint=False, dtype=int)
                msg1 = 'Labels: {}'
                msg0 = 'Indices: {}'
            else:
                index = [index]
                msg1 = 'Label: {}'
                msg0 = 'Index: {}'

            n_slices = self.x_plus[0].shape[0]
            print(msg0.format(index))
            if max_slices is None or max_slices >= n_slices:
                show_n_slices = n_slices
                ind_slices = range(self.x_plus[0].shape[0])
            elif max_slices < n_slices:
                show_n_slices = max_slices
                ind_slices = np.sort(np.random.choice(n_slices, show_n_slices,
                                                      False))

            if self.sample_type == 'tmlar4d':
                print(msg1.format(self.y_plus[index]))
                for i in range(len(index)):
                    for j in range(self.x_plus[index[i]].shape[1]):
                        pp_subplots(self.x_plus[index[i], ind_slices, j],
                                    maxplots=show_n_slices, axis=False,
                                    horsp=0.05, colorb=False, cmap=cmap,
                                    dpi=dpi, **kwargs)
                    print('')

                print(msg1.format(self.y_minus[index]))
                for i in range(len(index)):
                    for j in range(self.x_minus[index[i]].shape[1]):
                        pp_subplots(self.x_minus[index[i], ind_slices, j],
                                    maxplots=show_n_slices, axis=False,
                                    horsp=0.05, colorb=False, cmap=cmap,
                                    dpi=dpi, **kwargs)
                    print('')
            else:
                print(msg1.format(self.y_plus[index]))
                for i in range(len(index)):
                    pp_subplots(self.x_plus[index[i]][ind_slices],
                                maxplots=show_n_slices, axis=False, horsp=0.05,
                                colorb=False, cmap=cmap, dpi=dpi, **kwargs)

                print(msg1.format(self.y_minus[index]))
                for i in range(len(index)):
                    pp_subplots(self.x_minus[index[i]][ind_slices],
                                maxplots=show_n_slices, axis=False, horsp=0.05,
                                colorb=False, cmap=cmap, dpi=dpi, **kwargs)

        elif self.sample_type == 'pw2d':
            if index is None:
                if init_sample is None:
                    init_sample = np.random.randint(50, size=1)[0]
                index = np.linspace(init_sample, self.x_plus.shape[0],
                                    n_samples, endpoint=False, dtype=int)
                msg1 = 'Labels: {}'
                msg0 = 'Indices: {}'
            else:
                index = [index]
                msg1 = 'Label: {}'
                msg0 = 'Index: {}'

            print(msg0.format(index))
            print(msg1.format(self.y_plus[index]))
            pp_subplots(self.x_plus[index], maxplots=n_samples,
                        axis=False, horsp=0.05, colorb=False, cmap=cmap,
                        **kwargs)
            print(msg1.format(self.y_minus[index]))
            pp_subplots(self.x_minus[index], maxplots=n_samples,
                        axis=False, horsp=0.05, colorb=False, cmap=cmap,
                        **kwargs)

    # def estimate_fluxes(self, n_injections=100, kernel='rbf', epsilon=0.1,
    #                     c=1e5, gamma=1e-2, n_proc=None, figsize=(10, 5),
    #                     dpi=100, **kwargs):
    def estimate_fluxes(self, algo='pca', n_injections=100, kernel='rbf',
                        epsilon=0.1, c=1e4, gamma=1e-2, n_proc=None,
                        figsize=(10, 5), dpi=100, **kwargs):
        """
        """
        if n_proc is None:
            n_proc = copy.copy(self.n_proc)

        for i in range(self.n_cubes):
            print('-------')
            print('Cube {} :'.format(i + 1))
            print('-------')
            global GARRAY
            GARRAY = np.array(self.cube[i])
            global GARRPSF
            GARRPSF = np.array(self.psf[i])
            global GARRPA
            GARRPA = np.array(self.pa[i])
            global GARRDIST
            GARRDIST = np.array(self.distances[i])

            if len(self.snrs_list) < i + 1:
                if self.sample_type in ('mlar', 'tmlar', 'tmlar4d'):
                    distances = GARRDIST
                    inter_extrap = False
                elif self.sample_type in ('pw2d', 'pw3d'):
                    cy, cx = frame_center(GARRAY[0])
                    sampling_sep = int(round(self.fwhm))
                    max_rad = cy - (self.patch_size_px * 2 + sampling_sep)
                    n_annuli = int(max_rad / sampling_sep)
                    # same distances as with the mlar case
                    distances = [int(self.radius_int + i * sampling_sep) for i
                                 in range(n_annuli)]
                    inter_extrap = True

                fluxest = FluxEstimator(GARRAY, GARRPSF, distances, GARRPA,
                                        self.fwhm, self.plsc, wavelengths=None,
                                        n_injections=n_injections,
                                        algo=algo, min_snr=self.min_adi_snr,
                                        max_snr=self.max_adi_snr,
                                        inter_extrap=inter_extrap,
                                        inter_extrap_dist=GARRDIST,
                                        random_seed=self.random_seed,
                                        n_proc=n_proc)
                fluxest.sampling()
                self.fluxes_list.append(fluxest.fluxes_list)
                self.snrs_list.append(fluxest.snrs_list)
                self.radprof.append(fluxest.radprof)

            fluxest.run(kernel=kernel, epsilon=epsilon, c=c, gamma=gamma,
                        figsize=figsize, dpi=dpi, **kwargs)

            self.flo.append(fluxest.estimated_fluxes_low)
            self.fhi.append(fluxest.estimated_fluxes_high)
            print('\n')

    def run(self):
        """
        """
        starttime = time_ini()

        if len(self.fhi) == 0:
            raise RuntimeError('Execute `estimate_fluxes` before `run`')

        close_hdf5_files()
        with tables.open_file('temp.hdf5', mode='w') as fhd:
            for i in range(self.n_cubes):
                print('\n-------')
                print("Cube {}:".format(i + 1))
                print('-------')

                ################################################################
                # SODINN-SVD
                ################################################################
                if self.sample_type in ('mlar', 'tmlar', 'tmlar4d'):
                    width = 1 * self.fwhm
                    self.k_list = []

                    print("Making C+ and C- MLAR samples for each annulus:")
                    distances = self.distances[i]
                    for d in range(len(distances)):
                        inrad = distances[d]
                        outrad = inrad + width

                        if i == 0 and d == 0:
                            force_klen = False
                        else:
                            force_klen = True

                        # Grabbing all the patches in the current annulus
                        f0 = make_mlar_samples_ann_noise
                        res0 = f0(self.cube[i], self.pa[i], self.cevr_thresh,
                                  self.n_ks, force_klen, inrad, outrad,
                                  self.patch_size_px, self.fwhm,
                                  self.normalization, 1, self.interpolation,
                                  self.lr_mode, self.sample_type,
                                  self.kss_window, self.tss_window,
                                  self.random_seed)

                        self.k_list.append(res0[1])
                        if i == 0 and d == 0:
                            if self.n_ks > res0[2]:
                                msg = "Cannot grab {} Ks. Only {} available " \
                                      "for CEVR = {}"
                                print(msg.format(self.n_ks, res0[2],
                                                 self.cevr_thresh))
                            self.n_ks = res0[2]

                        res0 = res0[0]
                        n_samp_annulus = res0.shape[0]
                        print('-', end='')

                        # Injecting the same number of companions to balance
                        f1 = make_mlar_samples_ann_signal
                        res1 = f1(self.cube[i], self.pa[i], self.psf[i],
                                  n_samp_annulus, self.cevr_thresh, self.n_ks,
                                  force_klen, inrad, outrad, self.patch_size_px,
                                  self.flo[i][d], self.fhi[i][d], self.plsc,
                                  self.normalization, self.n_proc, 1,
                                  self.interpolation, self.lr_mode,
                                  self.sample_type, self.kss_window,
                                  self.tss_window, self.random_seed)
                        print('+', end='')

                        # Saving data to HDF5 file
                        if i == 0 and d == 0:
                            atom = tables.Atom.from_dtype(res0.dtype)
                            if self.sample_type in ('mlar', 'tmlar'):
                                dshape = (0, res0.shape[1], res0.shape[2],
                                          res0.shape[3])
                            elif self.sample_type == 'tmlar4d':
                                dshape = (0, res0.shape[1], res0.shape[2],
                                          res0.shape[3], res0.shape[4])
                            data_cmin = fhd.create_earray(where=fhd.root,
                                                          name='c_minus',
                                                          atom=atom,
                                                          shape=dshape)
                            labe_cmin = fhd.create_earray(where=fhd.root,
                                                          name='c_minus_labels',
                                                          atom=atom, shape=(0,))
                            data_cpl = fhd.create_earray(where=fhd.root,
                                                         name='c_plus',
                                                         atom=atom,shape=dshape)
                            labe_cpl = fhd.create_earray(where=fhd.root,
                                                         name='c_plus_labels',
                                                         atom=atom, shape=(0,))
                        data_cmin.append(res0)
                        labe_cmin.append(np.zeros((res0.shape[0])))
                        data_cpl.append(res1)
                        labe_cpl.append(np.ones((res1.shape[0])))
                    print('')

                ################################################################
                # SODINN-PW
                ################################################################
                elif self.sample_type in ('pw2d', 'pw3d'):

                    print("Making C- samples")
                    negs = pool_map(self.n_proc,
                                    self._make_andro_samples_ann_noise, i,
                                    fixed(self.distances[i]))

                    # list of lists of 2d or 3d ndarrays
                    if self.sample_dim == 3 and self.slice3d:
                        self._do_slice_3d(negs)
                    negs = np.vstack(negs)

                    if self.sample_dim == 2:
                        dshape = (0, negs.shape[1], negs.shape[2])
                    elif self.sample_dim == 3:
                        dshape = (0, negs.shape[1], negs.shape[2],
                                  negs.shape[3])

                    # Saving data to HDF5 file
                    if i == 0:
                        atom = tables.Atom.from_dtype(negs.dtype)
                        data_cmin = fhd.create_earray(where=fhd.root,
                                                      name='c_minus', atom=atom,
                                                      shape=dshape)
                        labe_cmin = fhd.create_earray(where=fhd.root,
                                                      name='c_minus_labels',
                                                      atom=atom, shape=(0,))
                        data_cpl = fhd.create_earray(where=fhd.root,
                                                     name='c_plus', atom=atom,
                                                     shape=dshape)
                        labe_cpl = fhd.create_earray(where=fhd.root,
                                                     name='c_plus_labels',
                                                     atom=atom, shape=(0,))
                    data_cmin.append(negs)
                    labe_cmin.append(np.zeros((negs.shape[0])))
                    del negs
                    timing(starttime)

                    print("Making C+ samples")
                    pos = pool_map(self.n_proc,
                                   self._make_andro_samples_ann_signal, i,
                                   fixed(self.distances[i]), fixed(self.flo[i]),
                                   fixed(self.fhi[i]), False, None)

                    if self.sample_dim == 3 and self.slice3d:
                        self._do_slice_3d(pos)
                    pos = np.vstack(pos)

                    data_cpl.append(pos)
                    labe_cpl.append(np.ones((pos.shape[0])))
                    del pos

                    timing(starttime)

                else:
                    raise ValueError('sample_type not recognized')

            cmin_shape = fhd.root.c_plus.shape
            cplu_shape = fhd.root.c_minus.shape
            self.n_init_samples = int(cmin_shape[0] + cplu_shape[0])
            msg = "\nShape of C+ samples array: {}"
            print(msg.format(cmin_shape))
            msg = "Shape of C- samples array: {}"
            print(msg.format(cplu_shape))
            msg = "Total number of samples: {}"
            print(msg.format(self.n_init_samples))

            self.x_plus = np.array(fhd.root.c_plus, dtype='float32')
            self.x_minus = np.array(fhd.root.c_minus, dtype='float32')
            self.y_plus = np.array(fhd.root.c_plus_labels, dtype='float32')
            self.y_minus = np.array(fhd.root.c_minus_labels, dtype='float32')

            self.runtime = time_fin(starttime)
            if self.save_filename_labdata is not None:
                self.save(self.save_filename_labdata)
            timing(starttime)

        os.remove('./temp.hdf5')

    def save(self, filename):
        """
        # return dill.dump(self, open(filename + '.pkl', 'wb'), protocol=4)
        """
        if self.runtime is None:
            raise RuntimeError('The DataLabeler has not been executed (.run())')

        # Creating HDF5 file
        if not filename.endswith('.hdf5'):
            filename += '.hdf5'

        with tables.open_file(filename, mode='w') as fh5:
            # Writing to HDF5 file
            for key in self.__dict__.keys():
                attr = self.__dict__[key]
                f32atom = tables.Float32Atom()

                if attr is None:
                    attr = np.char.array('None')
                    _ = fh5.create_array('/', key, obj=attr)

                elif isinstance(attr, str):
                    attr = np.char.array(attr)
                    _ = fh5.create_array('/', key, obj=attr)

                elif isinstance(attr, np.ndarray):
                    if attr.dtype == 'float64':
                        attr = attr.astype('float32')
                    _ = fh5.create_array('/', key, obj=attr, atom=f32atom)

                elif isinstance(attr, list):
                    if isinstance(attr[0], np.ndarray):
                        if attr[0].dtype in ('float32', 'float64'):
                            attr = np.array(attr, dtype='float32')
                            _ = fh5.create_array('/', key, obj=attr,
                                                 atom=f32atom)
                    else:
                        _ = fh5.create_array('/', key, obj=attr)

                elif isinstance(attr, tuple):
                    if isinstance(attr[0], np.ndarray):
                        attr = np.array(attr, dtype='float32')
                        _ = fh5.create_array('/', key, obj=attr, atom=f32atom)
                    elif isinstance(attr[0], int):
                        attr = np.array(attr, dtype='int')
                        _ = fh5.create_array('/', key, obj=attr)
                    elif isinstance(attr[0], tuple):
                        attr = np.array(attr)
                        _ = fh5.create_array('/', key, obj=attr)
                    else:
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
        # Opening HDF5 file
        if not filename.endswith('.hdf5'):
            filename += '.hdf5'
        fh5 = tables.open_file(filename, mode='r')
        fh5r = fh5.root

        identifier = int(str(fh5r.labda_identifier[0].decode()).split('v')[-1])
        # dir_path = str(fh5r.dir_path[0].decode())
        dir_path = str(fh5r.save_filename_labdata[0].decode()).split('labda')[0]

        try:
            kss_window = fh5r.kss_window.read()
            if not isinstance(kss_window, int):
                kss_window = None
        except:
            kss_window = None
        try:
            tss_window = fh5r.tss_window.read()
            if not isinstance(tss_window, int):
                tss_window = None
        except:
            tss_window = None

        obj = cls(sample_type=str(fh5r.sample_type[0].decode()),
                  cube=np.array(fh5r.cube),
                  pa=np.array(fh5r.pa), psf=np.array(fh5r.psf),
                  radius_int=fh5r.radius_int.read(), fwhm=fh5r.fwhm.read(),
                  plsc=fh5r.plsc.read(),
                  delta_rot=fh5r.delta_rot.read(),
                  patch_size=fh5r.patch_size.read(),
                  slice3d=fh5r.slice3d.read(),
                  high_pass=str(fh5r.high_pass[0].decode()),
                  kernel_size=fh5r.kernel_size.read(),
                  normalization=str(fh5r.normalization[0].decode()),
                  min_adi_snr=fh5r.min_adi_snr.read(),
                  max_adi_snr=fh5r.max_adi_snr.read(),
                  cevr_thresh=fh5r.cevr_thresh.read(), n_ks=fh5r.n_ks.read(),
                  kss_window=kss_window, tss_window=tss_window,
                  lr_mode=str(fh5r.lr_mode[0].decode()),
                  imlib=str(fh5r.imlib[0].decode()),
                  interpolation=str(fh5r.interpolation[0].decode()),
                  n_proc=fh5r.n_proc.read(),
                  random_seed=fh5r.random_seed.read(), identifier=identifier,
                  dir_path=dir_path, reload=True)

        obj.augmented = fh5r.augmented.read()
        if hasattr(fh5r, 'cubehp'):
            obj.cubehp = np.array(fh5r.cubehp)
        obj.distances = fh5r.distances.read()
        obj.fhi = fh5r.fhi.read()
        obj.flo = fh5r.flo.read()
        obj.min_n_slices = fh5r.min_n_slices.read()
        obj.n_aug_aver = fh5r.n_aug_aver.read()
        obj.n_aug_inj = fh5r.n_aug_inj.read()
        obj.n_aug_mupcu = fh5r.n_aug_mupcu.read()
        obj.n_aug_rotshi = fh5r.n_aug_rotshi.read()
        obj.n_cubes = fh5r.n_cubes.read()
        obj.n_init_samples = fh5r.n_init_samples.read()
        obj.n_total_samples = fh5r.n_total_samples.read()
        obj.nsamp_sep = fh5r.nsamp_sep.read()
        obj.runtime = str(fh5r.runtime[0].decode())
        obj.sampling_sep = fh5r.sampling_sep.read()
        obj.x_minus = fh5r.x_minus.read()
        obj.x_plus = fh5r.x_plus.read()
        obj.y_minus = fh5r.y_minus.read()
        obj.y_plus = fh5r.y_plus.read()

        fh5.close()
        return obj


def _create_mupcu(cube, shift_amplitude, upsampling_factor, imlib,
                  interpolation, border_mode, random_seed):
    """
    """
    random_state = np.random.RandomState(random_seed)
    cube_out = np.zeros_like(cube)
    for i in range(cube.shape[0]):
        shy = random_state.uniform(-shift_amplitude, shift_amplitude, 1)
        shx = random_state.uniform(-shift_amplitude, shift_amplitude, 1)
        tempshi = frame_shift(cube[i], shy, shx, imlib, interpolation,
                              border_mode)
        scale = random_state.uniform(upsampling_factor - 0.05,
                                     upsampling_factor + 0.05, 1)[0]
        tempresc = frame_px_resampling(tempshi, scale, imlib, interpolation)
        pad = cube[i].shape[0] - tempresc.shape[0]
        if pad > 0:
            tempresc = np.pad(tempresc, pad, 'reflect')
        cube_out[i] = frame_crop(tempresc, cube_out.shape[1], force=True,
                                 verbose=False)

    return cube_out


def _pairwise_diff_residuals(array, angle_list, ann_center, fwhm, delta_rot=0.5,
                             debug=False):
    """

    Parameters
    ----------
    cube : array_like, 3d
        Input cube.
    angle_list : array_like, 1d
        Corresponding parallactic angle for each frame.
    fwhm : float, optional
        Known size of the FHWM in pixels to be used. Default is 4.
    delta_rot : float, optional
        Minimum parallactic angle distance between the pairs.
    inner_radius : int, optional
        The radius of the innermost annulus. By default is 0, if >0 then the
        central circular area is discarded.
    asize : int, optional
        The size of the annuli, in pixels.
    verbose: bool, optional
        If True prints info to stdout.
    debug : bool, optional
        If True the distance matrices will be plotted and additional information
        will be given.

    Returns
    -------
    res : array_like, 3d
        Cube of residuals

    """
    n_frames = array.shape[0]
    y = array.shape[1]
    if not fwhm < y // 2:
        raise ValueError("asize is too large")

    pa_threshold = _compute_pa_thresh(ann_center, fwhm, delta_rot)
    if debug:
        print('PA thresh {:.3f}'.format(pa_threshold))

    # annulus-wise pair-wise subtraction
    res = []
    for i in range(n_frames):
        indp, indn = _find_indices_adi(angle_list, i, pa_threshold,
                                       out_closest=True)
        if debug:
            print(indp, indn)
        if i == indn:
            indn += 1
        res.append(array[i] - array[indn])
        if indn == n_frames - 1:
            break

    return np.array(res)


def _rotations_and_shifts(array, ang, samp_dim, border_mode, shift_amplitude,
                          imlib, interpolation, random_seed):
    """
    """
    random_state = np.random.RandomState(random_seed)
    if samp_dim == 4:
        n_slices = array.shape[0]
        ang = ang * np.ones(n_slices)
        samp_rot = []

        for i in range(n_slices):
            slice3drot = cube_derotate(array[i], ang, imlib=imlib,
                                       interpolation=interpolation,
                                       border_mode=border_mode)
            if shift_amplitude is not None:
                # random shift pixels in x and y
                shy = random_state.uniform(-shift_amplitude, shift_amplitude, 1)
                shx = random_state.uniform(-shift_amplitude, shift_amplitude, 1)
                slice3drot = cube_shift(slice3drot, shy, shx, imlib,
                                        interpolation, border_mode=border_mode)
            samp_rot.append(slice3drot)
        samp_rot = np.array(samp_rot)

    elif samp_dim == 3:
        n_slices = array.shape[0]
        ang = ang * np.ones(n_slices)
        samp_rot = cube_derotate(array, ang, imlib=imlib,
                                 interpolation=interpolation,
                                 border_mode=border_mode)
        if shift_amplitude is not None:
            # random shift pixels in x and y
            shy = random_state.uniform(-shift_amplitude, shift_amplitude, 1)
            shx = random_state.uniform(-shift_amplitude, shift_amplitude, 1)
            samp_rot = cube_shift(samp_rot, shy, shx, imlib, interpolation,
                                  border_mode=border_mode)

    elif samp_dim == 2:
        samp_rot = frame_rotate(array, ang, imlib=imlib,
                                interpolation=interpolation,
                                border_mode=border_mode)
        if shift_amplitude is not None:
            # random shift pixels in x and y
            shy = random_state.uniform(-shift_amplitude, shift_amplitude, 1)
            shx = random_state.uniform(-shift_amplitude, shift_amplitude, 1)
            samp_rot = frame_shift(samp_rot, shy, shx, imlib, interpolation,
                                   border_mode=border_mode)

    return samp_rot


def _concat_training_data(*args):
    """
    """
    x_list = []
    y_list = []
    r = iter(range(len(args)))
    for i in r:
        if isinstance(args[i], tuple):
            x, y = args[i]
        else:
            x, y = args[i:i+2]
            next(r)

        if isinstance(y, (int, np.integer)):
            y = np.zeros(x.shape[0]) + y

        x_list.append(x.astype(np.float32))
        y_list.append(y.astype(np.float32))

    x = np.concatenate(x_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    return x, y


def _sample_flux_snr(distances, fwhm, plsc, n_injections, flux_min, flux_max,
                     nproc=10, random_seed=42):
    """
    Sensible flux intervals depend on a combination of factors, # of frames,
    range of rotation, correlation, glare intensity.
    """
    starttime = time_ini()
    frsize = int(GARRAY.shape[1])
    ninj = n_injections
    random_state = np.random.RandomState(random_seed)
    flux_dist_theta_all = list()
    snrs_list = list()
    fluxes_list = list()

    for i, d in enumerate(distances):
        yy, xx = get_annulus_segments((frsize, frsize), d, 1, 1)[0]
        num_patches = yy.shape[0]

        fluxes_dist = random_state.uniform(flux_min[i], flux_max[i], size=ninj)
        inds_inj = random_state.randint(0, num_patches, size=ninj)

        for j in range(ninj):
            injx = xx[inds_inj[j]]
            injy = yy[inds_inj[j]]
            injx -= frame_center(GARRAY[0])[1]
            injy -= frame_center(GARRAY[0])[0]
            dist = np.sqrt(injx ** 2 + injy ** 2)
            theta = np.mod(np.arctan2(injy, injx) / np.pi * 180, 360)
            flux_dist_theta_all.append((fluxes_dist[j], dist, theta))

    # Multiprocessing pool
    res = pool_map(nproc, _get_adi_snrs, GARRPSF, GARRPA, fwhm, plsc,
                   fixed(flux_dist_theta_all))

    for i in range(len(distances)):
        flux_dist = []
        snr_dist = []
        for j in range(ninj):
            flux_dist.append(res[j + (ninj * i)][0])
            snr_dist.append(res[j + (ninj * i)][1])
        fluxes_list.append(flux_dist)
        snrs_list.append(snr_dist)

    timing(starttime)
    return fluxes_list, snrs_list


def _compute_residual_frame(cube, angle_list, radius, fwhm, mode='pca', ncomp=2,
                            svd_mode='lapack', scaling=None, collapse='median',
                            imlib='opencv', interpolation='lanczos4'):
    """
    """
    if cube.ndim == 3:
        if mode == 'pca':
            annulus_width = 3 * fwhm
            data, ind = prepare_matrix(cube, scaling, mode='annular',
                                       annulus_radius=radius, verbose=False,
                                       annulus_width=annulus_width)
            yy, xx = ind
            V = svd_wrapper(data, svd_mode, ncomp, False, False)
            transformed = np.dot(V, data.T)
            reconstructed = np.dot(transformed.T, V)
            residuals = data - reconstructed
            cube_empty = np.zeros_like(cube)
            cube_empty[:, yy, xx] = residuals
            cube_res_der = cube_derotate(cube_empty, angle_list, imlib=imlib,
                                         interpolation=interpolation)
            res_frame = cube_collapse(cube_res_der, mode=collapse)

        elif mode == 'median':
            res_frame = median_sub(cube, angle_list, verbose=False)

    elif cube.ndim == 4:
        if mode == 'pca':
            pass

        elif mode == 'median':
            res_frame = median_sub(cube, angle_list, verbose=False)

    return res_frame


def _get_adi_snrs(psf, angle_list, fwhm, plsc, flux_dist_theta_all, mode,
                  ncomp):
    """ Get the mean S/N (at 3 equidistant positions) for a given flux and
    distance, on a median/PCA subtracted frame.
    """
    snrs = []
    theta = flux_dist_theta_all[2]
    flux = flux_dist_theta_all[0]
    dist = flux_dist_theta_all[1]

    # 3 equidistant azimuthal positions
    for ang in [theta, theta + 120, theta + 240]:
        cube_fc, cx, cy = create_synt_cube(GARRAY, psf, angle_list, plsc,
                                           flux=flux, dist=dist, theta=ang,
                                           verbose=False)
        fr_temp = _compute_residual_frame(cube_fc, angle_list, dist, fwhm,
                                          mode, ncomp, svd_mode='lapack',
                                          scaling=None, collapse='median',
                                          imlib='opencv',
                                          interpolation='lanczos4')
        res = frame_quick_report(fr_temp, fwhm, source_xy=(cx, cy),
                                 verbose=False)
        # mean S/N in circular aperture
        snrs.append(np.mean(res[-1]))

    # median of mean S/N at 3 equidistant positions
    median_snr = np.median(snrs)
    return flux, median_snr



