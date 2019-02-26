"""

"""
from __future__ import print_function, division, absolute_import

__all__ = ['FluxEstimator']

import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate
from scipy.interpolate import interp1d
from vip_hci.conf import time_ini, timing, time_fin
from vip_hci.var import frame_center
from vip_hci.stats import frame_average_radprofile
from vip_hci.conf.utils_conf import (pool_map, fixed, make_chunks)
from vip_hci.var import (cube_filter_highpass, pp_subplots,
                         get_annulus_segments, prepare_matrix)
from vip_hci.metrics import cube_inject_companions
from vip_hci.preproc import (check_pa_vector, cube_derotate, cube_crop_frames,
                             frame_rotate, frame_shift, frame_px_resampling,
                             frame_crop, cube_collapse, check_pa_vector,
                             check_scal_vector)
from vip_hci.preproc import cube_rescaling_wavelengths as scwave
from vip_hci.metrics import frame_quick_report
from vip_hci.medsub import median_sub
from vip_hci.pca import pca, svd_wrapper

import warnings
# To silence UserWarning when scaling data with sklearn
warnings.filterwarnings("ignore")


class FluxEstimator:
    """
    Fluxes (proxy of contrast) estimator for injecting fake companions.
    """
    def __init__(self, cube, psf, distances, angles, fwhm, plsc,
                 wavelengths=None, n_injections=30, algo='pca', min_snr=1,
                 max_snr=3, inter_extrap=False, inter_extrap_dist=None,
                 random_seed=42, n_proc=2):
        """ Initialization of the flux estimator object.

        Parameters
        ----------
        cube : array_like, 3d or 4d
            Input sequence (ADI or IFS+ADI).
        psf : array_like, 2d or 3d
            Input corresponding template PSF.
        distances : list
            Distances from the center at which the fluxes will be estimated.
        angles : array_like, 1d
            Corresponding vector or parallactic angles.
        fwhm : int or float
            FWHM for the input dataset.
        plsc : float
            Plate scale for the input dataset.
        wavelengths : array_like, 1d
            Wavelengths for the input dataset (in case of a 4d array).
        n_injections : int, optional
            Number of fake companion injections for sampling the flux vs SNR
            dependency.
        algo : {'pca', 'median'}, str optional
            Algorithm to be used as a baseline for obtaining SNRs. 'pca' for a
            principal component analysis based post-processing. 'median' for a
            median subtraction approach.
        min_snr : int, optional
            Minimum target SNR for which a flux will be estimated at given
            distances.
        max_snr : int, optional
            Maximum target SNR for which a flux will be estimated at given
            distances.
        inter_extrap : {False, True}, bool optional
            Whether to inter/extrapolate the estimated fluxes for higher
            sampling. Only valid when ``len(distances) > 2``.
        inter_extrap_dist : array_like 1d or list
            New distances for inter/extrapolate the estimated fluxes.
        random_seed : int, optional
            Random seed.
        n_proc : int, optional
            Number of processes to be used.
        """
        global GARRAY
        global GARRPSF
        global GARRWL
        global GARRPA
        GARRAY = cube
        GARRPSF = psf
        GARRPA = angles
        GARRWL = wavelengths
        self.min_fluxes = None
        self.max_fluxes = None
        self.radprof = None
        self.sampled_fluxes = None
        self.sampled_snrs = None
        self.estimated_fluxes_low = None
        self.estimated_fluxes_high = None
        self.distances = distances
        self.angles = angles
        self.fwhm = fwhm
        self.plsc = plsc
        self.scaling = 'temp-standard'
        self.wavelengths = wavelengths
        self.n_injections = n_injections
        self.algo = algo
        self.min_snr = min_snr
        self.max_snr = max_snr
        self.random_seed = random_seed
        self.n_proc = n_proc
        self.inter_extrap = inter_extrap
        self.inter_extrap_dist = inter_extrap_dist
        self.n_dist = range(len(self.distances))
        self.fluxes_list = list()
        self.snrs_list = list()

        if cube.ndim == 4:
            if wavelengths is None:
                raise ValueError('`wavelengths` parameter must be provided')

    def get_min_flux(self):
        """ Obtaining the low end of the interval for sampling the SNRs. Based
        on the initial estimation of the radial profile of the mean frame.
        """
        starttime = time_ini()

        # Getting the radial profile in the mean frame of the cube
        sampling_sep = 1
        radius_int = 1
        if GARRAY.ndim == 3:
            global_frame = np.mean(GARRAY, axis=0)
        elif GARRAY.ndim == 4:
            global_frame = np.mean(GARRAY.reshape(-1, GARRAY.shape[2],
                                                  GARRAY.shape[3]), axis=0)

        me = frame_average_radprofile(global_frame, sep=sampling_sep,
                                      init_rad=radius_int, plot=False)
        radprof = np.array(me.radprof)
        radprof = radprof[np.array(self.distances) + 1]
        radprof[radprof < 0] = 0.01
        self.radprof = radprof

        print("Estimating the min values for sampling the S/N vs flux function")
        flux_min = pool_map(self.n_proc, _get_min_flux, fixed(self.n_dist),
                            self.distances, radprof, self.fwhm, self.plsc,
                            self.min_snr, self.wavelengths, self.algo,
                            self.scaling, self.random_seed)

        self.min_fluxes = flux_min
        timing(starttime)

    def get_max_flux(self):
        """ Obtaining the high end of the interval for sampling the SNRs.
        """
        if self.min_fluxes is None:
            self.get_min_flux()

        starttime = time_ini()

        print("Estimating the max values for sampling the S/N vs flux function")
        flux_max = pool_map(self.n_proc, _get_max_flux, fixed(self.n_dist),
                            self.distances, self.min_fluxes, self.fwhm,
                            self.plsc, self.max_snr, self.wavelengths,
                            self.algo, self.scaling, self.random_seed)

        self.max_fluxes = flux_max
        timing(starttime)

    def sampling(self):
        """ Using the computed interval of fluxes for sampling the flux vs SNR
        relationship.
        """
        if not self.min_fluxes:
            self.get_min_flux()

        if not self.max_fluxes:
            self.get_max_flux()

        starttime = time_ini()
        print("Sampling by injecting fake companions")
        res = _sample_flux_snr(self.distances, self.fwhm, self.plsc,
                               self.n_injections, self.min_fluxes,
                               self.max_fluxes, self.n_proc,
                               self.random_seed, self.wavelengths, self.algo,
                               self.scaling)
        self.sampled_fluxes, self.sampled_snrs = res
        timing(starttime)

    def run(self, dpi=100):
        """ Obtaining the flux vs S/N relationship.

        dpi : int, optional
            DPI of the figures.
        """
        if not self.sampled_fluxes or not self.sampled_snrs:
            self.sampling()

        starttime = time_ini()

        plotvlines = [self.min_snr, self.max_snr]
        nsubplots = len(self.distances)
        ncols = min(4, nsubplots)

        if nsubplots > 1 and nsubplots % 2 != 0:
            nsubplots -= 1

        if nsubplots < 3:
            figsize = (10, 2)
            if nsubplots == 2:
                figsizex = figsize[0] * 0.66
            elif nsubplots == 1:
                figsizex = figsize[0] * 0.33
            nrows = 1
        else:
            if nsubplots <= 8:
                figsize = (10, 4)
            else:
                figsize = (10, 6)
            figsizex = figsize[0]
            nrows = int(nsubplots / ncols) + 1

        fig, axs = plt.subplots(nrows, ncols, figsize=(figsizex, figsize[1]),
                                dpi=dpi, sharey='row')
        fig.subplots_adjust(wspace=0.05, hspace=0.3)
        if isinstance(axs, np.ndarray):
            axs = axs.ravel()
        fhi = list()
        flo = list()

        print("Building the regression models for each separation")
        # Regression for each distance
        for i, d in enumerate(self.distances):
            if isinstance(axs, np.ndarray):
                axis = axs[i]
            else:
                axis = axs

            fluxes = np.array(self.sampled_fluxes[i])
            snrs = np.array(self.sampled_snrs[i])
            mask = np.where(snrs > 0.1)
            snrs = snrs[mask]
            fluxes = fluxes[mask]
            f = interp1d(np.sort(snrs), np.sort(fluxes), kind='linear')
            minsnr = max(self.min_snr, min(snrs))
            maxsnr = min(self.max_snr, max(snrs))
            snrs_pred = np.linspace(minsnr, maxsnr, num=50)
            fluxes_pred = f(snrs_pred)
            flux_for_lowsnr = f(minsnr)
            flux_for_higsnr = f(maxsnr)
            fhi.append(flux_for_higsnr)
            flo.append(flux_for_lowsnr)

            # Figure of flux vs s/n
            axis.xaxis.set_tick_params(labelsize=6)
            axis.yaxis.set_tick_params(labelsize=6)
            axis.plot(fluxes, snrs, '.', alpha=0.2, markersize=4)
            axis.plot(fluxes_pred, snrs_pred, '-', alpha=1, color='orangered')
            axis.grid(which='major', alpha=0.3)
            axis.set_xlim(0)
            for l in plotvlines:
                axis.plot((0, max(fluxes)), (l, l), ':', color='darksalmon')
            axis = fig.add_subplot(111, frame_on=False)
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_xlabel('Fakecomp flux scaling', labelpad=25, size=8)
            axis.set_ylabel('Signal to noise ratio', labelpad=25, size=8)

        if isinstance(axs, np.ndarray):
            for i in range(len(self.distances), len(axs)):
                axs[i].axis('off')

        flo = np.array(flo).flatten()
        fhi = np.array(fhi).flatten()

        if self.inter_extrap and len(self.distances) > 2:
            x = self.distances
            f1 = interpolate.interp1d(x, flo, fill_value='extrapolate')
            f2 = interpolate.interp1d(x, fhi, fill_value='extrapolate')
            fhi = f2(self.inter_extrap_dist)
            flo = f1(self.inter_extrap_dist)
            plot_x = self.inter_extrap_dist
        else:
            plot_x = self.distances

        self.estimated_fluxes_high = fhi
        self.estimated_fluxes_low = flo

        plt.figure(figsize=(10, 4), dpi=dpi)
        plt.plot(self.distances, self.radprof, '--', alpha=0.8, color='gray',
                 lw=2, label='average radial profile')
        plt.plot(plot_x, flo, '.-', alpha=0.6, lw=2, color='dodgerblue',
                 label='flux lower bound')
        plt.plot(plot_x, fhi, '.-', alpha=0.6, color='dodgerblue', lw=2,
                 label='flux upper bound')
        plt.fill_between(plot_x, flo, fhi, where=flo <= fhi, alpha=0.2,
                         facecolor='dodgerblue', interpolate=True)
        plt.grid(which='major', alpha=0.4)
        plt.xlabel('Distance from the center [Pixels]')
        plt.ylabel('Fakecomp flux scaling [Counts]')
        plt.minorticks_on()
        plt.xlim(0)
        plt.ylim(0)
        plt.legend()
        plt.show()
        timing(starttime)


def _get_min_flux(i, distances, radprof, fwhm, plsc, min_snr, wavelengths=None,
                  mode='pca', scaling='temp-standard', random_seed=42):
    """
    """
    d = distances[i]
    fmin = radprof[i] * 0.1
    random_state = np.random.RandomState(random_seed)
    n_ks = 1
    theta_init = random_state.randint(0, 360)
    _, snr = _get_adi_snrs(GARRPSF, GARRPA, fwhm, plsc, (fmin, d, theta_init),
                           wavelengths, mode, n_ks, scaling)

    while snr > min_snr:
        theta = random_state.randint(0, 360)
        f, snr = _get_adi_snrs(GARRPSF, GARRPA, fwhm, plsc, (fmin, d, theta),
                               wavelengths, mode, n_ks, scaling)
        fmin *= 0.5

    return fmin


def _get_max_flux(i, distances, flux_min, fwhm, plsc, max_snr, wavelengths=None,
                  mode='pca', scaling='temp-standard', random_seed=42):
    """
    """
    d = distances[i]
    snr = 0.01
    flux = flux_min[i] * 2
    snrs = []
    counter = 1
    random_state = np.random.RandomState(random_seed)
    n_ks = 1

    while snr < max_snr:
        theta = random_state.randint(0, 360)
        f, snr = _get_adi_snrs(GARRPSF, GARRPA, fwhm, plsc, (flux, d, theta),
                               wavelengths, mode, n_ks, scaling)

        # checking that the snr does not decrease
        if counter > 3 and snr <= snrs[-1]:
            print('Breaking... could not reach the max_snr value')
            flux *= 2
            break

        snrs.append(snr)
        flux *= 2
        counter += 1

    return flux


def _sample_flux_snr(distances, fwhm, plsc, n_injections, flux_min, flux_max,
                     nproc=10, random_seed=42, wavelengths=None, mode='median',
                     scaling='temp-standard'):
    """
    Sensible flux intervals depend on a combination of factors, # of frames,
    range of rotation, correlation, glare intensity.
    """
    if GARRAY.ndim == 3:
        frsize = int(GARRAY.shape[1])
    elif GARRAY.ndim == 4:
        frsize = int(GARRAY.shape[2])
    ninj = n_injections
    random_state = np.random.RandomState(random_seed)
    flux_dist_theta_all = list()
    snrs_list = list()
    fluxes_list = list()
    n_ks = 3

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

    # multiprocessing (pool) for each distance
    res = pool_map(nproc, _get_adi_snrs, GARRPSF, GARRPA, fwhm, plsc,
                   fixed(flux_dist_theta_all), wavelengths, mode, n_ks, scaling)

    for i in range(len(distances)):
        flux_dist = []
        snr_dist = []
        for j in range(ninj):
            flux_dist.append(res[j + (ninj * i)][0])
            snr_dist.append(res[j + (ninj * i)][1])
        fluxes_list.append(flux_dist)
        snrs_list.append(snr_dist)

    return fluxes_list, snrs_list


def _get_adi_snrs(psf, angle_list, fwhm, plsc, flux_dist_theta_all,
                  wavelengths=None, mode='median', n_ks=3,
                  scaling='temp-standard'):
    """ Get the mean S/N (at 3 equidistant positions) for a given flux and
    distance, on a median subtracted frame.
    """
    theta = flux_dist_theta_all[2]
    flux = flux_dist_theta_all[0]
    dist = flux_dist_theta_all[1]

    if mode == 'median':
        snrs = []
        # 3 equidistant azimuthal positions
        for ang in [theta, theta + 120, theta + 240]:
            cube_fc, posx, posy = create_synt_cube(GARRAY, psf, angle_list,
                                                   plsc, flux=flux, dist=dist,
                                                   theta=ang, verbose=False)
            fr_temp = _compute_residual_frame(cube_fc, angle_list, dist, fwhm,
                                              wavelengths, mode, n_ks,
                                              'randsvd', scaling,
                                              collapse='median', imlib='opencv',
                                              interpolation='bilinear')
            res = frame_quick_report(fr_temp, fwhm, source_xy=(posx, posy),
                                     verbose=False)
            # mean S/N in circular aperture
            snrs.append(np.mean(res[-1]))

        # max of mean S/N at 3 equidistant positions
        snr = np.max(snrs)

    elif mode == 'pca':
        snrs = []
        # 3 equidistant azimuthal positions
        for ang in [theta, theta + 120, theta + 240]:
            cube_fc, posx, posy = create_synt_cube(GARRAY, psf, angle_list,
                                                   plsc, flux=flux, dist=dist,
                                                   theta=ang, verbose=False)
            fr_temp = _compute_residual_frame(cube_fc, angle_list, dist, fwhm,
                                              wavelengths, mode, n_ks,
                                              svd_mode='randsvd',
                                              scaling=scaling,
                                              collapse='median', imlib='opencv',
                                              interpolation='bilinear')
            snrs_ks = []
            for i in range(len(fr_temp)):
                res = frame_quick_report(fr_temp[i], fwhm/2, source_xy=(posx,
                                                                        posy),
                                         verbose=False)
                snrs_ks.append(np.median(res[-1]))

            maxsnr_ks = max(snrs_ks)
            if np.isinf(maxsnr_ks) or np.isnan(maxsnr_ks) or maxsnr_ks < 0:
                maxsnr_ks = 0.01

            snrs.append(maxsnr_ks)

            # DEBUG
            # print(flux, maxsnr_ks)
            # pp_subplots(np.array(fr_temp), axis=False, horsp=0.05,
            #             colorb=False)

        # max of mean S/N at 3 equidistant positions
        snr = np.max(snrs)

    return flux, snr


def _compute_residual_frame(cube, angle_list, radius, fwhm, wavelengths=None,
                            mode='pca', n_ks=3, svd_mode='randsvd',
                            scaling='temp-standard', collapse='median',
                            imlib='opencv', interpolation='bilinear'):
    """
    """
    annulus_width = 3 * fwhm

    if cube.ndim == 3:
        if mode == 'pca':
            angle_list = check_pa_vector(angle_list)
            data, pxind = prepare_matrix(cube, scaling, mode='annular',
                                         annulus_radius=radius, verbose=False,
                                         annulus_width=annulus_width)
            yy, xx = pxind
            max_pcs = min(data.shape[0], data.shape[1])
            U, S, V = svd_wrapper(data, svd_mode, max_pcs, False, False, True)
            exp_var = (S ** 2) / (S.shape[0] - 1)
            full_var = np.sum(exp_var)
            explained_variance_ratio = exp_var / full_var
            ratio_cumsum = np.cumsum(explained_variance_ratio)
            if n_ks == 1:
                ind = max(2, np.searchsorted(ratio_cumsum, 0.85))
                k_list = [ind]
            elif n_ks == 3:
                k_list = list()
                k_list.append(max(2, np.searchsorted(ratio_cumsum, 0.90)))
                k_list.append(np.searchsorted(ratio_cumsum, 0.95))
                k_list.append(np.searchsorted(ratio_cumsum, 0.99))

            res_frame = []
            for k in k_list:
                transformed = np.dot(V[:k], data.T)
                reconstructed = np.dot(transformed.T, V[:k])
                residuals = data - reconstructed
                cube_empty = np.zeros_like(cube)
                cube_empty[:, yy, xx] = residuals
                cube_res_der = cube_derotate(cube_empty, angle_list,
                                             imlib=imlib,
                                             interpolation=interpolation)
                res_frame.append(cube_collapse(cube_res_der, mode=collapse))

        elif mode == 'median':
            res_frame = median_sub(cube, angle_list, verbose=False)

    elif cube.ndim == 4:
        if mode == 'pca':
            z, n, y_in, x_in = cube.shape
            angle_list = check_pa_vector(angle_list)
            scale_list = check_scal_vector(wavelengths)
            big_cube = []

            # Rescaling the spectral channels to align the speckles
            for i in range(n):
                cube_resc = scwave(cube[:, i, :, :], scale_list)[0]
                cube_resc = cube_crop_frames(cube_resc, size=y_in,
                                             verbose=False)
                big_cube.append(cube_resc)

            big_cube = np.array(big_cube)
            big_cube = big_cube.reshape(z * n, y_in, x_in)

            data, pxind = prepare_matrix(big_cube, scaling, mode='annular',
                                         annulus_radius=radius, verbose=False,
                                         annulus_width=annulus_width)
            yy, xx = pxind
            max_pcs = min(data.shape[0], data.shape[1])
            U, S, V = svd_wrapper(data, svd_mode, max_pcs, False, False, True)
            exp_var = (S ** 2) / (S.shape[0] - 1)
            full_var = np.sum(exp_var)
            explained_variance_ratio = exp_var / full_var
            ratio_cumsum = np.cumsum(explained_variance_ratio)
            if n_ks == 1:
                ind = max(2, np.searchsorted(ratio_cumsum, 0.95))
                k_list = [ind]
            elif n_ks == 3:
                k_list = list()
                k_list.append(max(2, np.searchsorted(ratio_cumsum, 0.95)))
                k_list.append(np.searchsorted(ratio_cumsum, 0.97))
                k_list.append(np.searchsorted(ratio_cumsum, 0.99))

            res_frame = []
            for k in k_list:
                transformed = np.dot(V[:k], data.T)
                reconstructed = np.dot(transformed.T, V[:k])
                residuals = data - reconstructed
                res_cube = np.zeros_like(big_cube)
                res_cube[:, yy, xx] = residuals

                # Descaling the spectral channels
                resadi_cube = np.zeros((n, y_in, x_in))
                for i in range(n):
                    frame_i = scwave(res_cube[i * z:(i + 1) * z, :, :],
                                     scale_list, full_output=False,
                                     inverse=True, y_in=y_in, x_in=x_in,
                                     collapse=collapse)
                    resadi_cube[i] = frame_i

                cube_res_der = cube_derotate(resadi_cube, angle_list,
                                             imlib=imlib,
                                             interpolation=interpolation)
                res_frame.append(cube_collapse(cube_res_der, mode=collapse))

        elif mode == 'median':
            res_frame = median_sub(cube, angle_list, scale_list=wavelengths,
                                   verbose=False)

    return res_frame


def create_synt_cube(cube, psf, ang, plsc, dist=None, theta=None, flux=None,
                     random_seed=42, verbose=False):
    """
    """
    centy_fr, centx_fr = frame_center(cube[0])
    random_state = np.random.RandomState(random_seed)
    if theta is None:
        theta = random_state.randint(0,360)

    posy = dist * np.sin(np.deg2rad(theta)) + centy_fr
    posx = dist * np.cos(np.deg2rad(theta)) + centx_fr
    if verbose:
        print('Theta:', theta)
        print('Flux_inj:', flux)
    cubefc = cube_inject_companions(cube, psf, ang, flevel=flux, plsc=plsc,
                                    rad_dists=[dist], n_branches=1, theta=theta,
                                    verbose=verbose)
    return cubefc, posx, posy

