"""

"""
from __future__ import print_function, division, absolute_import

__all__ = ['FluxEstimator']

import numpy as np
import hciplot as hp
from pandas import DataFrame
from matplotlib import pyplot as plt
from scipy import interpolate
from scipy.interpolate import interp1d
from vip_hci.conf import time_ini, timing
from vip_hci.stats import frame_average_radprofile
from vip_hci.conf.utils_conf import pool_map, iterable, check_array
from vip_hci.var import get_annulus_segments, frame_center
from vip_hci.metrics import cube_inject_companions
from vip_hci.preproc import (cube_derotate, cube_collapse, check_pa_vector,
                             check_scal_vector)
from vip_hci.preproc import cube_rescaling_wavelengths as scwave
from vip_hci.metrics import snr
from vip_hci.medsub import median_sub
from vip_hci.pca import SVDecomposer

import warnings
# To silence UserWarning when scaling data with sklearn
warnings.filterwarnings("ignore")


class FluxEstimator:
    """
    Fluxes (proxy of contrast) estimator for injecting fake companions.

    Parameters
    ----------
    cube : array_like, 3d or 4d
        Input sequence (ADI or IFS+ADI).
    psf : array_like, 2d or 3d
        Input corresponding template PSF.
    distances : tuple or list
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
    min_snr : int or tuple/list, optional
        Minimum target SNR for which a flux will be estimated at given
        distances.
    max_snr : int or tuple/list, optional
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
    def __init__(self, cube, psf, distances, angles, fwhm, plsc,
                 wavelengths=None, spectrum=None, n_injections=30, algo='pca',
                 min_snr=1, max_snr=3, inter_extrap=False,
                 inter_extrap_dist=None, random_seed=42, n_proc=2):
        """ Initialization of the flux estimator object.
        """
        global GARRAY
        global GARRPSF
        global GARRWL
        global GARRPA
        GARRAY = cube
        GARRPSF = psf
        GARRPA = angles
        GARRWL = wavelengths

        check_array(cube, dim=(3, 4), msg='cube')
        check_array(psf, dim=(2, 3), msg='psf')
        check_array(angles, dim=1, msg='angles')
        check_array(distances, dim=1, msg='distances')

        if isinstance(min_snr, (tuple, list)):
            if not len(min_snr) == len(distances):
                raise ValueError('`min_snr` length does not match `distances`')
        elif isinstance(min_snr, (int, float)):
            min_snr = [min_snr] * len(distances)
        else:
            raise TypeError('`min_snr` must be a float/int or a list/tuple')

        if isinstance(max_snr, (tuple, list)):
            if not len(max_snr) == len(distances):
                raise ValueError('`max_snr` length does not match `distances`')
        elif isinstance(max_snr, (int, float)):
            max_snr = [max_snr] * len(distances)
        else:
            raise TypeError('`max_snr` must be a float/int or a list/tuple')

        if cube.ndim == 4:
            if wavelengths is None:
                raise ValueError('`wavelengths` must be provided when `cube` '
                                 'is a 4d array')
            if spectrum is None:
                raise ValueError('`spectrum` must be provided when `cube` is a '
                                 '4d array')
            check_array(wavelengths, dim=1, msg='wavelengths')
            check_array(spectrum, dim=1, msg='spectrum')

            cy, cx = frame_center(cube)
            maxd = cy - 5 * fwhm
            if not max(distances) <= maxd:
                raise ValueError('`distances` contains a value that is too '
                                 'high wrt the frame size. Values must be '
                                 'smaller than {:.2f}'.format(maxd))

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
        self.spectrum = spectrum
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

    def get_min_flux(self):
        """ Obtaining the low end of the interval for sampling the S/N. Based
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

        print("Estimating the lower flux interval for sampling the S/N vs flux "
              "function")
        flux_min = pool_map(self.n_proc, _get_min_flux, iterable(self.n_dist),
                            self.distances, radprof, self.fwhm, self.plsc,
                            iterable(self.min_snr), self.wavelengths,
                            self.spectrum, self.algo, self.scaling,
                            self.random_seed)

        self.min_fluxes = flux_min
        timing(starttime)

    def get_max_flux(self, debug=False):
        """ Obtaining the high end of the interval for sampling the S/N.
        """
        if self.min_fluxes is None:
            self.get_min_flux()

        starttime = time_ini(verbose=False)

        print("Estimating the upper flux interval for sampling the S/N vs flux "
              "function")
        flux_max = pool_map(self.n_proc, _get_max_flux, iterable(self.n_dist),
                            self.distances, self.min_fluxes, self.fwhm,
                            self.plsc, iterable(self.max_snr), self.wavelengths,
                            self.spectrum, self.algo, self.scaling,
                            self.random_seed, debug)

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

        starttime = time_ini(verbose=False)
        print("Sampling by injecting fake companions")
        res = _sample_flux_snr(self.distances, self.fwhm, self.plsc,
                               self.n_injections, self.min_fluxes,
                               self.max_fluxes, self.n_proc, self.random_seed,
                               self.wavelengths, self.spectrum, self.algo,
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

        starttime = time_ini(verbose=False)
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

        print("Interpolating the Flux vs S/N function")
        # Regression for each distance
        for i, d in enumerate(self.distances):
            plotvlines = [self.min_snr[i], self.max_snr[i]]
            if isinstance(axs, np.ndarray):
                axis = axs[i]
            else:
                axis = axs

            fluxes = np.array(self.sampled_fluxes[i])
            snrs = np.array(self.sampled_snrs[i])
            mask = np.where(snrs > 0.1)
            snrs = snrs[mask]
            fluxes = fluxes[mask]
            f = interp1d(np.sort(snrs), np.sort(fluxes), kind='slinear',
                         fill_value='extrapolate')
            minsnr = max(self.min_snr[i], min(snrs))
            maxsnr = min(self.max_snr[i], max(snrs))
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

        # figure with fluxes as a function of the separation
        if len(self.distances) > 1:
            plt.figure(figsize=(10, 4), dpi=dpi)
            plt.plot(self.distances, self.radprof, '--', alpha=0.8,
                     color='gray', lw=2, label='average radial profile')
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
                  spectrum=None, mode='pca', scaling='temp-standard',
                  random_seed=42):
    """
    """
    d = distances[i]
    fmin = radprof[i] * 0.1
    random_state = np.random.RandomState(random_seed)
    n_ks = 3
    theta_init = random_state.randint(0, 360)
    _, snr = _get_adi_snrs(GARRPSF, GARRPA, fwhm, plsc, (fmin, d, theta_init),
                           wavelengths, spectrum, mode, n_ks, scaling)

    while snr > min_snr:
        theta = random_state.randint(0, 360)
        f, snr = _get_adi_snrs(GARRPSF, GARRPA, fwhm, plsc, (fmin, d, theta),
                               wavelengths, spectrum, mode, n_ks, scaling)
        fmin *= 0.5

    return fmin


def _get_max_flux(i, distances, flux_min, fwhm, plsc, max_snr, wavelengths=None,
                  spectrum=None, mode='pca', scaling='temp-standard',
                  random_seed=42, debug=False):
    """
    """
    d = distances[i]
    snr = 0.01
    flux = flux_min[i] * 2
    snr_list = []
    flux_list = []
    counter = 0
    counter_decrease = 0
    random_state = np.random.RandomState(random_seed)
    n_ks = 3

    while snr < max_snr:
        theta = random_state.randint(0, 360)
        _, snr = _get_adi_snrs(GARRPSF, GARRPA, fwhm, plsc, (flux, d, theta),
                               wavelengths, spectrum, mode, n_ks, scaling,
                               debug)

        # making sure the snr increases
        if counter > 3:
            if snr <= snr_list[-1]:
                counter_decrease += 1

            if counter_decrease > 5:
                print('Breaking... S/N keeps falling w/o reaching the max_snr')
                flux *= 10
                break

        snr_list.append(snr)
        flux_list.append(flux)
        flux *= 2
        counter += 1

    if debug:
        df = DataFrame({'Flux': flux_list, 'Max S/N': snr_list})
        print(df)

    return flux


def _sample_flux_snr(distances, fwhm, plsc, n_injections, flux_min, flux_max,
                     nproc=10, random_seed=42, wavelengths=None, spectrum=None,
                     mode='median', scaling='temp-standard'):
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
                   iterable(flux_dist_theta_all), wavelengths, spectrum, mode,
                   n_ks, scaling)

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
                  wavelengths=None, spectrum=None, mode='pca', n_ks=3,
                  scaling='temp-standard', debug=False):
    """ Get the mean S/N (at 3 equidistant positions) for a given flux and
    distance, on a residual frame.
    """
    theta = flux_dist_theta_all[2]
    flux = flux_dist_theta_all[0]
    dist = flux_dist_theta_all[1]

    # grey spectrum (same flux in all wls)
    if spectrum is None:
        spectrum = np.ones((GARRAY.shape[0]))

    snrs = []
    # 3 equidistant azimuthal positions, 1 or several K values
    for ang in [theta, theta + 120, theta + 240]:
        cube_fc, pos = cube_inject_companions(GARRAY, psf, angle_list,
                                              flevel=flux * spectrum, plsc=plsc,
                                              rad_dists=[dist], theta=ang,
                                              verbose=False, full_output=True)
        posy, posx = pos[0]
        fr_temp = _compute_residual_frame(cube_fc, angle_list, dist, fwhm,
                                          wavelengths, mode, n_ks, 'randsvd',
                                          scaling, 'median', 'opencv',
                                          'bilinear')
        # handling the case of mode='median'
        if isinstance(fr_temp, np.ndarray):
            fr_temp = [fr_temp]
        snrs_ks = []
        for i in range(len(fr_temp)):
            res = snr(fr_temp[i], source_xy=(posx, posy), fwhm=fwhm,
                      exclude_negative_lobes=True)
            snrs_ks.append(res)

        maxsnr_ks = max(snrs_ks)
        if np.isinf(maxsnr_ks) or np.isnan(maxsnr_ks) or maxsnr_ks < 0:
            maxsnr_ks = 0.01

        snrs.append(maxsnr_ks)

        if debug:
            print(' ')
            cy, cx = frame_center(GARRAY[0])
            label = 'Flux: {:.1f}, Max S/N: {:.2f}'.format(flux, maxsnr_ks)
            hp.plot_frames(tuple(np.array(fr_temp)), axis=False, horsp=0.05,
                           colorbar=False, circle=((posx, posy), (cx, cy)),
                           circle_radius=(5, dist), label=label, dpi=60)

    # max of mean S/N at 3 equidistant positions
    snr_value = np.max(snrs)

    return flux, snr_value


def _compute_residual_frame(cube, angle_list, radius, fwhm, wavelengths=None,
                            mode='pca', n_ks=3, svd_mode='randsvd',
                            scaling='temp-standard', collapse='median',
                            imlib='opencv', interpolation='bilinear',
                            debug=False):
    """
    """
    if cube.ndim == 3:
        annulus_width = 3 * fwhm
        inrad = radius - int(np.round(annulus_width / 2.))
        outrad = radius + int(np.round(annulus_width / 2.))

        if mode == 'pca':
            angle_list = check_pa_vector(angle_list)
            svdecomp = SVDecomposer(cube, mode='annular', inrad=inrad,
                                    outrad=outrad, svd_mode=svd_mode,
                                    scaling=scaling, wavelengths=None,
                                    verbose=False)
            _ = svdecomp.get_cevr(plot=False)

            if n_ks == 1:
                k_list = [svdecomp.cevr_to_ncomp(0.90)]
            elif n_ks == 3:
                k_list = list()
                k_list.append(svdecomp.cevr_to_ncomp(0.90))
                k_list.append(svdecomp.cevr_to_ncomp(0.95))
                k_list.append(svdecomp.cevr_to_ncomp(0.99))

            res_frame = []
            for k in k_list:
                transformed = np.dot(svdecomp.v[:k], svdecomp.matrix.T)
                reconstructed = np.dot(transformed.T, svdecomp.v[:k])
                residuals = svdecomp.matrix - reconstructed
                cube_empty = np.zeros_like(cube)
                cube_empty[:, svdecomp.yy, svdecomp.xx] = residuals
                cube_res_der = cube_derotate(cube_empty, angle_list,
                                             imlib=imlib,
                                             interpolation=interpolation)
                res_frame.append(cube_collapse(cube_res_der, mode=collapse))

        elif mode == 'median':
            res_frame = median_sub(cube, angle_list, verbose=False)

    elif cube.ndim == 4:
        inrad = max(1, radius - int(np.round(2 * fwhm)))
        outrad = min(int(cube.shape[-1] / 2.), radius + int(np.round(5 * fwhm)))

        if mode == 'pca':
            z, n, y_in, x_in = cube.shape
            angle_list = check_pa_vector(angle_list)
            scale_list = check_scal_vector(wavelengths)
            svdecomp = SVDecomposer(cube, mode='annular', inrad=inrad,
                                    outrad=outrad, svd_mode=svd_mode,
                                    scaling=scaling, wavelengths=scale_list,
                                    verbose=False)
            _ = svdecomp.get_cevr(plot=False)
            if n_ks == 1:
                k_list = [svdecomp.cevr_to_ncomp(0.90)]
            elif n_ks == 3:
                k_list = list()
                k_list.append(svdecomp.cevr_to_ncomp(0.90))
                k_list.append(svdecomp.cevr_to_ncomp(0.95))
                k_list.append(svdecomp.cevr_to_ncomp(0.99))

            if debug:
                print(k_list)

            res_frame = []
            for k in k_list:
                transformed = np.dot(svdecomp.v[:k], svdecomp.matrix.T)
                reconstructed = np.dot(transformed.T, svdecomp.v[:k])
                residuals = svdecomp.matrix - reconstructed
                res_cube = np.zeros(svdecomp.cube4dto3d_shape)
                res_cube[:, svdecomp.yy, svdecomp.xx] = residuals

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