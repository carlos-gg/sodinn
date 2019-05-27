"""
Generation of labeled data for supervised learning. To be used to train the
discriminative models. 
"""
from __future__ import print_function, division, absolute_import

import warnings
from multiprocessing import cpu_count
from multiprocessing import get_start_method

import cupy
# import torch
import numpy as np
from vip_hci.conf.utils_conf import (pool_map, iterable)
from vip_hci.pca.svd import svd_wrapper
from vip_hci.preproc import cube_crop_frames, cube_derotate
from vip_hci.var import (get_annulus_segments, prepare_matrix,
                         frame_center)

from ..utils import (normalize_01, create_synt_cube, cube_move_subsample)

warnings.filterwarnings(action='ignore', category=UserWarning)


def make_mlar_samples_ann_signal(input_array, angle_list, psf, n_samples,
                                 cevr_thresh, n_ks, force_klen, inrad, outrad,
                                 patch_size, flux_low, flux_high, plsc=0.01,
                                 normalize='slice', nproc=1, nproc2=1,
                                 interp='bilinear', lr_mode='eigen',
                                 mode='mlar', kss_window=None, tss_window=None,
                                 random_seed=42, verbose=False):
    """
    n_samples : For ONEs, half_n_samples SVDs

    mask is a list of tuples X,Y

    inputarr is a 3d array or list of 3d arrays

    orig_zero_patches : percentage of original zero patches

    """
    dist_flux_p1 = flux_low
    dist_flux_p2 = flux_high
    collapse_func = np.mean
    scaling = None  # 'temp-mean'
    random_state = np.random.RandomState(random_seed)

    # making ones, injecting companions. The other half of n_samples
    if verbose:
        print("Creating the ONEs samples")

    frsize = int(input_array.shape[1])
    if frsize > outrad + outrad + patch_size + 2:
        frsize = int(outrad + outrad + patch_size + 2)
        cube = cube_crop_frames(input_array, frsize, force=True, verbose=False)

    width = outrad - inrad
    yy, xx = get_annulus_segments((frsize, frsize), inrad, width, 1)[0]
    num_patches = yy.shape[0]

    k_list = get_cumexpvar(cube, 'annular', inrad, outrad, patch_size,
                           k_list=None, cevr_thresh=cevr_thresh, n_ks=n_ks,
                           match_len=force_klen, verbose=False)

    n_req_inject = n_samples
    if mode == 'mlar':
        # 4D: n_samples/2, n_k_list, patch_size, patch_size
        X_ones_array = np.empty((n_req_inject, len(k_list), patch_size,
                                patch_size))
    elif mode == 'tmlar':
        nfr = cube.shape[0]
        X_ones_array = np.empty((n_req_inject, nfr, patch_size, patch_size))

    elif mode == 'tmlar4d':
        nfr = cube.shape[0]
        X_ones_array = np.empty((n_req_inject, nfr, len(k_list), patch_size,
                                 patch_size))

    if verbose:
        print("{} injections".format(n_req_inject))

    if not dist_flux_p2 > dist_flux_p1:
        err_msg = 'dist_flux_p2 must be larger than dist_flux_p1'
        raise ValueError(err_msg)
    fluxes = random_state.uniform(dist_flux_p1, dist_flux_p2,
                                  size=n_req_inject)
    fluxes = np.sort(fluxes)
    inds_inj = random_state.randint(0, num_patches, size=n_req_inject)

    dists = []
    thetas = []
    for m in range(n_req_inject):
        injx = xx[inds_inj[m]]
        injy = yy[inds_inj[m]]
        injx -= frame_center(cube[0])[1]
        injy -= frame_center(cube[0])[0]
        dist = np.sqrt(injx**2+injy**2)
        theta = np.mod(np.arctan2(injy, injx) / np.pi * 180, 360)
        dists.append(dist)
        thetas.append(theta)

    if not nproc:
        nproc = int((cpu_count()/4))

    if nproc == 1:
        for m in range(n_req_inject):
            cufc, cox, coy = create_synt_cube(cube, psf, angle_list,
                                              plsc, theta=thetas[m],
                                              flux=fluxes[m], dist=dists[m],
                                              verbose=False)
            cox = int(np.round(cox))
            coy = int(np.round(coy))

            cube_residuals = svd_decomp(cufc, angle_list, patch_size,
                                        inrad, outrad, scaling, k_list,
                                        collapse_func, neg_ang=False,
                                        lr_mode=lr_mode, nproc=nproc2,
                                        interp=interp, mode=mode)

            # one patch residuals per injection
            X_ones_array[m] = cube_crop_frames(np.asarray(cube_residuals),
                                               patch_size, xy=(cox, coy),
                                               verbose=False)

    elif nproc > 1:
        if lr_mode in ['cupy', 'randcupy', 'eigencupy']:
            raise RuntimeError('CUPY does not play well with multiproc')

        if get_start_method() == 'fork' and lr_mode in ['pytorch',
                                                        'eigenpytorch',
                                                        'randpytorch']:
            raise RuntimeError("Cannot use pytorch and multiprocessing "
                               "outside main (i.e. from a jupyter cell). "
                               "See: http://pytorch.org/docs/0.3.1/notes/"
                               "multiprocessing.html.")

        flux_dist_theta = zip(fluxes, dists, thetas)

        res = pool_map(nproc, _inject_FC, cube, psf, angle_list, plsc,
                       inrad, outrad, iterable(flux_dist_theta), k_list,
                       scaling, collapse_func, patch_size, lr_mode, interp,
                       mode)

        for m in range(n_req_inject):
            X_ones_array[m] = res[m]

    # Moving-subsampling
    move_subsampling = 'median'
    if mode == 'mlar' and kss_window is not None:
        X_ones_array = cube_move_subsample(X_ones_array, kss_window, axis=1,
                                           mode=move_subsampling)
    elif mode == 'tmlar' and tss_window is not None:
        X_ones_array = cube_move_subsample(X_ones_array, kss_window, axis=1,
                                           mode=move_subsampling)
    elif mode == 'tmlar4d':
        if tss_window is not None:
            X_ones_array = cube_move_subsample(X_ones_array, tss_window,
                                               axis=1, mode=move_subsampling)
        if kss_window is not None:
            X_ones_array = cube_move_subsample(X_ones_array, kss_window,
                                               axis=2, mode=move_subsampling)

    if normalize is not None:
        if mode == 'tmlar4d':
            for i in range(X_ones_array.shape[0]):
                X_ones_array[i] = normalize_01(X_ones_array[i], normalize)
        else:
            X_ones_array = normalize_01(X_ones_array, normalize)

    return X_ones_array.astype('float32')


def make_mlar_samples_ann_noise(input_array, angle_list, cevr_thresh, n_ks,
                                force_klen, inrad, outrad, patch_size, fwhm=4,
                                normalize='slice', nproc2=1, interp='bilinear',
                                lr_mode='eigen', mode='mlar', kss_window=None,
                                tss_window=None, random_seed=42, nsamp_sep=None,
                                verbose=False):
    """
    patch_size in pxs

    mask is a list of tuples X,Y

    inputarr is a 3d array or list of 3d arrays

    orig_zero_patches : percentage of original zero patches

    """
    collapse_func = np.mean
    scaling = None  # 'temp-mean'
    random_state = np.random.RandomState(random_seed)
    patches_list = []
    frsize = int(input_array.shape[1])

    if frsize > outrad + outrad + patch_size + 2:
        frsize = int(outrad + outrad + patch_size + 2)
        cube = cube_crop_frames(input_array, frsize, force=True, verbose=False)

    # making zeros
    if verbose:
        print("Creating the ZEROs samples")

    if not inrad >= int(patch_size/2.) + fwhm:
        msg = "Warning: The patches are overlapping with the inner 1xFWHM "
        msg += "annulus"
    if not inrad > int(np.round(patch_size/2.)):
        raise RuntimeError("Inner radius must be > half patch_size")

    k_list = get_cumexpvar(cube, 'annular', inrad, outrad, patch_size,
                           k_list=None, cevr_thresh=cevr_thresh, n_ks=n_ks,
                           match_len=force_klen, verbose=False)

    resdec = svd_decomp(cube, angle_list, patch_size, inrad, outrad,
                        scaling, k_list, collapse_func, interp=interp,
                        nproc=nproc2, lr_mode=lr_mode, neg_ang=True, mode=mode)
    cube_residuals_negang = resdec

    width = outrad - inrad
    yy, xx = get_annulus_segments((frsize, frsize), inrad, width, 1)[0]
    if nsamp_sep is None:
        num_patches = yy.shape[0]
    else:
        num_patches = nsamp_sep
        if num_patches < yy.shape[0]:
            ind = random_state.choice(yy.shape[0], nsamp_sep, replace=False)
        else:
            ind = random_state.choice(yy.shape[0], nsamp_sep, replace=True)
        yy = yy[ind]
        xx = xx[ind]

    cube_residuals_negang = np.asarray(cube_residuals_negang)
    if verbose:
        print("Total patches in annulus = {:}".format(num_patches))

    for i in range(num_patches):
        xy = (int(xx[i]), int(yy[i]))
        patches_list.append(cube_crop_frames(cube_residuals_negang,
                                             patch_size, xy=xy, verbose=False,
                                             force=True))

    # For MLAR and TMLAR X_zeros_array is 4d:
    # [n_patches_annulus, n_k_list || n_time_steps, patch_size, patch_size]
    # For TMLAR4D X_zeros_array is 5d:
    # [n_patches_annulus, n_time_steps, n_k_list, patch_size, patch_size]
    X_zeros_array = np.array(patches_list)

    n_ks = len(k_list)

    # Moving-subsampling
    move_subsampling = 'median'
    if mode == 'mlar' and kss_window is not None:
        X_zeros_array = cube_move_subsample(X_zeros_array, kss_window, axis=1,
                                            mode=move_subsampling)
    elif mode == 'tmlar' and tss_window is not None:
        X_zeros_array = cube_move_subsample(X_zeros_array, kss_window, axis=1,
                                            mode=move_subsampling)
    elif mode == 'tmlar4d':
        if tss_window is not None:
            X_zeros_array = cube_move_subsample(X_zeros_array, tss_window,
                                                axis=1, mode=move_subsampling)
        if kss_window is not None:
            X_zeros_array = cube_move_subsample(X_zeros_array, kss_window,
                                                axis=2, mode=move_subsampling)

    # Normalization
    if normalize is not None:
        if mode == 'tmlar4d':
            for i in range(X_zeros_array.shape[0]):
                X_zeros_array[i] = normalize_01(X_zeros_array[i], normalize)
        else:
            X_zeros_array = normalize_01(X_zeros_array, normalize)
    return X_zeros_array.astype('float32'), k_list, n_ks


def _inject_FC(cube, psf, angle_list, plsc, inrad, outrad, flux_dist_theta,
               k_list, sca, collapse_func, patch_size, lr_mode, interp, mode):
    """ One patch residuals per injection
    """
    cubefc, cox, coy = create_synt_cube(cube, psf, angle_list, plsc,
                                        flux=flux_dist_theta[0],
                                        dist=flux_dist_theta[1],
                                        theta=flux_dist_theta[2], verbose=False)
    cox = int(np.round(cox))
    coy = int(np.round(coy))

    cube_residuals = svd_decomp(cubefc, angle_list, patch_size, inrad, outrad,
                                sca, k_list, collapse_func, neg_ang=False,
                                lr_mode=lr_mode, nproc=1, interp=interp,
                                mode=mode)
    patch = cube_crop_frames(np.array(cube_residuals), patch_size,
                             xy=(cox, coy), verbose=False, force=True)
    return patch


def svd_decomp(array, angle_list, size_patch, inrad, outrad, sca, k_list,
               collapse_func, neg_ang=True, lr_mode='eigen', nproc=1,
               interp='bilinear', mode='mlar'):
    """
    """
    cube = array
    nfr, frsize, _ = cube.shape
    half_sizep = np.ceil(size_patch / 2.)
    inradius = inrad - half_sizep - 1
    outradius = outrad + half_sizep + 1

    matrix, ann_ind = prepare_matrix(cube, scaling=sca, mask_center_px=None,
                                     mode='annular', inner_radius=inradius,
                                     outer_radius=outradius, verbose=False)

    V = svd_wrapper(matrix, lr_mode, k_list[-1], False, False, to_numpy=False)
    cube_residuals = []

    if neg_ang:
        fac = -1
    else:
        fac = 1

    for k in k_list:
        if lr_mode in ['cupy', 'randcupy', 'eigencupy']:
            matrix = cupy.array(matrix)
            transformed = cupy.dot(V[:k], matrix.T)
            reconstructed = cupy.dot(transformed.T, V[:k])
            residuals_ann = matrix - reconstructed
            residuals_ann = cupy.asnumpy(residuals_ann)
        # elif lr_mode in ['pytorch', 'randpytorch', 'eigenpytorch']:
        #     matrix = matrix.astype('float32')
        #     matrix_gpu = torch.Tensor.cuda(torch.from_numpy(matrix))
        #     transformed = torch.mm(V[:k], torch.transpose(matrix_gpu, 0, 1))
        #     reconstructed = torch.mm(torch.transpose(transformed, 0, 1), V[:k])
        #     residuals_ann = matrix_gpu - reconstructed
        else:
            transformed = np.dot(V[:k], matrix.T)
            reconstructed = np.dot(transformed.T, V[:k])
            residuals_ann = matrix - reconstructed

        # TODO: fix bottleneck when nframes grows. Derot. in parallel batches
        residual_frames = np.zeros((nfr, frsize, frsize))
        residual_frames[:, ann_ind[0], ann_ind[1]] = residuals_ann
        residual_frames_rot = cube_derotate(residual_frames, fac * angle_list,
                                            nproc=nproc, interpolation=interp)
        if mode == 'mlar':
            cube_residuals.append(collapse_func(residual_frames_rot, axis=0))
        elif mode == 'tmlar':
            cube_residuals.append(residual_frames_rot)
        elif mode == 'tmlar4d':
            cube_residuals.append(residual_frames_rot)

    cube_residuals = np.array(cube_residuals)
    if mode == 'tmlar':
        cube_residuals = np.mean(cube_residuals, axis=0)
    elif mode == 'tmlar4d':
        cube_residuals = np.moveaxis(cube_residuals, 1, 0)

    return cube_residuals


def get_cumexpvar(cube, expvar_mode, inrad, outrad, size_patch, k_list=None,
                  cevr_thresh=0.99, n_ks=20, match_len=False, verbose=True):
    """
    """
    n_frames = cube.shape[0]
    half_sizep = np.ceil(size_patch / 2.)
    inradius = inrad - half_sizep - 1
    outradius = outrad + half_sizep + 1

    matrix_svd = prepare_matrix(cube, scaling='temp-standard',
                                mask_center_px=None, mode=expvar_mode,
                                inner_radius=inradius, outer_radius=outradius,
                                verbose=False)
    if expvar_mode == 'annular':
        matrix_svd = matrix_svd[0]

    U, S, V = svd_wrapper(matrix=matrix_svd, mode='lapack',
                          ncomp=min(matrix_svd.shape[0], matrix_svd.shape[1]),
                          verbose=False, full_output=True)

    exp_var = (S ** 2) / (S.shape[0] - 1)
    full_var = np.sum(exp_var)
    explained_variance_ratio = exp_var / full_var  # % variance expl. by each PC
    ratio_cumsum = np.cumsum(explained_variance_ratio)

    if k_list is not None:
        ratio_cumsum_klist = []
        for k in k_list:
            ratio_cumsum_klist.append(ratio_cumsum[k - 1])

        if verbose:
            print("SVD on input matrix (annulus from cube)")
            print("  Number of PCs :\t")
            print("  ", k_list)
            print("  Cum. explained variance ratios :\t")
            list_vals = ["{0:0.2f}".format(i) for i in ratio_cumsum_klist]
            print("  ", str(list_vals).replace("'", ""), "\n")
        return ratio_cumsum, ratio_cumsum_klist
    else:
        if cevr_thresh is not None and n_ks is not None:
            ind = np.searchsorted(ratio_cumsum, cevr_thresh)
            if verbose:
                ratio_cumsum_klist = ratio_cumsum[:ind + 1]
                print(ratio_cumsum_klist)
            k_list = list(range(1, n_frames + 1))[:ind + 1]
            # print('*', ind, k_list)

            if match_len:
                while len(k_list) < n_ks:
                    k_list.append(k_list[-1] + 1)
            # print('**', ind, k_list)

            if n_ks < len(k_list):
                ind_for_nks = np.linspace(0, ind, n_ks, dtype=int).tolist()
                k_list = np.array(k_list)[ind_for_nks]
            # print('***', ind, k_list)
            return k_list
        else:
            ratio_cumsum_klist = ratio_cumsum
            return ratio_cumsum, ratio_cumsum_klist




