"""
Prediction procedures for MLAR, TMLAR and TMLAR4D samples.
"""
from __future__ import print_function
from __future__ import absolute_import

__all__ = ['predict_mlar']

import numpy as np
from vip_hci.preproc import (cube_derotate, cube_crop_frames, cube_derotate,
                             check_pa_vector)
from vip_hci.conf import time_ini, timing, time_fin, Progressbar
from vip_hci.var import (pp_subplots as plots,
                         frame_center, dist, cube_filter_highpass,
                         get_annulus_segments)
from vip_hci.conf.utils_conf import (pool_map, fixed, make_chunks)
from ..utils import normalize_01, create_feature_matrix, cube_move_subsample
from ..data_labeling import svd_decomp, get_cumexpvar


def predict_mlar(mode, model, cube, angle_list, fwhm, in_ann, out_ann,
                 patch_size, cevr_thresh, n_ks, kss_window=None,
                 tss_window=None, normalize='slice', lr_mode='eigen', n_proc=20,
                 verbose=False):
    """
    """
    global GARRAY
    GARRAY = cube

    starttime = time_ini(verbose=verbose)

    collapse_func = np.median
    scaling = None
    n_annuli = out_ann - in_ann
    patches = []
    coords = []
    frame_probas_one = np.zeros_like(cube[0])

    if verbose:
        print('N annuli: {}'.format(n_annuli))
        print('Grabbing MLAR/TMLAR/TMLAR4D samples per annulus')

    res = pool_map(n_proc, get_mlar_patches, fixed(range(in_ann, out_ann)),
                   fwhm, angle_list, patch_size, collapse_func, scaling,
                   lr_mode, cevr_thresh, n_ks, normalize, False, mode)
    for i in range(len(res)):
        patches.append(res[i][0])
        coords.append(res[i][1])

    patches = np.vstack(patches)
    coords = np.vstack(coords)

    # Moving-subsampling
    move_subsampling = 'mean'
    if mode == 'mlar' and kss_window is not None:
        patches = cube_move_subsample(patches, kss_window, axis=1,
                                      mode=move_subsampling)
    elif mode == 'tmlar' and tss_window is not None:
        patches = cube_move_subsample(patches, kss_window, axis=1,
                                      mode=move_subsampling)
    elif mode == 'tmlar4d':
        if tss_window is not None:
            patches = cube_move_subsample(patches, tss_window, axis=1,
                                          mode=move_subsampling)
        if kss_window is not None:
            patches = cube_move_subsample(patches, kss_window, axis=2,
                                          mode=move_subsampling)

    if verbose:
        timing(starttime)
        print('Generating a prediction on each MLAR/TMLAR/TMLAR4D sample')

    # Predicting on each MLAR sample using the model
    if hasattr(model, 'base_estimator'):  # Random forest
        # vectorizing the 3d samples to get a feature matrix
        patches = patches.reshape(patches.shape[0], -1)
        probas = model.predict_proba(patches)

    elif hasattr(model, 'get_weights'):  # Neural network
        flayer_name = model.layers[0].name

        if not flayer_name == 'conv2d_layer1' and \
           len(model.layers[0].input_shape) == 5:
            # adding extra dimension (channel) for TF/Keras model
            patches = np.expand_dims(patches, -1)
            if mode == 'tmlar4d':
                patches = list(np.moveaxis(patches, 2, 0))
            probas = model.predict(patches, verbose=verbose)
            if mode == 'tmlar4d':
                patches = np.moveaxis(np.array(patches), 0, 2)

        elif flayer_name == 'conv2d_layer1':
            ntotal = patches.shape[0]
            min_n_pairwfr = model.layers[0].input_shape[-1]
            newpatches = np.empty((ntotal, min_n_pairwfr, patches[0].shape[1],
                                   patches[0].shape[2]))
            for i in range(ntotal):
                ind = np.linspace(0, patches[i].shape[0], num=min_n_pairwfr,
                                  endpoint=False, dtype=int)
                newpatches[i] = patches[i][ind]
            patches = np.moveaxis(newpatches, 1, -1)
            probas = model.predict(patches, verbose=verbose)
    else:
        raise RuntimeError('Model not recognized')

    for j in range(coords.shape[0]):
        y = coords[j][0]
        x = coords[j][1]
        frame_probas_one[y, x] = probas[j]

    if verbose:
        timing(starttime)
    return frame_probas_one, probas, patches, coords


def get_mlar_patches(i, fwhm, angle_list, patch_size, collapse_func, scaling,
                     lr_mode, cevr_thresh, n_ks, normalize, verbose, mode):
    """
    """
    inrad = int(fwhm * i)
    outrad = int((fwhm * i) + fwhm)

    # enforcing the same number of k slices (match_klen=True)
    k_list = get_cumexpvar(GARRAY, 'annular', inrad, outrad, patch_size,
                           k_list=None, cevr_thresh=cevr_thresh, n_ks=n_ks,
                           match_len=True, verbose=False)

    frsize = int(GARRAY.shape[1])

    # Obtaining MLAR patches with original angles
    cube_residuals = svd_decomp(GARRAY, angle_list, patch_size, inrad, outrad,
                                scaling, k_list, collapse_func, lr_mode=lr_mode,
                                neg_ang=False, mode=mode)

    width = outrad - inrad
    yy, xx = get_annulus_segments((frsize, frsize), inrad, width, 1)[0]
    num_patches = yy.shape[0]
    if verbose:
        print("Ann {} to {} pxs: {} patches".format(inrad, outrad, num_patches))

    patches_frame = []
    coords = []
    for i in range(num_patches):
        patches_frame.append(cube_crop_frames(cube_residuals, patch_size,
                                              xy=(int(xx[i]), int(yy[i])),
                                              force=True, verbose=False))
        coords.append(np.array((int(yy[i]), int(xx[i]))))

    patches_frame = np.array(patches_frame)

    if normalize is not None:
        if mode == 'tmlar4d':
            for i in range(patches_frame.shape[0]):
                patches_frame[i] = normalize_01(patches_frame[i], normalize)
        else:
            patches_frame = normalize_01(patches_frame, normalize)

    return patches_frame, coords


def inspect_patch_multik(model, cube, angle_list, k_list, inrad=10, outrad=14,
                         size_patch=11, xy=(0, 0), scaling=None,
                         collapse_func=np.mean, normalize='slice', plot=True,
                         dpi=70, psf=None):
    """
    """
    if hasattr(model, 'base_estimator'):
        mode = 'rf'
    elif hasattr(model, 'name'):
        mode = 'nn'
    else:
        raise RuntimeError('Model not recognized')

    im_zeros = np.zeros_like(cube[0])
    im_zeros[xy[1], xy[0]] = 1

    cube_residuals = svd_decomp(cube, angle_list, size_patch, inrad, outrad,
                                scaling, k_list, collapse_func, neg_ang=False)

    y, x = np.where(im_zeros == 1)
    patch = cube_crop_frames(np.array(cube_residuals), size_patch,
                             xy=(int(x), int(y)), verbose=False)

    patch_reshaped = np.expand_dims(patch, 0)
    if normalize is not None:
        patch_reshaped = normalize_01(patch_reshaped, normalize)

    if mode == 'nn':
        # adding extra dimension (channel) for keras model
        patch_reshaped = np.expand_dims(patch_reshaped, -1)
        proba = model.predict(patch_reshaped, verbose=0)

    elif mode == 'rf':
        if psf is not None:
            patch_vector = create_feature_matrix(patch_reshaped, psf)
        else:
            # vectorizing the 3d samples to get a feature matrix
            patch_vector = patch_reshaped.flatten()
        proba = model.predict_proba(patch_vector)

    if plot:
        plots(np.squeeze(patch_reshaped), cmap='viridis', axis=False, dpi=dpi,
              maxplots=np.squeeze(patch_reshaped).shape[0], colorb=False)
    print('Proba :', proba, '\n')

    return patch, proba