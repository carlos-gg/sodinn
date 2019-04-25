"""
Prediction procedures.
"""
from __future__ import print_function
from __future__ import absolute_import

__all__ = ['predict_pairwise']

import numpy as np
from vip_hci.preproc import (cube_crop_frames, cube_derotate,
                             check_pa_vector)
from vip_hci.conf import time_ini, timing, Progressbar
from vip_hci.var import (pp_subplots as plots, frame_center, dist,
                         cube_filter_highpass, get_annulus_segments)
from ..utils import normalize_01_pw
from multiprocessing import cpu_count
from vip_hci.conf.utils_conf import (pool_imap, iterable, make_chunks)
from vip_hci.preproc import check_pa_vector
from ..data_labeling.labeling import _pairwise_diff_residuals


def prepare_patches(cube, angle_list, xy, fwhm, patch_size_px, delta_rot=0.5,
                    normalization='slice', imlib='opencv',
                    interpolation='bilinear', debug=False):
    """ Prepare patches for SODINN-PW.
    """
    centy_fr, centx_fr = frame_center(cube[0])

    angle_list = check_pa_vector(angle_list)

    xy_dist = dist(centy_fr, centx_fr, xy[1], xy[0])
    res = _pairwise_diff_residuals(cube, angle_list, ann_center=xy_dist,
                                   fwhm=fwhm, delta_rot=delta_rot, debug=False)

    res_der = cube_derotate(res, angle_list, imlib=imlib,
                            interpolation=interpolation)
    res_der_crop = cube_crop_frames(res_der, patch_size_px, xy=xy,
                                    verbose=False)

    patches = normalize_01_pw(res_der_crop, normalization)

    if debug:
        print('dist : {}'.format(xy_dist))
        plots(patches, axis=False, colorb=False, maxplots=patches.shape[0])
    return patches


def predict_from_patches(model, patches, ntotal):
    """
    """
    flayer_name = model.layers[0].name

    # for SODINN-PW3d
    if flayer_name == 'conv3d_layer1' or len(model.layers[0].input_shape) == 5:
        min_n_pairwfr = model.layers[0].input_shape[1]
        newpatches = np.empty((ntotal, min_n_pairwfr, patches[0].shape[1],
                              patches[0].shape[2]))
        for i in range(ntotal):
            ind = np.linspace(0, patches[i].shape[0], num=min_n_pairwfr,
                              endpoint=False, dtype=int)
            newpatches[i] = patches[i][ind]
        newpatches = np.expand_dims(newpatches, -1)
        probas = model.predict(newpatches, verbose=1)

    # for SODINN-PW2d and SODINN-PW2d_pseudo3d
    elif flayer_name == 'conv2d_layer1':
        if model.layers[0].input_shape[-1] > 1:
            min_n_pairwfr = model.layers[0].input_shape[-1]
            newpatches = np.empty((ntotal, min_n_pairwfr, patches[0].shape[1],
                                   patches[0].shape[2]))
            for i in range(ntotal):
                ind = np.linspace(0, patches[i].shape[0], num=min_n_pairwfr,
                                  endpoint=False, dtype=int)
                newpatches[i] = patches[i][ind]
            newpatches = np.moveaxis(newpatches, 1, -1)
            probas = model.predict(newpatches, verbose=1)

        elif model.layers[0].input_shape[-1] == 1:
            probas = []
            for pat in patches:
                newpat = np.expand_dims(pat, -1)
                probas.append(np.median(model.predict(newpat, verbose=0)))

    return probas


def predict_pairwise(model, cube, angle_list, fwhm, patch_size_px, delta_rot,
                     radius_int=None, high_pass='laplacian', kernel_size=5,
                     normalization='slice', imlib='opencv',
                     interpolation='bilinear', nproc=1, verbose=True,
                     chunks_per_proc=2):
    """
    Parameters
    ----------
    model : Keras model
    cube : 3d ndarray
    angle_list : 1d ndarray
    fwhm : float
    patch_size : int, optional
    delta_rot : float, optional
    verbose: bool or int, optional
        0 / False: no output
        1 / True: full output (timing + progress bar)
        2: progress bar only
    [...]


    Returns
    -------
    probmap : 2d ndarray

    Notes
    -----
    - support for 4D cubes?
    """
    starttime = time_ini(verbose=verbose)
    if radius_int is None:
        radius_int = fwhm
    n_frames, sizey, sizex = cube.shape

    width = int(sizey / 2 - patch_size_px / 2 - radius_int)
    ind = get_annulus_segments(cube[0], inner_radius=radius_int, width=width,
                               nsegm=1)
    probmap = np.zeros((sizey, sizex))

    if high_pass is not None:
        cube = cube_filter_highpass(cube, high_pass, kernel_size=kernel_size,
                                    verbose=False)

    indices = list(range(ind[0][0].shape[0]))

    if nproc is None:
        nproc = cpu_count() // 2  # Hyper-threading doubles the # of cores

    # prepare patches in parallel
    nchunks = nproc * chunks_per_proc
    print("Grabbing patches with {} processes".format(nproc))
    res_ = list(Progressbar(pool_imap(nproc, _parallel_make_patches_chunk,
                                      iterable(make_chunks(indices, nchunks)),
                                      ind, cube, angle_list, fwhm,
                                      patch_size_px, delta_rot, normalization,
                                      imlib, interpolation),
                            total=nchunks, leave=False, verbose=False))
    xx = []
    yy = []
    pats = []
    for r in res_:
        x, y, pp = r
        for i in range(len(x)):
            xx.append(x[i])
            yy.append(y[i])
            pats.append(pp[i])

    if verbose == 1:
        timing(starttime)
        print("Prediction on patches:")

    probas = predict_from_patches(model, pats, len(xx))
    for i in range(len(xx)):
        probmap[xx[i], yy[i]] = probas[i]

    if verbose == 1:
        timing(starttime)
    return probmap


def _parallel_make_patches_chunk(i_chunk, ind, cube, angle_list, fwhm,
                                 patch_size_px, delta_rot, normalization, imlib,
                                 interpolation):
    """worker function for Pool.map"""
    x_all = []
    y_all = []
    patches_all = []

    for i in i_chunk:
        xy = (int(ind[0][1][i]), int(ind[0][0][i]))  # TODO: yx?
        patches = prepare_patches(cube, angle_list, xy, fwhm, patch_size_px,
                                  delta_rot, normalization, imlib,
                                  interpolation, False)
        x = xy[1]
        y = xy[0]
        x_all.append(x)
        y_all.append(y)
        patches_all.append(patches)

    return x_all, y_all, patches_all

