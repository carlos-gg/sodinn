"""
Various helping functions. Plotting, saving/loading results, creating synt 
cubes, image processing.
"""

from __future__ import division
from __future__ import print_function

__all__ = ['close_hdf5_files']

import gc
import tables
import numpy as np
from skimage.draw import circle
from matplotlib.pyplot import (figure, show, subplot, title, hist)
import cv2
from vip_hci.metrics import noise_per_annulus, cube_inject_companions
from vip_hci.var import pp_subplots as plots, frame_center
from vip_hci.preproc import frame_crop, check_pa_vector, frame_shift


def close_hdf5_files():
    """ Closes Pytables HDF5 opened files.
    """
    for obj in gc.get_objects():  # Browse through ALL objects
        if isinstance(obj, tables.File):
            try:
                obj.close()
                print('Closing')
            except:
                print('Nothing to close')  # No opened HDF5 files


def cube_shift(cube, y, x, imlib, interpolation, border_mode='reflect'):
    """ Shifts the X-Y coordinates of a cube or 3D array by x and y values. """
    cube_out = np.zeros_like(cube)
    for i in range(cube.shape[0]):
        cube_out[i] = frame_shift(cube[i], y, x, imlib, interpolation,
                                  border_mode=border_mode)
    return cube_out


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


def normalize_01_pw(cube, mode):
    """
    mode : {'slice', 'sample'}, str
        "Slice" for per slice normalization, and "sample" for per sample
        normalization.
    """
    if mode == 'slice':
        patch_size = cube.shape[1]
        mat = cube.reshape(cube.shape[0], patch_size * patch_size)
        minvec = np.abs(np.min(mat, axis=1))
        mat += minvec[:, np.newaxis]
        maxvec = np.max(mat, axis=1)
        mat /= maxvec[:, np.newaxis]
        return mat.reshape(cube.shape[0], patch_size, patch_size)
    elif mode == 'sample':
        cube = cube.copy()
        minv = np.abs(np.min(cube))
        cube += minv
        maxv = np.max(cube)
        cube /= maxv
        return cube
    else:
        raise ValueError("`normalization` not recognized")


def normalize_01(array, mode='slice'):
    """
    """
    n1, n2, n3, n4 = array.shape
    array = array.copy()

    if mode == 'slice':
        array_reshaped = array.reshape(n1 * n2, n3 * n4)
    elif mode == 'sample':
        array_reshaped = array.reshape(n1, n2 * n3 * n4)
    else:
        raise RuntimeError('Normalization mode not recognized')

    minvec = np.abs(np.min(array_reshaped, axis=1))
    array_reshaped += minvec[:, np.newaxis]
    maxvec = np.max(array_reshaped, axis=1)
    array_reshaped /= maxvec[:, np.newaxis]
    return array_reshaped.reshape(n1,n2,n3,n4)


def plot_traindata(T, zeroind=None, oneind=None, full_info=False, 
                   plot_pair=True, dpi=100, indices=None, save_plot=False):
    """
    """
    xarr = T.x
    yarr = T.y
    if 'xnor' in T:
        xarrn = T.xnor
    
    if zeroind is None:
        zeroind = np.random.randint(0,xarr.shape[0]/2.)
    if oneind is None:
        oneind = np.random.randint(xarr.shape[0]/2.,xarr.shape[0])
    
    if full_info:
        msg1 = 'N samples : {} | Runtime : {}'
        print(msg1.format(T.nsamp, T.runtime))
        msg2 = 'FWHM : {} | PLSC : {} | K list : {}'
        print(msg2.format(T.fwhm, T.plsc, T.klist))
        msg3 = 'In Rad : {} | Out Rad : {} | Patch size : {}'
        print(msg3.format(T.inrad, T.outrad, T.sizepatch))
        msg4 = 'Collapse func : {} | Scaling : {}'
        print(msg4.format(T.collaf.__name__, T.scaling))
        msg5 = 'N patches : {} | Perc orig zeros : {}'
        print(msg5.format(T.npatches, T.perorigzeros))
        msg6 = 'Flux distro : {} | Par1 : {} | Par2 : {}'
        print(msg6.format(T.fluxdistro, T.fluxdistrop1, T.fluxdistrop2))
        msg7 = 'N injections : {} | Perc aug ones : {}'
        print(msg7.format(T.nsamp*0.5*T.peraugones, T.peraugones))
        msg8 = 'Aug shifts : {} | Aug range rotat : {}'
        print(msg8.format(T.shifts, T.rangerot))
        figure(figsize=(12,2))
        subplot(1, 3, 1)
        hist(T.fluxes, bins=int(np.sqrt(T.fluxes.shape[0])))
        title('Fluxes histogram')
        subplot(1, 3, 2)
        hist(np.array(T.dists).flatten(), 
                      bins=int(np.sqrt(T.fluxes.shape[0])))
        title('Distances histogram')
        subplot(1, 3, 3)
        hist(np.array(T.thetas).flatten(), 
                      bins=int(np.sqrt(T.fluxes.shape[0])))
        title('Thetas histogram')
        show()
        print()
    
    npatches = xarr[zeroind].shape[0]
    if plot_pair or save_plot:
        if indices is not None:
            zerarr = xarr[zeroind][indices]
            onearr = xarr[oneind][indices]
            if xarrn is not None: zerarrn = xarrn[zeroind][indices]
            if xarrn is not None: onearrn = xarrn[oneind][indices]
        else:
            zerarr = xarr[zeroind]
            onearr = xarr[oneind]
            if xarrn is not None: zerarrn = xarrn[zeroind]
            if xarrn is not None: onearrn = xarrn[oneind]

        if save_plot:
            print('{} | Sample {}'.format(int(yarr[zeroind]), zeroind))
            plots(zerarr, dpi=dpi, axis=False, vmin=xarr[zeroind].min(), 
                  vmax=xarr[zeroind].max(), save='patch_zero.pdf', colorb=False,
                  maxplots=npatches, horsp=0.1)
            if xarrn is not None:
                plots(zerarrn, axis=False, dpi=dpi, colorb=False,
                      save='patch_zero_nor.pdf', maxplots=npatches, horsp=0.1)
            print(int(yarr[oneind]),'| Sample', oneind) 
            plots(onearr, axis=False, vmin=xarr[oneind].min(), 
                  vmax=xarr[oneind].max(), dpi=dpi, save='patch_one.pdf', 
                  colorb=False, maxplots=npatches, horsp=0.1)
            if xarrn is not None:
                plots(onearr, axis=False, dpi=dpi, horsp=0.1,
                      save='patch_one_nor.pdf', colorb=False, maxplots=npatches)
        
        else:
            plots(zerarr, title='Unnormalized ZERO multiK patch', dpi=dpi,
                  axis=False, vmin=xarr[zeroind].min(), vmax=xarr[zeroind].max(),
                  maxplots=npatches, horsp=0.1)
            if xarrn is not None:
                plots(zerarrn, title='Normalized ZERO multiK patch', 
                      axis=False, dpi=dpi, maxplots=npatches, horsp=0.1)
            plots(onearr, title='Unnormalized ONE multiK patch', axis=False,
                  vmin=xarr[oneind].min(), vmax=xarr[oneind].max(), dpi=dpi,
                  maxplots=npatches, horsp=0.1)
            if xarrn is not None:
                plots(onearrn, title='Normalized ONE multiK patch', 
                      axis=False, dpi=dpi, maxplots=npatches, horsp=0.1)


def create_feature_matrix(X, psf):
    """
    http://docs.opencv.org/2.4/modules/imgproc/doc/object_detection.html?highlight=matchtemplate#matchtemplate
    """
    psf_corr = frame_crop(psf, 7, verbose=False)
    psf_corr = psf_corr + np.abs(np.min(psf_corr))
    psf_corr = psf_corr / np.abs(np.max(psf_corr))
    
    n_samples = X.shape[0]
    n_features = X.shape[1]
    
    Xfeatmat = np.zeros((n_samples, n_features))
    for i in range(n_samples):
        for j in range(n_features):
            Xfeatmat[i,j] = cv2.matchTemplate(X[i][j].astype('float32'),
                                              psf_corr.astype('float32'),
                                              cv2.TM_CCOEFF_NORMED)

    Xfeatmat_std = np.zeros((n_samples, n_features))
    for i in range(n_samples):
        for j in range(n_features):
            Xfeatmat_std[i,j] = np.std(X[i][j])
        
    Xfeatmat = np.concatenate((Xfeatmat, Xfeatmat_std), axis=1)
    return Xfeatmat


def get_indices_annulus(shape, inrad, outrad, mask=None, maskrad=None,
                        verbose=False):
    """ mask is a list of tuples X,Y
    # TODO: documentation
    """
    framemp = np.zeros(shape)
    if mask is not None:
        if not isinstance(mask, list):
            raise TypeError('Mask should be a list of tuples')
        if maskrad is None:
            raise ValueError('Fwhm not given')
        for xy in mask:
            # patch_size/2 diameter aperture
            cir = circle(xy[1], xy[0], maskrad, shape)
            framemp[cir] = 1

    annulus_width = outrad - inrad
    cy, cx = frame_center(framemp)
    yy, xx = np.mgrid[:framemp.shape[0], :framemp.shape[1]]
    circ = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    donut_mask = (circ <= (inrad + annulus_width)) & (circ >= inrad)
    y, x = np.where(donut_mask)
    if mask is not None:
        npix = y.shape[0]
        ymask, xmask = np.where(framemp)    # masked pixels where == 1
        inds = []
        for i, tup in enumerate(zip(y, x)):
            if tup in zip(ymask, xmask):
                inds.append(i)
        y = np.delete(y, inds)
        x = np.delete(x, inds)

    if verbose:
        print(y.shape[0], 'pixels in annulus')
    return y, x

