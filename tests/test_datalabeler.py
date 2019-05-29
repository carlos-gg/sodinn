"""
Tests for DataLabeler using different sample type
"""

import copy
from ..data_labeling.labeling import DataLabeler
from vip_hci.preproc import frame_crop
from numpy import corrcoef


def test_dataLabeler_mlar(example_dataset_adi):
    """
    Parameters
    ----------
    example_dataset_adi : fixture
        Taken automatically from ``conftest.py``.
    """

    dataset = copy.copy(example_dataset_adi)

    if dataset.cube.shape[0] > 80:
        dataset.cube = dataset.cube[0:80]
        dataset.angles = dataset.angles[0:80]

    psf_croped = frame_crop(dataset.psf, int(round(dataset.fwhm))*2+1,
                            force=True, verbose=False)

    try:
        labeler_mlar = dataLabeler_type_test(dataset, "mlar")
    except TypeError:
        raise

    return True


def test_dataLabeler_tmlar(example_dataset_adi):
    """
    Parameters
    ----------
    example_dataset_adi : fixture
        Taken automatically from ``conftest.py``.
    """

    dataset = copy.copy(example_dataset_adi)

    if dataset.cube.shape[0] > 80:
        dataset.cube = dataset.cube[0:80]
        dataset.angles = dataset.angles[0:80]

    psf_croped = frame_crop(dataset.psf, int(round(dataset.fwhm)) * 2 + 1,
                            force=True, verbose=False)

    try:
        labeler_mlar = dataLabeler_type_test(dataset, "tmlar")
    except TypeError:
        raise

    return True


def test_dataLabeler_tmlar4d(example_dataset_adi):
    """
    Parameters
    ----------
    example_dataset_adi : fixture
        Taken automatically from ``conftest.py``.
    """

    dataset = copy.copy(example_dataset_adi)

    if dataset.cube.shape[0] > 80:
        dataset.cube = dataset.cube[0:80]
        dataset.angles = dataset.angles[0:80]

    psf_croped = frame_crop(dataset.psf, int(round(dataset.fwhm)) * 2 + 1,
                            force=True, verbose=False)

    try:
        labeler_mlar = dataLabeler_type_test(dataset, "tmlar4d")
    except TypeError:
        raise

    return True


def test_dataLabeler_pw3d(example_dataset_adi):
    """
    Parameters
    ----------
    example_dataset_adi : fixture
        Taken automatically from ``conftest.py``.
    """

    dataset = copy.copy(example_dataset_adi)

    if dataset.cube.shape[0] > 80:
        dataset.cube = dataset.cube[0:80]
        dataset.angles = dataset.angles[0:80]

    psf_croped = frame_crop(dataset.psf, int(round(dataset.fwhm)) * 2 + 1,
                            force=True, verbose=False)

    try:
        labeler_mlar = dataLabeler_type_test(dataset, "pw3d")
    except TypeError:
        raise

    return True


def test_dataLabeler_pw2d(example_dataset_adi):
    """
    Parameters
    ----------
    example_dataset_adi : fixture
        Taken automatically from ``conftest.py``.
    """

    dataset = copy.copy(example_dataset_adi)

    if dataset.cube.shape[0] > 80:
        dataset.cube = dataset.cube[0:80]
        dataset.angles = dataset.angles[0:80]

    psf_croped = frame_crop(dataset.psf, int(round(dataset.fwhm)) * 2 + 1,
                            force=True, verbose=False)

    try:
        labeler_mlar = dataLabeler_type_test(dataset, "pw2d")
    except TypeError:
        raise

    return True


def dataLabeler_type_test(dataset, sample_type):

    radius_int = round(dataset.fwhm * 2)

    try:
        labeler = DataLabeler(sample_type, dataset.cube, dataset.angles,
                              dataset.psf, radius_int=radius_int,
                              fwhm=dataset.fwhm, plsc=0.02719,
                              min_snr=3, max_snr=5, n_proc=2)
    except TypeError:
        raise

    try:
        labeler.estimate_fluxes(plot=False)
    except TypeError:
        raise

    try:
        labeler.run()
    except TypeError:
        raise

    try:
        labeler.augment()
    except TypeError:
        raise

    return labeler
