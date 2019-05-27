"""
Tests for DataLabeler using mlar sample type
"""

import copy
from data_labeling.labeling import DataLabeler
from pytest import fixture


@fixture(scope="module")
def test_dataLabeler_mlar(example_dataset_adi):
    """
    Parameters
    ----------
    example_dataset_adi : fixture
        Taken automatically from ``conftest.py``.
    """

    dataset = copy.copy(example_dataset_adi)

    radius_int = dataset.cube.shape[2]/2

    try:
        labeler = DataLabeler('mlar', dataset.cube, dataset.angles,
                              dataset.psf, radius_int=radius_int,
                              fwhm=dataset.fwhm, plsc=float(dataset.plsc[0]),
                              min_snr=3, max_snr=5, n_proc=2)
    except TypeError:
        raise

    try:
        labeler.estimate_fluxes()
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

    try:
        labeler.inspect_samples()
    except TypeError:
        raise

    return True
