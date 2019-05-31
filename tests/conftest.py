"""
Configuration file for pytest, containing global ("session-level") fixtures.

"""

import copy
from pytest import fixture
from astropy.utils.data import download_file
import vip_hci as vip
from sodinn import DataLabeler, Model


@fixture(scope="session")
def example_dataset_adi():
    """
    Download example FITS cube from github + prepare HCIDataset object.

    Returns
    -------
    dataset : HCIDataset

    Notes
    -----
    Astropy's ``download_file`` uses caching, so the file is downloaded at most
    once per test run.

    """
    print("downloading data...")

    url_prefix = "https://github.com/carlgogo/VIP_extras/raw/master/datasets"

    f1 = download_file("{}/naco_betapic_cube.fits".format(url_prefix),
                       cache=True)
    f2 = download_file("{}/naco_betapic_psf.fits".format(url_prefix),
                       cache=True)
    f3 = download_file("{}/naco_betapic_pa.fits".format(url_prefix),
                       cache=True)

    # load fits
    cube = vip.fits.open_fits(f1)
    angles = vip.fits.open_fits(f3).flatten()  # shape (61,1) -> (61,)
    psf = vip.fits.open_fits(f2)

    # create dataset object
    dataset = vip.Dataset(cube, angles=angles, psf=psf,
                          px_scale=vip.conf.VLT_NACO['plsc'])

    dataset.normalize_psf(size=20, force_odd=False)

    # overwrite PSF for easy access
    dataset.psf = dataset.psfn

    return dataset


@fixture(scope="session")
def example_dataset_adi_fc(example_dataset_adi):
    data = copy.copy(example_dataset_adi)

    cube_fc = vip.metrics.cube_inject_companions(data.cube, data.psf,
                                                 data.angles, flevel=1000,
                                                 plsc=data.px_scale,
                                                 rad_dists=data.fwhm*4)

    dataset_fc = vip.Dataset(cube_fc, angles=data.angles, psf=data.psf,
                             px_scale=data.px_scale)

    return dataset_fc


@fixture(scope="session")
def dataLabeler_mlar(example_dataset_adi):
    """
    Parameters
    ----------
    example_dataset_adi : fixture
        Taken automatically from ``conftest.py``.
    """

    dataset = copy.copy(example_dataset_adi)

    labeler_mlar = dataLabeler_type_test(dataset, "mlar")

    return labeler_mlar


@fixture(scope="session")
def dataLabeler_tmlar(example_dataset_adi):
    """
    Parameters
    ----------
    example_dataset_adi : fixture
        Taken automatically from ``conftest.py``.
    """

    dataset = copy.copy(example_dataset_adi)

    labeler_tmlar = dataLabeler_type_test(dataset, "tmlar")

    return labeler_tmlar


@fixture(scope="session")
def dataLabeler_tmlar4d(example_dataset_adi):
    """
    Parameters
    ----------
    example_dataset_adi : fixture
        Taken automatically from ``conftest.py``.
    """

    dataset = copy.copy(example_dataset_adi)

    labeler_tmlar4d = dataLabeler_type_test(dataset, "tmlar4d")

    return labeler_tmlar4d


@fixture(scope="session")
def dataLabeler_pw3d(example_dataset_adi):
    """
    Parameters
    ----------
    example_dataset_adi : fixture
        Taken automatically from ``conftest.py``.
    """

    dataset = copy.copy(example_dataset_adi)

    labeler_pw3d = dataLabeler_type_test(dataset, "pw3d")

    return labeler_pw3d


@fixture(scope="session")
def dataLabeler_pw2d(example_dataset_adi):
    """
    Parameters
    ----------
    example_dataset_adi : fixture
        Taken automatically from ``conftest.py``.
    """

    dataset = copy.copy(example_dataset_adi)

    labeler_pw2d = dataLabeler_type_test(dataset, "pw2d")

    return labeler_pw2d


@fixture(scope="session")
def models_mlar(dataLabeler_mlar):

    labeler_mlar = dataLabeler_mlar

    model = Model(labeler_mlar, layer_type=('conv3d', 'conv3d'),
                  conv_nfilters=(40, 80), kernel_sizes=((3, 3, 3), (2, 2, 2)),
                  conv_strides=((1, 1, 1), (1, 1, 1)), conv_padding='same',
                  dilation_rate=((1, 1, 1), (1, 1, 1)), pool_layers=2,
                  pool_func='ave', pool_sizes=((2, 2, 2), (2, 2, 2)),
                  pool_strides=((2, 2, 2), (2, 2, 2)), rec_hidden_states=128,
                  dense_units=128, activation='relu')

    model.train(epochs=1, retrain=False)

    return model


@fixture(scope="session")
def models_tmlar(dataLabeler_tmlar):

    labeler_tmlar = dataLabeler_tmlar

    model = Model(labeler_tmlar, layer_type=('conv3d', 'conv3d'),
                  conv_nfilters=(40, 80), kernel_sizes=((3, 3, 3), (2, 2, 2)),
                  conv_strides=((1, 1, 1), (1, 1, 1)), conv_padding='same',
                  dilation_rate=((1, 1, 1), (1, 1, 1)), pool_layers=2,
                  pool_func='ave', pool_sizes=((2, 2, 2), (2, 2, 2)),
                  pool_strides=((2, 2, 2), (2, 2, 2)), rec_hidden_states=128,
                  dense_units=128, activation='relu')

    model.train(epochs=1, retrain=False)

    return model


@fixture(scope="session")
def models_tmlar4d(dataLabeler_tmlar4d):

    labeler_tmlar4d = dataLabeler_tmlar4d

    model = Model(labeler_tmlar4d, layer_type=('conv3d', 'conv3d'),
                  conv_nfilters=(40, 80), kernel_sizes=((3, 3, 3), (2, 2, 2)),
                  conv_strides=((1, 1, 1), (1, 1, 1)), conv_padding='same',
                  dilation_rate=((1, 1, 1), (1, 1, 1)), pool_layers=2,
                  pool_func='ave', pool_sizes=((2, 2, 2), (2, 2, 2)),
                  pool_strides=((2, 2, 2), (2, 2, 2)), rec_hidden_states=128,
                  dense_units=128, activation='relu')

    model.train(epochs=1, retrain=False)

    return model


@fixture(scope="session")
def models_pw3d(dataLabeler_pw3d):

    labeler_pw3d = dataLabeler_pw3d

    model = Model(labeler_pw3d, layer_type=('conv3d', 'conv3d'),
                  conv_nfilters=(40, 80), kernel_sizes=((3, 3, 3), (2, 2, 2)),
                  conv_strides=((1, 1, 1), (1, 1, 1)), conv_padding='same',
                  dilation_rate=((1, 1, 1), (1, 1, 1)), pool_layers=2,
                  pool_func='ave', pool_sizes=((2, 2, 2), (2, 2, 2)),
                  pool_strides=((2, 2, 2), (2, 2, 2)), rec_hidden_states=128,
                  dense_units=128, activation='relu')

    model.train(epochs=1, retrain=False)

    return model


@fixture(scope="session")
def models_pw2d(dataLabeler_pw2d):

    labeler_pw2d = dataLabeler_pw2d

    model = Model(labeler_pw2d, layer_type=('conv3d', 'conv3d'),
                  conv_nfilters=(40, 80), kernel_sizes=((3, 3, 3), (2, 2, 2)),
                  conv_strides=((1, 1, 1), (1, 1, 1)), conv_padding='same',
                  dilation_rate=((1, 1, 1), (1, 1, 1)), pool_layers=2,
                  pool_func='ave', pool_sizes=((2, 2, 2), (2, 2, 2)),
                  pool_strides=((2, 2, 2), (2, 2, 2)), rec_hidden_states=128,
                  dense_units=128, activation='relu')

    model.train(epochs=1, retrain=False)

    return model



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