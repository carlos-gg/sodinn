"""
Configuration file for pytest, containing global ("session-level") fixtures.

"""

from pytest import fixture
from astropy.utils.data import download_file
import vip_hci as vip


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