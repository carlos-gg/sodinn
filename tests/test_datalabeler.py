"""
Tests for DataLabeler using different sample type
"""


def test_dataLabeler_mlar(dataLabeler_mlar):

    try:
        labeler_mlar = dataLabeler_mlar
    except TypeError:
        raise

    return True


def test_dataLabeler_tmlar(dataLabeler_tmlar):

    try:
        labeler_tmlar = dataLabeler_tmlar
    except TypeError:
        raise

    return True


def test_dataLabeler_tmlar4d(dataLabeler_tmlar4d):

    try:
        labeler_tmlar4d = dataLabeler_tmlar4d
    except TypeError:
        raise

    return True


def test_dataLabeler_pw3d(dataLabeler_pw3d):

    try:
        labeler_pw3d = dataLabeler_pw3d
    except TypeError:
        raise

    return True


def test_dataLabeler_pw2d(dataLabeler_pw2d):

    try:
        labeler_pw2d = dataLabeler_pw2d
    except TypeError:
        raise

    return True
