"""
Tests for prediction
"""
from sodinn import Predictor


def test_prediction_mlar(example_dataset_adi_fc, dataLabeler_mlar, models_mlar):

    try:
        predictor = Predictor(dataLabeler_mlar, models_mlar)
    except TypeError:
        raise

    dataset = example_dataset_adi_fc

    cube = dataset.cube
    pa = dataset.angles

    try:
        predictor.run(cube=cube, pa=pa)
    except TypeError:
        raise

    return True
