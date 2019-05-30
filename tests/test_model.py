"""
Tests for models
"""

from sodinn import Model


def test_models_mlar(dataLabeler_mlar):

    labeler_mlar = dataLabeler_mlar

    try:
        model = Model(labeler_mlar, layer_type=('conv3d', 'conv3d'),
                      conv_nfilters=(40, 80),
                      kernel_sizes=((3, 3, 3), (2, 2, 2)),
                      conv_strides=((1, 1, 1), (1, 1, 1)), conv_padding='same',
                      dilation_rate=((1, 1, 1), (1, 1, 1)), pool_layers=2,
                      pool_func='ave', pool_sizes=((2, 2, 2), (2, 2, 2)),
                      pool_strides=((2, 2, 2), (2, 2, 2)),
                      rec_hidden_states=128, dense_units=128, activation='relu')
    except TypeError:
        raise

    try:
        model.train(epochs=1, retrain=False)
    except TypeError:
        raise

    return True


def test_models_tmlar(dataLabeler_tmlar):

    labeler_tmlar = dataLabeler_tmlar

    try:
        model = Model(labeler_tmlar, layer_type=('conv3d', 'conv3d'),
                      conv_nfilters=(40, 80),
                      kernel_sizes=((3, 3, 3), (2, 2, 2)),
                      conv_strides=((1, 1, 1), (1, 1, 1)), conv_padding='same',
                      dilation_rate=((1, 1, 1), (1, 1, 1)), pool_layers=2,
                      pool_func='ave', pool_sizes=((2, 2, 2), (2, 2, 2)),
                      pool_strides=((2, 2, 2), (2, 2, 2)),
                      rec_hidden_states=128, dense_units=128, activation='relu')
    except TypeError:
        raise

    try:
        model.train(epochs=1, retrain=False)
    except TypeError:
        raise

    return True


def test_models_tmlar4d(dataLabeler_tmlar4d):

    labeler_tmlar4d = dataLabeler_tmlar4d

    try:
        model = Model(labeler_tmlar4d, layer_type=('conv3d', 'conv3d'),
                      conv_nfilters=(40, 80),
                      kernel_sizes=((3, 3, 3), (2, 2, 2)),
                      conv_strides=((1, 1, 1), (1, 1, 1)), conv_padding='same',
                      dilation_rate=((1, 1, 1), (1, 1, 1)), pool_layers=2,
                      pool_func='ave', pool_sizes=((2, 2, 2), (2, 2, 2)),
                      pool_strides=((2, 2, 2), (2, 2, 2)),
                      rec_hidden_states=128, dense_units=128, activation='relu')
    except TypeError:
        raise

    try:
        model.train(epochs=1, retrain=False)
    except TypeError:
        raise

    return True


def test_models_pw3d(dataLabeler_pw3d):

    labeler_pw3d = dataLabeler_pw3d

    try:
        model = Model(labeler_pw3d, layer_type=('conv3d', 'conv3d'),
                      conv_nfilters=(40, 80),
                      kernel_sizes=((3, 3, 3), (2, 2, 2)),
                      conv_strides=((1, 1, 1), (1, 1, 1)), conv_padding='same',
                      dilation_rate=((1, 1, 1), (1, 1, 1)), pool_layers=2,
                      pool_func='ave', pool_sizes=((2, 2, 2), (2, 2, 2)),
                      pool_strides=((2, 2, 2), (2, 2, 2)),
                      rec_hidden_states=128, dense_units=128, activation='relu')
    except TypeError:
        raise

    try:
        model.train(epochs=1, retrain=False)
    except TypeError:
        raise

    return True


def test_models_pw2d(dataLabeler_pw2d):

    labeler_pw2d = dataLabeler_pw2d

    try:
        model = Model(labeler_pw2d, layer_type=('conv3d', 'conv3d'),
                      conv_nfilters=(40, 80),
                      kernel_sizes=((3, 3, 3), (2, 2, 2)),
                      conv_strides=((1, 1, 1), (1, 1, 1)), conv_padding='same',
                      dilation_rate=((1, 1, 1), (1, 1, 1)), pool_layers=2,
                      pool_func='ave', pool_sizes=((2, 2, 2), (2, 2, 2)),
                      pool_strides=((2, 2, 2), (2, 2, 2)),
                      rec_hidden_states=128, dense_units=128, activation='relu')
    except TypeError:
        raise

    try:
        model.train(epochs=1, retrain=False)
    except TypeError:
        raise

    return True
