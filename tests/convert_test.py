import os
import numpy as np

from util import load_image

from . import IMG_PATH
from . import ARR_PATH


def lr(path, id, ext):
    return os.path.join(path, f'DIV2K_train_LR_bicubic/X2/000{id}x2.{ext}')


def hr(path, id, ext):
    return os.path.join(path, f'DIV2K_train_HR/000{id}.{ext}')


def test_conversion(conversion):
    for i in range(1, 5):
        lr_orig = np.array(load_image(lr(IMG_PATH, i, 'png')))
        hr_orig = np.array(load_image(hr(IMG_PATH, i, 'png')))

        lr_conv = np.load(lr(ARR_PATH, i, 'npy'))
        hr_conv = np.load(hr(ARR_PATH, i, 'npy'))

        assert np.array_equal(lr_orig, lr_conv)
        assert np.array_equal(hr_orig, hr_conv)
