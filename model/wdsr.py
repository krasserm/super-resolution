import tensorflow as tf

from keras import backend as K
from keras.layers import Add, Conv2D, Input, Lambda, Activation
from keras.models import Model

from .common import SubpixelConv2D, Normalization, Denormalization


def wdsr_a(scale, num_filters=32, num_res_blocks=8, res_block_expansion=4, res_block_scaling=None):
    return wdsr(scale, num_filters, num_res_blocks, res_block_expansion, res_block_scaling, res_block_a)


def wdsr_b(scale, num_filters=32, num_res_blocks=8, res_block_expansion=6, res_block_scaling=None):
    return wdsr(scale, num_filters, num_res_blocks, res_block_expansion, res_block_scaling, res_block_b)


def wdsr(scale, num_filters, num_res_blocks, res_block_expansion, res_block_scaling, res_block):
    x_in = Input(shape=(None, None, 3))
    x = Normalization()(x_in)

    # pad input if in test phase
    x = PadSymmetricInTestPhase()(x)

    # main branch (revise padding)
    m = Conv2D(num_filters, 3, padding='valid')(x)
    for i in range(num_res_blocks):
        m = res_block(m, num_filters, res_block_expansion, kernel_size=3, scaling=res_block_scaling)
    m = Conv2D(3 * scale ** 2, 3, padding='valid', name=f'conv2d_main_scale_{scale}')(m)
    m = SubpixelConv2D(scale)(m)

    # skip branch
    s = Conv2D(3 * scale ** 2, 5, padding='valid', name=f'conv2d_skip_scale_{scale}')(x)
    s = SubpixelConv2D(scale)(s)

    x = Add()([m, s])
    x = Denormalization()(x)

    return Model(x_in, x, name="wdsr-b")


def res_block_a(x_in, num_filters, expansion, kernel_size, scaling):
    x = Conv2D(num_filters * expansion, kernel_size, padding='same')(x_in)
    x = Activation('relu')(x)
    x = Conv2D(num_filters, kernel_size, padding='same')(x)
    x = Add()([x_in, x])
    if scaling:
        x = Lambda(lambda t: t * scaling)(x)
    return x


def res_block_b(x_in, num_filters, expansion, kernel_size, scaling):
    linear = 0.8
    x = Conv2D(num_filters * expansion, 1, padding='same')(x_in)
    x = Activation('relu')(x)
    x = Conv2D(int(num_filters * linear), 1, padding='same')(x)
    x = Conv2D(num_filters, kernel_size, padding='same')(x)
    x = Add()([x_in, x])
    if scaling:
        x = Lambda(lambda t: t * scaling)(x)
    return x


def PadSymmetricInTestPhase():
    pad = Lambda(lambda x: K.in_train_phase(x, tf.pad(x, tf.constant([[0, 0], [2, 2], [2, 2], [0, 0]]), 'SYMMETRIC')))
    pad.uses_learning_phase = True
    return pad
