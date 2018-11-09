import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Add, Input, Lambda, Activation
from tensorflow.keras.models import Model

from .common import SubpixelConv2D, Normalization, Denormalization
from layer import conv2d_weight_norm


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
    m = conv2d_weight_norm(x, num_filters, 3, padding='valid')
    for i in range(num_res_blocks):
        m = res_block(m, num_filters, res_block_expansion, kernel_size=3, scaling=res_block_scaling)
    m = conv2d_weight_norm(m, 3 * scale ** 2, 3, padding='valid', name=f'conv2d_main_scale_{scale}')
    m = SubpixelConv2D(scale)(m)

    # skip branch
    s = conv2d_weight_norm(x, 3 * scale ** 2, 5, padding='valid', name=f'conv2d_skip_scale_{scale}')
    s = SubpixelConv2D(scale)(s)

    x = Add()([m, s])
    x = Denormalization()(x)

    return Model(x_in, x, name="wdsr-b")


def res_block_a(x_in, num_filters, expansion, kernel_size, scaling):
    x = conv2d_weight_norm(x_in, num_filters * expansion, kernel_size, padding='same')
    x = Activation('relu')(x)
    x = conv2d_weight_norm(x, num_filters, kernel_size, padding='same')
    x = Add()([x_in, x])
    if scaling:
        x = Lambda(lambda t: t * scaling)(x)
    return x


def res_block_b(x_in, num_filters, expansion, kernel_size, scaling):
    linear = 0.8
    x = conv2d_weight_norm(x_in, num_filters * expansion, 1, padding='same')
    x = Activation('relu')(x)
    x = conv2d_weight_norm(x, int(num_filters * linear), 1, padding='same')
    x = conv2d_weight_norm(x, num_filters, kernel_size, padding='same')
    x = Add()([x_in, x])
    if scaling:
        x = Lambda(lambda t: t * scaling)(x)
    return x


def PadSymmetricInTestPhase():
    pad = Lambda(lambda x: K.in_train_phase(x, tf.pad(x, tf.constant([[0, 0], [2, 2], [2, 2], [0, 0]]), 'SYMMETRIC')))
    pad.uses_learning_phase = True
    return pad
