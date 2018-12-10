from keras.layers import Add, Conv2D, Input, Lambda, Activation
from keras.models import Model

from .common import SubpixelConv2D, Normalization, Denormalization, Denormalization_m11


def edsr_generator(scale=4, num_filters=64, num_res_blocks=16):
    """
    Returns an EDSR model that can be used as generator in an SRGAN-like network.
    """
    return edsr(scale=scale, num_filters=num_filters, num_res_blocks=num_res_blocks, tanh_activation=True)


def edsr(scale, num_filters=64, num_res_blocks=8, res_block_scaling=None, tanh_activation=False):
    x_in = Input(shape=(None, None, 3))
    x = Normalization()(x_in)

    x = b = Conv2D(num_filters, 3, padding='same')(x)
    for i in range(num_res_blocks):
        b = res_block(b, num_filters, res_block_scaling)
    b = Conv2D(num_filters, 3, padding='same')(b)
    x = Add()([x, b])

    x = upsample(x, scale, num_filters)
    x = Conv2D(3, 3, padding='same')(x)

    if tanh_activation:
        x = Activation('tanh')(x)
        x = Denormalization_m11()(x)
    else:
        x = Denormalization()(x)

    return Model(x_in, x, name="edsr")


def res_block(x_in, filters, scaling):
    x = Conv2D(filters, 3, padding='same')(x_in)
    x = Activation('relu')(x)
    x = Conv2D(filters, 3, padding='same')(x)
    x = Add()([x_in, x])
    if scaling:
        x = Lambda(lambda t: t * scaling)(x)
    return x


def upsample(x, scale, num_filters):
    def upsample_1(x, factor, **kwargs):
        x = Conv2D(num_filters * (factor ** 2), 3, padding='same', **kwargs)(x)
        return SubpixelConv2D(factor)(x)

    if scale == 2:
        x = upsample_1(x, 2, name='conv2d_1_scale_2')
    elif scale == 3:
        x = upsample_1(x, 3, name='conv2d_1_scale_3')
    elif scale == 4:
        x = upsample_1(x, 2, name='conv2d_1_scale_2')
        x = upsample_1(x, 2, name='conv2d_2_scale_2')

    return x
