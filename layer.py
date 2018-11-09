# ----------------------------------------------------------------------------------
#  See https://github.com/ychfan/tf_estimator_barebone/blob/master/common/layers.py
# ----------------------------------------------------------------------------------

import tensorflow as tf


class Conv2DWeightNorm(tf.layers.Conv2D):

    def build(self, input_shape):
        self.wn_g = self.add_weight(
            name='wn_g',
            shape=(self.filters,),
            dtype=self.dtype,
            initializer=tf.initializers.ones,
            trainable=True,
        )
        super(Conv2DWeightNorm, self).build(input_shape)
        square_sum = tf.reduce_sum(
            tf.square(self.kernel), [0, 1, 2], keepdims=False)
        inv_norm = tf.rsqrt(square_sum)
        self.kernel = self.kernel * (inv_norm * self.wn_g)


def conv2d_weight_norm(inputs,
                       filters,
                       kernel_size,
                       strides=(1, 1),
                       padding='valid',
                       data_format='channels_last',
                       dilation_rate=(1, 1),
                       activation=None,
                       use_bias=True,
                       kernel_initializer=None,
                       bias_initializer=tf.zeros_initializer(),
                       kernel_regularizer=None,
                       bias_regularizer=None,
                       activity_regularizer=None,
                       kernel_constraint=None,
                       bias_constraint=None,
                       trainable=True,
                       name=None,
                       reuse=None):
    layer = Conv2DWeightNorm(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint,
        trainable=trainable,
        name=name,
        dtype=inputs.dtype.base_dtype,
        _reuse=reuse,
        _scope=name)
    return layer.apply(inputs)