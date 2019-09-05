import tensorflow as tf
from keras.layers import Layer, Conv2D, LeakyReLU,  Add, Flatten, Dense, Reshape, Lambda
from keras import Model, Input


class PixelShuffler(Layer):
    """ Assumes channels-last """
    def __init__(self, size=(2, 2), **kwargs):
        super(PixelShuffler, self).__init__(**kwargs)
        self.block_size = size[0]

    def call(self, inputs, **kwargs):
        return tf.nn.depth_to_space(inputs, self.block_size, data_format='NHWC')

    def compute_output_shape(self, input_shape):
        if len(input_shape) != 4:
            raise ValueError(f'Inputs should have rank 4; Received input shape: {input_shape}')
        height = input_shape[1] * self.block_size if input_shape[1] is not None else None
        width = input_shape[2] * self.block_size if input_shape[2] is not None else None
        channels = input_shape[3] // self.block_size // self.block_size

        if channels * self.block_size * self.block_size != input_shape[3]:
            raise ValueError('channels of input and size are incompatible')

        return (input_shape[0],
                height,
                width,
                channels)


class Downscale(object):
    def __init__(self, dim, **kwargs):
        self.filters = dim * 4

    def __call__(self, inputs, **kwargs):
        x = Conv2D(self.filters, kernel_size=3, strides=1, padding='same')(inputs)
        outputs = LeakyReLU()(x)
        return outputs


class Upscale(object):
    def __init__(self, dim, **kwargs):
        self.filters = dim * 4

    def __call__(self, inputs, **kwargs):
        x = Conv2D(self.filters, kernel_size=3, strides=1, padding='same')(inputs)
        x = LeakyReLU()(x)
        outputs = PixelShuffler()(x)
        return outputs


class ResidualBlock(object):
    def __init__(self, filters, **kwargs):
        self.filters = filters

    def __call__(self, inputs, **kwargs):
        x = Conv2D(self.filters, kernel_size=3, padding='same')(inputs)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(self.filters, kernel_size=3, padding='same')(x)
        x = Add()([x, inputs])
        outputs = LeakyReLU(alpha=0.2)(x)
        return outputs


class ToBgr(object):
    def __init__(self, num_channels, padding='zero', **kwargs):
        self.filters = num_channels

    def __call__(self, inputs, **kwargs):
        outputs = Conv2D(self.filters, kernel_size=5, padding='same', activation='sigmoid')(inputs)
        return outputs


def df_encoder(num_channels=1, resolution=32, ch_dims=1, ae_dims=1, name='encoder') -> Model:
    lowest_dense_res = resolution // 16
    dims = num_channels * ch_dims

    inputs = Input(shape=(resolution, resolution, num_channels), name='image')
    x = Downscale(dims)(inputs)
    x = Downscale(dims * 2)(x)
    x = Downscale(dims * 4)(x)
    x = Flatten()(x)
    x = Dense(ae_dims)(x)
    x = Dense(lowest_dense_res**2 * ae_dims)(x)
    x = Reshape((lowest_dense_res, lowest_dense_res, ae_dims))(x)
    outputs = Upscale(ae_dims)(x)
    return Model(inputs=inputs, outputs=outputs, name=name)


def liae_encoder(num_channels=1, resolution=32, ch_dims=1, name='encoder') -> Model:
    dims = num_channels * ch_dims

    inputs = Input(shape=(resolution, resolution, num_channels), name='image')
    x = Downscale(dims)(inputs)
    x = Downscale(dims * 2)(x)
    x = Downscale(dims * 4)(x)
    outputs = Flatten()(x)
    return Model(inputs=inputs, outputs=outputs, name=name)


def liae_interpolator(num_channels=1, resolution=32, ch_dims=1, ae_dims=1, name='interpolator'):
    lowest_dense_res = resolution // 16
    dims = num_channels * ch_dims
    size = dims * 16 * resolution**2

    inputs = Input(shape=(size,))
    x = Dense(ae_dims)(inputs)
    x = Dense(lowest_dense_res**2 * ae_dims * 2)(x)
    x = Reshape((lowest_dense_res, lowest_dense_res, ae_dims * 2))(x)
    outputs = Upscale(ae_dims * 2)(x)
    return Model(inputs=inputs, outputs=outputs, name=name)


def decoder(num_channels=1, resolution=32, ch_dims=1, ae_dims=1, add_residual_blocks=True, multiscale_count=3, name='decoder') -> Model:
    lowest_dense_res = 2 * (resolution // 16)
    dims = num_channels * ch_dims

    inputs = Input(shape=(lowest_dense_res, lowest_dense_res, ae_dims))
    outputs = []

    x = Upscale(dims * 8)(inputs)
    if add_residual_blocks:
        x = ResidualBlock(dims * 8)(x)
        x = ResidualBlock(dims * 8)(x)
    if multiscale_count >= 3:
        output = ToBgr(num_channels)(x)
        outputs.append(output)
    x = Upscale(dims * 4)(x)
    if add_residual_blocks:
        x = ResidualBlock(dims * 4)(x)
        x = ResidualBlock(dims * 4)(x)
    if multiscale_count >= 2:
        output = ToBgr(num_channels)(x)
        outputs.append(output)
    x = Upscale(dims * 2)(x)
    if add_residual_blocks:
        x = ResidualBlock(dims * 2)(x)
        x = ResidualBlock(dims * 2)(x)
    output = ToBgr(num_channels)(x)
    outputs.append(output)

    return Model(inputs=inputs, outputs=outputs, name=name)


def mask_decoder(num_channels=1, resolution=32, ch_dims=1, ae_dims=1, name='mask_decoder') -> Model:
    lowest_dense_res = 2 * (resolution // 16)
    dims = num_channels * ch_dims

    inputs = Input(shape=(lowest_dense_res, lowest_dense_res, ae_dims))
    x = Upscale(dims * 8)(inputs)
    x = Upscale(dims * 4)(x)
    x = Upscale(dims * 2)(x)
    outputs = ToBgr(num_channels)(x)

    return Model(inputs=inputs, outputs=outputs, name=name)
