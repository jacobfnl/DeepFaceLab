import keras
from keras.layers import PReLU, LeakyReLU, K, BatchNormalization, Add, np


class Upscale(object):
    def __init__(self, dim, padding='zero', norm='', act='', **kwargs):
        self.dim = dim
        self.padding = padding
        self.norm = norm
        self.act = act

    def __call__(self, x):
        x = Conv2D(self.dim * 4, kernel_size=3, strides=1, padding=self.padding)(x)
        x = Act(self.act)(x)
        x = Norm(self.norm)(x)
        x = PixelShuffler()(x)
        return x


class Downscale(object):
    def __init__(self, dim, padding='zero', norm='', act='', **kwargs):
        self.dim = dim
        self.padding = padding
        self.norm = norm
        self.act = act

    def __call__(self, x):
        x = Conv2D(self.dim, kernel_size=5, strides=2, padding=self.padding)(x)
        x = Act(self.act)(x)
        x = Norm(self.norm)(x)
        return x


class ResidualBlock(object):
    def __init__(self, filters, kernel_size=3, padding='zero', norm='', act='', **kwargs):
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.norm = norm
        self.act = act

    def __call__(self, inp):
        x = inp
        x = Conv2D(self.filters, kernel_size=self.kernel_size, padding=self.padding)(x)
        x = Act(self.act, lrelu_alpha=0.2)(x)
        x = Norm(self.norm)(x)
        x = Conv2D(self.filters, kernel_size=self.kernel_size, padding=self.padding)(x)
        x = Add()([x, inp])
        x = Act(self.act, lrelu_alpha=0.2)(x)
        x = Norm(self.norm)(x)
        return x


class Act(object):
    def __init__(self, act='', lrelu_alpha=0.1):
        self.act = act,
        self.lrelu_alpha = lrelu_alpha

    def __call__(self, x):
        if self.act == 'prelu':
            return PReLU()(x)
        else:
            return LeakyReLU(alpha=self.lrelu_alpha)(x)


class ToBgr(object):
    def __init__(self, output_nc, padding='zero', **kwargs):
        self.output_nc = output_nc,
        self.padding = padding

    def __call__(self, x):
        return Conv2D(self.output_nc[0], kernel_size=5, padding=self.padding, activation='sigmoid')(x)


class Conv2D(object):
    def __init__(self, *args, **kwargs):
        self.reflect_pad = False
        padding = kwargs.get('padding', '')
        if padding == 'zero':
            kwargs['padding'] = 'same'
        if padding == 'reflect':
            kernel_size = kwargs['kernel_size']
            if (kernel_size % 2) == 1:
                self.pad = (kernel_size // 2,) * 2
                kwargs['padding'] = 'valid'
                self.reflect_pad = True
        self.func = keras.layers.Conv2D(*args, **kwargs)

    def __call__(self, x):
        if self.reflect_pad:
            return ReflectionPadding2D(self.pad)(x)
        return self.func(x)


class ReflectionPadding2D(keras.layers.Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [keras.layers.InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        """ If you are using "channels_last" configuration"""
        return s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3]

    def call(self, x, mask=None):
        w_pad, h_pad = self.padding
        return K.tf.pad(x, [[0, 0], [h_pad, h_pad], [w_pad, w_pad], [0, 0]], 'REFLECT')


class Norm(object):
    def __init__(self, norm=''):
        self.norm = norm

    def __call__(self, x):
        if self.norm == 'bn':
            return BatchNormalization(axis=-1)(x)
        else:
            return x

class PixelShuffler(keras.layers.Layer):
    def __init__(self, size=(2, 2), data_format='channels_last', **kwargs):
        super(PixelShuffler, self).__init__(**kwargs)
        self.data_format = data_format
        self.size = size

    def call(self, inputs):
        input_shape = K.shape(inputs)
        if K.int_shape(input_shape)[0] != 4:
            raise ValueError('Inputs should have rank 4; Received input shape:', str(K.int_shape(inputs)))

        if self.data_format == 'channels_first':
            return K.tf.depth_to_space(inputs, self.size[0], 'NCHW')

        elif self.data_format == 'channels_last':
            return K.tf.depth_to_space(inputs, self.size[0], 'NHWC')

    def compute_output_shape(self, input_shape):
        if len(input_shape) != 4:
            raise ValueError('Inputs should have rank ' +
                             str(4) +
                             '; Received input shape:', str(input_shape))

        if self.data_format == 'channels_first':
            height = input_shape[2] * self.size[0] if input_shape[2] is not None else None
            width = input_shape[3] * self.size[1] if input_shape[3] is not None else None
            channels = input_shape[1] // self.size[0] // self.size[1]

            if channels * self.size[0] * self.size[1] != input_shape[1]:
                raise ValueError('channels of input and size are incompatible')

            return (input_shape[0],
                    channels,
                    height,
                    width)

        elif self.data_format == 'channels_last':
            height = input_shape[1] * self.size[0] if input_shape[1] is not None else None
            width = input_shape[2] * self.size[1] if input_shape[2] is not None else None
            channels = input_shape[3] // self.size[0] // self.size[1]

            if channels * self.size[0] * self.size[1] != input_shape[3]:
                raise ValueError('channels of input and size are incompatible')

            return (input_shape[0],
                    height,
                    width,
                    channels)

    def get_config(self):
        config = {'size': self.size,
                  'data_format': self.data_format}
        base_config = super(PixelShuffler, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))
