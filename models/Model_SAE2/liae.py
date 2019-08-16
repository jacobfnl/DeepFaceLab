from keras.layers import Dense, Reshape, Flatten, K

from models.Model_SAE2.layers.layers import Downscale, Upscale, ToBgr, ResidualBlock


def LIAEEncFlow(resolution, ch_dims, **kwargs):
    def func(input):
        dims = K.int_shape(input)[-1] * ch_dims

        x = input
        x = Downscale(dims)(x)
        x = Downscale(dims * 2)(x)
        x = Downscale(dims * 4)(x)
        x = Downscale(dims * 8)(x)

        x = Flatten()(x)
        return x

    return func


def LIAEInterFlow(resolution, ae_dims=256, **kwargs):
    lowest_dense_res = resolution // 16

    def func(input):
        x = input[0]
        x = Dense(ae_dims)(x)
        x = Dense(lowest_dense_res * lowest_dense_res * ae_dims * 2)(x)
        x = Reshape((lowest_dense_res, lowest_dense_res, ae_dims * 2))(x)
        x = Upscale(ae_dims * 2)(x)
        return x

    return func


def LIAEDecFlow(output_nc, ch_dims, multiscale_count=1, add_residual_blocks=False, **kwargs):
    dims = output_nc * ch_dims

    def func(input):
        x = input[0]

        outputs = []
        x1 = Upscale(dims * 8)(x)

        if add_residual_blocks:
            x1 = ResidualBlock(dims * 8)(x1)
            x1 = ResidualBlock(dims * 8)(x1)

        if multiscale_count >= 3:
            outputs += [ToBgr(output_nc)(x1)]

        x2 = Upscale(dims * 4)(x1)

        if add_residual_blocks:
            x2 = ResidualBlock(dims * 4)(x2)
            x2 = ResidualBlock(dims * 4)(x2)

        if multiscale_count >= 2:
            outputs += [ToBgr(output_nc)(x2)]

        x3 = Upscale(dims * 2)(x2)

        if add_residual_blocks:
            x3 = ResidualBlock(dims * 2)(x3)
            x3 = ResidualBlock(dims * 2)(x3)

        outputs += [ToBgr(output_nc)(x3)]

        return outputs
    return func
