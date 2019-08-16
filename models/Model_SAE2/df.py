from functools import partial

from keras.layers import Input, Dense, K, Flatten, Reshape
from keras.models import Model

from models import Model_SAE2
from models.Model_SAE2.layers.layers import Downscale, Upscale, ResidualBlock


def encoder(bgr_shape, resolution, ae_dims, e_ch_dims, **common_flow_kwargs):
    inputs = Input(bgr_shape)
    outputs = DFEncFlow(resolution, ae_dims=ae_dims, ch_dims=e_ch_dims, **common_flow_kwargs)(inputs)
    return Model(inputs=inputs, outputs=outputs)


def decoder(encoder_outputs, bgr_shape, d_ch_dims, ms_count, d_residual_blocks, **common_flow_kwargs):
    inputs = [Input(K.int_shape(x)[1:]) for x in encoder_outputs]
    outputs = DFDecFlow(bgr_shape[2], ch_dims=d_ch_dims, multiscale_count=ms_count,
                        add_residual_blocks=d_residual_blocks, **common_flow_kwargs)(inputs)
    return Model(inputs=inputs, outputs=outputs)


def decoder_mask(encoder_outputs, mask_shape, d_ch_dims, **common_flow_kwargs):
    inputs = [Input(K.int_shape(x)[1:]) for x in encoder_outputs]
    outputs = DFDecFlow(mask_shape[2], ch_dims=d_ch_dims, **common_flow_kwargs)(inputs)
    return Model(inputs=inputs, outputs=outputs)


def DFEncFlow(resolution, ae_dims, ch_dims, **kwargs):
    lowest_dense_res = resolution // 16

    def func(input):
        x = input

        dims = K.int_shape(input)[-1] * ch_dims
        x = Downscale(dims)(x)
        x = Downscale(dims * 2)(x)
        x = Downscale(dims * 4)(x)
        x = Downscale(dims * 8)(x)

        x = Dense(ae_dims)(Flatten()(x))
        x = Dense(lowest_dense_res * lowest_dense_res * ae_dims)(x)
        x = Reshape((lowest_dense_res, lowest_dense_res, ae_dims))(x)
        x = Upscale(ae_dims)(x)
        return x

    return func


def DFDecFlow(output_nc, ch_dims, multiscale_count=1, add_residual_blocks=False, **kwargs):
    to_bgr = partial(Model_SAE2.Model.to_bgr, **kwargs)
    dims = output_nc * ch_dims

    def func(input):
        x = input[0]

        outputs = []
        x1 = Upscale(dims * 8)(x)

        if add_residual_blocks:
            x1 = ResidualBlock(dims * 8)(x1)
            x1 = ResidualBlock(dims * 8)(x1)


        if multiscale_count >= 3:
            outputs += [to_bgr(output_nc)(x1)]

        x2 = Upscale(dims * 4)(x1)

        if add_residual_blocks:
            x2 = ResidualBlock(dims * 4)(x2)
            x2 = ResidualBlock(dims * 4)(x2)

        if multiscale_count >= 2:
            outputs += [to_bgr(output_nc)(x2)]

        x3 = Upscale(dims * 2)(x2)

        if add_residual_blocks:
            x3 = ResidualBlock(dims * 2)(x3)
            x3 = ResidualBlock(dims * 2)(x3)

        outputs += [to_bgr(output_nc)(x3)]

        return outputs

    return func
