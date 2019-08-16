from functools import partial

from keras.layers import Input, Dense, K, Flatten, Reshape, np
from keras.models import Model

from models import Model_SAE2
from models.Model_SAE2.layers.layers import Downscale, Upscale, ResidualBlock, ToBgr


class DF(object):
    def __init__(self, bgr_shape, mask_shape, resolution: int, ae_dims: int, e_ch_dims: int, d_ch_dims: int,
                 multiscale_count,
                 add_residual_blocks, **common_flow_kwargs):
        self.bgr_shape = bgr_shape
        self.mask_shape = mask_shape
        self.resolution = resolution
        self.ae_dims = ae_dims
        self.e_ch_dims = e_ch_dims
        self.d_ch_dims = d_ch_dims
        self.multiscale_count = multiscale_count
        self.add_residual_blocks = add_residual_blocks
        self.common_flow_kwargs = common_flow_kwargs
        self.enc_shape = None

    def encoder(self):
        lowest_dense_res = self.resolution // 16
        dims = self.bgr_shape[-1] * self.e_ch_dims

        inputs = Input(self.bgr_shape)

        x = Downscale(dims)(inputs)
        x = Downscale(dims * 2)(x)
        x = Downscale(dims * 4)(x)
        x = Downscale(dims * 8)(x)

        x = Flatten()(x)
        x = Dense(self.ae_dims)(x)
        x = Dense(lowest_dense_res * lowest_dense_res * self.ae_dims)(x)
        x = Reshape((lowest_dense_res, lowest_dense_res, self.ae_dims))(x)
        outputs = Upscale(self.ae_dims)(x)

        e = Model(inputs=inputs, outputs=outputs)
        self.enc_shape = e.output_shape
        return e

    def decoder(self, **kwargs):

        output_nc = self.bgr_shape[2]
        dims = self.bgr_shape[2] * self.d_ch_dims
        inputs = [Input(self.enc_shape[1:])]

        x = inputs[0]

        outputs = []
        x1 = Upscale(dims * 8, **kwargs)(x)

        if self.add_residual_blocks:
            x1 = ResidualBlock(dims * 8, **kwargs)(x1)
            x1 = ResidualBlock(dims * 8, **kwargs)(x1)

        if self.multiscale_count >= 3:
            outputs += [ToBgr(output_nc, **kwargs)(x1)]

        x2 = Upscale(dims * 4, **kwargs)(x1)

        if self.add_residual_blocks:
            x2 = ResidualBlock(dims * 4, **kwargs)(x2)
            x2 = ResidualBlock(dims * 4, **kwargs)(x2)

        if self.multiscale_count >= 2:
            outputs += [ToBgr(output_nc, **kwargs)(x2)]

        x3 = Upscale(dims * 2, **kwargs)(x2)

        if self.add_residual_blocks:
            x3 = ResidualBlock(dims * 2, **kwargs)(x3)
            x3 = ResidualBlock(dims * 2, **kwargs)(x3)

        outputs += [ToBgr(output_nc, **kwargs)(x3)]

        return Model(inputs=inputs, outputs=outputs)

    def decoder_mask(self, **kwargs):
        output_nc = self.mask_shape[2]
        dims = self.mask_shape[2] * self.d_ch_dims
        inputs = [Input(self.enc_shape[1:])]

        x = inputs[0]

        outputs = []
        x1 = Upscale(dims * 8)(x)
        x2 = Upscale(dims * 4)(x1)
        x3 = Upscale(dims * 2)(x2)
        outputs += [ToBgr(output_nc)(x3)]

        return Model(inputs=inputs, outputs=outputs)
