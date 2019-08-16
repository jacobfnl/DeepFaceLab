import keras
from keras import Input, Model
from keras.losses import mean_squared_error

from keras.utils import plot_model
from tensorflow.python.keras.optimizers import RMSprop

from models.Model_SAE2.models import DF


class Autoencoder(object):
    def create_model(self, bgr_shape, mask_shape, resolution, ae_dims, e_ch_dims, d_ch_dims, ms_count, d_residual_blocks, src_train, dst_train, **common_flow_kwargs):
        src_input = Input(shape=bgr_shape, name='src')
        dst_input = Input(shape=bgr_shape, name='dst')

        df = DF(bgr_shape, mask_shape, resolution, ae_dims, e_ch_dims, d_ch_dims, ms_count, d_residual_blocks, **common_flow_kwargs)
        encoder = df.encoder()
        decoder_src = df.decoder(**common_flow_kwargs)
        decoder_dst = df.decoder(**common_flow_kwargs)

        src_pred = decoder_src(encoder(src_input))
        dst_pred = decoder_dst(encoder(dst_input))

        model = Model(inputs=[src_input, dst_input], outputs=[src_pred, dst_pred])
        plot_model(model, 'multi_input_and_output_model.png', show_shapes=True)

        model.compile(optimizer='adam', loss=[mean_squared_error, mean_squared_error])
        model.fit([dst_train, dst_train], [dst_train, dst_train], batch_size=4, epochs=10)





