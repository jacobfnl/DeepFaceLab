from functools import partial

import cv2
import numpy as np

from facelib import FaceType
from interact import interact as io
from mathlib import get_power_of_two
from models import ModelBase
from nnlib import nnlib
from samplelib import *

from facelib import PoseEstimator

class AVATARModel(ModelBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs,
                            ask_sort_by_yaw=False,
                            ask_random_flip=False,
                            ask_src_scale_mod=False)

    #override
    def onInitializeOptions(self, is_first_run, ask_override):
        def_resolution = 128
        if is_first_run:
            self.options['resolution'] = io.input_int("Resolution ( 128,256 ?:help skip:%d) : " % def_resolution, def_resolution, [128,256], help_message="More resolution requires more VRAM and time to train. Value will be adjusted to multiple of 16.")
        else:
            self.options['resolution'] = self.options.get('resolution', def_resolution)


    #override
    def onInitialize(self, batch_size=-1, **in_options):


        exec(nnlib.code_import_all, locals(), globals())
        self.set_vram_batch_requirements({2:1})

        resolution = self.options['resolution']
        in_bgr_shape = (64, 64, 3)
        out_bgr_shape = (resolution, resolution, 3)
        mask_shape = (resolution, resolution, 1)
        bgrm_shape = (resolution, resolution, 4)

        ngf = 64
        ndf = 64
        lambda_A = 10
        lambda_B = 10

        use_batch_norm = True #created_batch_size > 1

        self.enc = modelify(AVATARModel.DFEncFlow ())( [Input(in_bgr_shape),] )
        self.BVAEResampler = Lambda ( lambda x: x[0] + K.exp(0.5*x[1])*K.random_normal(K.shape(x[0])),
                                        output_shape=K.int_shape(self.enc.outputs[0])[1:] )

        dec_Inputs = [ Input(K.int_shape( self.enc.outputs[0] )[1:]) ]
        self.decA = modelify(AVATARModel.DFDecFlow (out_bgr_shape[2])) (dec_Inputs)
        self.decB = modelify(AVATARModel.DFDecFlow (out_bgr_shape[2])) (dec_Inputs)

        self.DA = modelify(AVATARModel.PatchDiscriminator(ndf=ndf) ) (Input(out_bgr_shape))
        self.DB = modelify(AVATARModel.PatchDiscriminator(ndf=ndf) ) (Input(out_bgr_shape))

        if not self.is_first_run():
            weights_to_load = [
                (self.enc, 'enc.h5'),
                (self.decA, 'decA.h5'),
                (self.decB, 'decB.h5'),
                (self.DA, 'DA.h5'),
                (self.DB, 'DB.h5'),
            ]
            self.load_weights_safe(weights_to_load)

        warped_A0 = Input(in_bgr_shape)
        real_A0 = Input(out_bgr_shape)
        real_A0m = Input(mask_shape)
        
        warped_B0 = Input(in_bgr_shape)
        real_B0 = Input(out_bgr_shape)
        real_B0m = Input(mask_shape)
        
        #real_A0m_blurred = gaussian_blur( max(1, K.int_shape(x)[1] // 32) )(real_A0m)
        
        def BCELoss(logits, ones):
            if ones:
                return K.mean(K.binary_crossentropy(K.ones_like(logits),logits))
            else:
                return K.mean(K.binary_crossentropy(K.zeros_like(logits),logits))

        def MSELoss(labels,logits):
            return K.mean(K.square(labels-logits))

        def DLoss(labels,logits):
            return K.mean(K.binary_crossentropy(labels,logits))

        def MAELoss(t1,t2):
            return dssim(kernel_size=int(resolution/11.6),max_value=2.0)(t1+1,t2+1 )
            return K.mean(K.abs(t1 - t2) )

        #warped_A0_mean, warped_A0_log = self.enc (warped_A0)
        #warped_B0_mean, warped_B0_log = self.enc (warped_B0)
        #warped_A0_code = self.BVAEResampler([warped_A0_mean, warped_A0_log])
        #warped_B0_code = self.BVAEResampler([warped_B0_mean, warped_B0_log])
        warped_A0_code = self.enc (warped_A0)
        warped_B0_code = self.enc (warped_B0)

        rec_A0 = self.decA (warped_A0_code)
        rec_B0 = self.decB (warped_B0_code)
        rec_A0_B0 = self.decA (warped_B0_code)

        real_A0_d = self.DA(real_A0)
        real_A0_d_ones = K.ones_like(real_A0_d)

        rec_A0_d = self.DA(rec_A0)
        rec_A0_d_ones = K.ones_like(rec_A0_d)
        rec_A0_d_zeros = K.zeros_like(rec_A0_d)

        rec_A0_B0_d = self.DA(rec_A0_B0)
        rec_A0_B0_d_ones = K.ones_like(rec_A0_B0_d)
        rec_A0_B0_d_zeros = K.zeros_like(rec_A0_B0_d)
        
        real_B0_d = self.DB(real_B0)
        real_B0_d_ones = K.ones_like(real_B0_d)

        rec_B0_d = self.DB(rec_B0)
        rec_B0_d_ones = K.ones_like(rec_B0_d)
        rec_B0_d_zeros = K.zeros_like(rec_B0_d)

        self.G_view = K.function([warped_A0, warped_B0],[rec_A0, rec_B0, rec_A0_B0])

        if self.is_training_mode:
            def BVAELoss(beta=4):
                #keep in mind loss per sample, not per minibatch
                def func(input):
                    mean_t, logvar_t = input
                    #import code
                    #code.interact(local=dict(globals(), **locals()))
                    return beta * K.sum( -0.5*(1 + logvar_t - K.exp(logvar_t) - K.square(mean_t)), axis=1 )

                    #return beta * K.mean ( K.sum( -0.5*(1 + logvar_t - K.exp(logvar_t) - K.square(mean_t)), axis=1 ), axis=0, keepdims=True )
                return func

            #loss_A = DLoss(fake_A0_d_ones, fake_A0_d)
            #loss_A = DLoss(rec_A0_B0_d_ones, rec_A0_B0_d)
            loss_A = K.mean( 10 * dssim(kernel_size=int(resolution/11.6),max_value=1.0) ( rec_A0, real_A0*real_A0m + (1-real_A0m)*0.5 ) )
            #loss_A = 10*K.mean( K.abs ( (rec_A0+1) - ( (real_A0+1)*real_A0m + (1.0-real_A0m)-1.0 ) ) )

            #loss_A += BVAELoss(4)([warped_A0_mean, warped_A0_log])

            weights_A = self.enc.trainable_weights + self.decA.trainable_weights

            #loss_B = ( DLoss(fake_B0_d_ones, fake_B0_d) + DLoss(rec_A0_B0_d_ones, rec_A0_B0_d) ) * 0.5
            loss_B = K.mean( 10 * dssim(kernel_size=int(resolution/11.6),max_value=1.0) ( rec_B0, real_B0*real_B0m + (1-real_B0m)*0.5) )
            #loss_B = 10*K.mean( K.abs ( (rec_B0+1) - ( (real_B0+1)*real_B0m + (1.0-real_B0m)-1.0 ) ) )
            #loss_B += BVAELoss(4)([warped_B0_mean, warped_B0_log])

            weights_B = self.enc.trainable_weights + self.decB.trainable_weights

            def opt(lr=2e-5):
                return Adam(lr=lr, beta_1=0.5, beta_2=0.999, tf_cpu_mode=2)#, clipnorm=1)

            self.A_train = K.function ([warped_A0, real_A0, real_A0m, warped_B0, real_B0, real_B0m],[ loss_A ],
                                        opt(lr=2e-5).get_updates(loss_A, weights_A) )

            self.B_train = K.function ([warped_A0, real_A0, real_A0m, warped_B0, real_B0, real_B0m],[ loss_B ],
                                        opt(lr=2e-5).get_updates(loss_B, weights_B) )


            ###########
            """
            loss_DA = ( DLoss(real_A0_d_ones, real_A0_d ) + \
                        DLoss(rec_A0_B0_d_zeros,  rec_A0_B0_d ) ) * 0.5
                        #DLoss(fake_A0_d_zeros, fake_A0_d ) ) * 0.5

            self.DA_train = K.function ([warped_A0, warped_B0, warped_rec_A0],[ loss_DA ],
                                        opt(lr=2e-5).get_updates(loss_DA, self.DA.trainable_weights) )

            ############

            loss_DB = ( DLoss(warped_rec_B0_d_ones, warped_rec_B0_d ) + \
                      ( DLoss(fake_B0_d_zeros, fake_B0_d) + DLoss(rec_A0_B0_d_zeros, rec_A0_B0_d) ) * 0.5  ) * 0.5

            self.DB_train = K.function ([warped_A0, warped_B0, warped_rec_B0],[ loss_DB ],
                                        opt(lr=2e-5).get_updates(loss_DB, self.DB.trainable_weights) )
            """
            ############

            t = SampleProcessor.Types

            output_sample_types=[ {'types': (t.IMG_WARPED_TRANSFORMED, t.FACE_TYPE_FULL_NO_ROTATION, t.MODE_BGR), 'resolution':64},
                                  {'types': (t.IMG_SOURCE, t.FACE_TYPE_FULL_NO_ROTATION, t.MODE_BGR), 'resolution':resolution},
                                  {'types': (t.IMG_SOURCE, t.NONE, t.MODE_BGR), 'resolution':resolution},
                                  {'types': (t.IMG_SOURCE, t.NONE, t.MODE_M), 'resolution':128}
                                ]

            self.set_training_data_generators ([
                    SampleGeneratorFace(self.training_data_src_path, debug=self.is_debug(), batch_size=self.batch_size,
                        sample_process_options=SampleProcessor.Options(random_flip=False, rotation_range=[0,0]),
                        output_sample_types=output_sample_types ),

                    SampleGeneratorFace(self.training_data_dst_path, debug=self.is_debug(), batch_size=self.batch_size,
                        sample_process_options=SampleProcessor.Options(random_flip=False, rotation_range=[0,0]),
                        output_sample_types=output_sample_types )
                   ])
        else:
            self.G_convert = K.function([warped_A0 ],[rec_A0_B0])

    #override
    def onSave(self):
        self.save_weights_safe( [
                                 [self.enc,  'enc.h5'],
                                 [self.decA, 'decA.h5'],
                                 [self.decB, 'decB.h5'],
                                 [self.DA,    'DA.h5'],
                                 [self.DB,    'DB.h5'],
                                 ])

    #override
    def onTrainOneIter(self, generators_samples, generators_list):
        warped_src, _, src, srcm, = generators_samples[0]
        warped_dst, _, dst, dstm, = generators_samples[1]

        loss_A, = self.A_train ( [warped_src, src, srcm, warped_dst, dst, dstm] )
        loss_B, = self.B_train ( [warped_src, src, srcm, warped_dst, dst, dstm] )
        loss_DA, = 0,#self.DA_train ( [warped_src, warped_dst, src] )
        loss_DB, = 0,#self.DB_train ( [warped_src, warped_dst, dst] )

        return ( ('A', loss_A), ('B', loss_B), )# ('DA', loss_DA), ('DB', loss_DB) )

    #override
    def onGetPreview(self, sample):
        test_A0   = sample[0][0][0:4]
        test_A0f  = sample[0][1][0:4]
        test_A0r  = sample[0][2][0:4]

        test_B0  = sample[1][0][0:4]
        test_B0f  = sample[1][1][0:4]
        test_B0r  = sample[1][2][0:4]

        G_view_result = self.G_view([test_A0, test_B0 ])

        #test_A0f, test_A0r, test_B0f, test_B0r, rec_A0, rec_B0, rec_A0_B0 = [ x[0] / 2 + 0.5 for x in ([test_A0f, test_A0r, test_B0f, test_B0r] + G_view_result)  ]
        test_A0f, test_A0r, test_B0f, test_B0r, rec_A0, rec_B0, rec_A0_B0 = [ x[0] for x in ([test_A0f, test_A0r, test_B0f, test_B0r] + G_view_result)  ]
        #r = np.concatenate ((np.concatenate ( (test_A0f, test_A0r), axis=1),
        #                     np.concatenate ( (test_B0, rec_B0), axis=1)
        #                     ), axis=0)
        r = np.concatenate ( (test_B0f, rec_B0, test_A0f, rec_A0, rec_A0_B0), axis=1 )

        return [ ('AVATAR', r ) ]

    def predictor_func (self, inp_face_bgr):
        feed = [ inp_face_bgr[np.newaxis,...]*2-1 ]
        x = self.G_convert (feed)[0]
        return np.clip ( x[0]/2+0.5, 0, 1)

    # #override
    # def get_converter(self, **in_options):
    #     from models import ConverterImage
    #     return ConverterImage(self.predictor_func,
    #                           predictor_input_size=self.options['resolution'],
    #                           **in_options)
    #override
    def get_converter(self):
        base_erode_mask_modifier = 30
        base_blur_mask_modifier = 0

        default_erode_mask_modifier = 0
        default_blur_mask_modifier = 0

        face_type = FaceType.FULL

        from converters import ConverterAvatar
        return ConverterAvatar(self.predictor_func,
                               predictor_input_size=64)


    @staticmethod
    def PatchDiscriminator(ndf=64):
        exec (nnlib.import_all(), locals(), globals())

        #use_bias = True
        #def XNormalization(x):
        #    return InstanceNormalization (axis=-1)(x)
        use_bias = False
        def XNormalization(x):
            return BatchNormalization (axis=-1)(x)

        XConv2D = partial(Conv2D, use_bias=use_bias)

        def func(input):
            b,h,w,c = K.int_shape(input)

            x = input

            x = ZeroPadding2D((1,1))(x)
            x = XConv2D( ndf, 4, strides=2, padding='valid', use_bias=True)(x)
            x = LeakyReLU(0.2)(x)

            x = ZeroPadding2D((1,1))(x)
            x = XConv2D( ndf*2, 4, strides=2, padding='valid')(x)
            x = XNormalization(x)
            x = LeakyReLU(0.2)(x)

            x = ZeroPadding2D((1,1))(x)
            x = XConv2D( ndf*4, 4, strides=2, padding='valid')(x)
            x = XNormalization(x)
            x = LeakyReLU(0.2)(x)

            x = ZeroPadding2D((1,1))(x)
            x = XConv2D( ndf*8, 4, strides=2, padding='valid')(x)
            x = XNormalization(x)
            x = LeakyReLU(0.2)(x)

            x = ZeroPadding2D((1,1))(x)
            x = XConv2D( ndf*8, 4, strides=2, padding='valid')(x)
            x = XNormalization(x)
            x = LeakyReLU(0.2)(x)

            x = ZeroPadding2D((1,1))(x)
            return XConv2D( 1, 4, strides=1, padding='valid', use_bias=True, activation='sigmoid')(x)#
        return func

    @staticmethod
    def DFEncFlow(padding='zero', **kwargs):
        exec (nnlib.import_all(), locals(), globals())

        #use_bias = True
        #def XNormalization(x):
        #    return x#BatchNormalization (axis=-1)(x)
        #XConv2D = partial(Conv2D, padding=padding, use_bias=use_bias)

        #def Act(lrelu_alpha=0.1):
        #    return LeakyReLU(alpha=lrelu_alpha)

        #def downscale (dim, **kwargs):
        #    def func(x):
        #        return Act() ( XNormalization(XConv2D(dim, kernel_size=5, strides=2)(x)) )
        #    return func

        #downscale = partial(downscale)
        def downscale (dim):
            def func(x):
                return LeakyReLU(0.1)(Conv2D(dim, 5, strides=2, padding='same')(x))
            return func

        def upscale (dim):
            def func(x):
                return PixelShuffler()(LeakyReLU(0.1)(Conv2D(dim * 4, 3, strides=1, padding='same')(x)))
            return func
                
        def func(input):
            x, = input
            b,h,w,c = K.int_shape(x)
            
            x = downscale(128)(x)
            x = downscale(256)(x)
            x = downscale(512)(x)
            x = downscale(768)(x)
#
            x = Dense(256)(Flatten()(x))
            x = Dense(4 * 4 * 512)(x)
            x = Reshape((4, 4, 512))(x)
            x = upscale(512)(x)
            return x

            x = Conv2D(64, kernel_size=5, strides=1, padding='same')(x)
            x = Conv2D(64, kernel_size=5, strides=1, padding='same')(x)
            x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)

            x = Conv2D(128, kernel_size=3, strides=1, padding='same')(x)
            x = Conv2D(128, kernel_size=3, strides=1, padding='same')(x)
            x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)

            x = Conv2D(256, kernel_size=3, strides=1, padding='same')(x)
            x = Conv2D(256, kernel_size=3, strides=1, padding='same')(x)
            x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)

            x = Conv2D(512, kernel_size=3, strides=1, padding='same')(x)
            x = Conv2D(512, kernel_size=3, strides=1, padding='same')(x)
            x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)

            x = Flatten()(x)

            x = Dense(128)(x)
            return x

            x = Dense(256)(x)
            x = ReLU()(x)
            x = Dense(256)(x)
            x = ReLU()(x)

            mean = Dense(128)(x)
            logvar = Dense(128)(x)

            return mean, logvar

        return func

    @staticmethod
    def DFDecFlow(output_nc, padding='zero', **kwargs):
        exec (nnlib.import_all(), locals(), globals())

        use_bias = False
        def XNormalization(x):
            return BatchNormalization (axis=-1)(x)
        XConv2D = partial(Conv2D, padding=padding, use_bias=use_bias)
        XConv2DTranspose = partial(Conv2DTranspose, padding='same', use_bias=use_bias)

        def Act(act='', lrelu_alpha=0.1):
            if act == 'prelu':
                return PReLU()
            elif act == 'relu':
                return ReLU()
            else:
                return LeakyReLU(alpha=lrelu_alpha)

        def upscale (dim, **kwargs):
            def func(x):
                return Act('relu')( XNormalization(XConv2DTranspose(dim, kernel_size=3, strides=2)(x)))
            return func

        def to_bgr (output_nc, **kwargs):
            def func(x):
                return XConv2D(output_nc, kernel_size=5, use_bias=True, activation='sigmoid')(x)
            return func

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
                x = Act(lrelu_alpha=0.2)(x)
                x = Conv2D(self.filters, kernel_size=self.kernel_size, padding=self.padding)(x)
                x = Add()([x, inp])
                x = Act(lrelu_alpha=0.2)(x)
                return x

        upscale = partial(upscale)
        to_bgr = partial(to_bgr)

        dims = 64

        def func(input):
            x = input[0]
            x = upscale(512)( x )
            x = ResidualBlock(512)(x)
            x = ResidualBlock(512)(x)
            
            x = upscale(256)( x )
            x = ResidualBlock(256)(x)
            x = ResidualBlock(256)(x)

            x = upscale(128)( x )
            x = ResidualBlock(128)(x)
            x = ResidualBlock(128)(x)

            x = upscale(64)( x )
            x = ResidualBlock(64)(x)
            x = ResidualBlock(64)(x)

            return to_bgr(output_nc) ( x )
            
        #def func(input):
        #    x = input[0]
        #    x = Dense(8 * 8 * dims*8, activation="relu")(x)
        #    x = Reshape((8, 8, dims*8))(x)
#
        #    x = upscale(dims*8)( x )
        #    x = ResidualBlock(dims*8)(x)
        #    x = ResidualBlock(dims*8)(x)
#
        #    x = upscale(dims*4)( x )
        #    x = ResidualBlock(dims*4)(x)
        #    x = ResidualBlock(dims*4)(x)
#
        #    x = upscale(dims*2)( x )
        #    x = ResidualBlock(dims*2)(x)
        #    x = ResidualBlock(dims*2)(x)
#
        #    x = upscale(dims)( x )
        #    x = ResidualBlock(dims)(x)
        #    x = ResidualBlock(dims)(x)
#
        #    #x = upscale(dims*4)( x )
        #    #x = ResidualBlock(dims*4)(x)
        #    #x = ResidualBlock(dims*4)(x)
#
        #    return to_bgr(output_nc) ( x )
            
        return func

Model = AVATARModel
