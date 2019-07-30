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
        bgr_64_mask_shape = (64,64,1)
        out_bgr_shape = (resolution, resolution, 3)
        bgr_t_shape = (resolution, resolution, 9)
        mask_shape = (resolution, resolution, 1)
        bgrm_shape = (resolution, resolution, 4)

        ngf = 64
        ndf = 64

        use_batch_norm = True #created_batch_size > 1

        self.enc = modelify(AVATARModel.DFEncFlow ())( [Input(in_bgr_shape),] )

        dec_Inputs = [ Input(K.int_shape( self.enc.outputs[0] )[1:]) ]
        self.decA64 = modelify(AVATARModel.DFDec64Flow (out_bgr_shape[2])) (dec_Inputs)
        self.decB64 = modelify(AVATARModel.DFDec64Flow (out_bgr_shape[2])) (dec_Inputs)

        self.decA = modelify(AVATARModel.DFDecFlow (out_bgr_shape[2])) (dec_Inputs)
        self.decB = modelify(AVATARModel.DFDecFlow (out_bgr_shape[2])) (dec_Inputs)
        
        
        self.C = modelify(AVATARModel.ResNet (9, use_batch_norm=False, n_blocks=6, ngf=128, use_dropout=True))(Input(bgr_t_shape))
        
        if not self.is_first_run():
            weights_to_load = [
                (self.enc, 'enc.h5'),
                (self.decA, 'decA.h5'),
                (self.decB, 'decB.h5'),
                (self.decA64, 'decA64.h5'),
                (self.decB64, 'decB64.h5'),
                (self.C, 'C.h5')
            ]
            self.load_weights_safe(weights_to_load)

        warped_A064 = Input(in_bgr_shape)
        real_A064 = Input(in_bgr_shape)
        real_A064m = Input(bgr_64_mask_shape)
        real_A0 = Input(out_bgr_shape)
        real_A0m = Input(mask_shape)
        
        real_B64_t0 = Input(in_bgr_shape)
        real_B64_t1 = Input(in_bgr_shape)
        real_B64_t2 = Input(in_bgr_shape)        
        
        real_A_t0 = Input(out_bgr_shape)
        real_Am_t0 = Input(mask_shape)
        real_A_t1 = Input(out_bgr_shape)
        real_Am_t1 = Input(mask_shape)
        real_A_t2 = Input(out_bgr_shape)
        real_Am_t2 = Input(mask_shape)
        
        warped_B064 = Input(in_bgr_shape)
        real_B064 = Input(in_bgr_shape)
        real_B064m = Input(bgr_64_mask_shape)
        real_B0 = Input(out_bgr_shape)
        real_B0m = Input(mask_shape)
        
        #real_A0m_blurred = gaussian_blur( max(1, K.int_shape(x)[1] // 32) )(real_A0m)


        warped_A0_code = self.enc (warped_A064)
        warped_B0_code = self.enc (warped_B064)
        
        
        rec_A064 = self.decA64 (warped_A0_code)
        rec_B064 = self.decB64 (warped_B0_code)
        rec_A0B064 = self.decA64 (warped_B0_code)

        rec_A0 = self.decA (warped_A0_code)
        rec_B0 = self.decB (warped_B0_code)

        rec_AB_t0 = self.decA (self.enc (real_B64_t0))
        rec_AB_t1 = self.decA (self.enc (real_B64_t1))
        rec_AB_t2 = self.decA (self.enc (real_B64_t2))

        x = self.C ( K.concatenate ( [real_A_t0*real_Am_t0 + (1-real_Am_t0)*0.5,
                                      real_A_t1*real_Am_t1 + (1-real_Am_t1)*0.5,
                                      real_A_t2*real_Am_t2 + (1-real_Am_t2)*0.5
                                     ] , axis=-1) )
        rec_C_A_t0 = Lambda ( lambda x: x[...,0:3], output_shape= ( K.int_shape(x)[1:3], 3 ) ) (x)
        rec_C_A_t1 = Lambda ( lambda x: x[...,3:6], output_shape= ( K.int_shape(x)[1:3], 3 ) ) (x)
        rec_C_A_t2 = Lambda ( lambda x: x[...,6:9], output_shape= ( K.int_shape(x)[1:3], 3 ) ) (x)

        x = self.C ( K.concatenate ( [rec_AB_t0, rec_AB_t1, rec_AB_t2] , axis=-1) )
        rec_C_AB_t0 = Lambda ( lambda x: x[...,0:3], output_shape= ( K.int_shape(x)[1:3], 3 ) ) (x)
        rec_C_AB_t1 = Lambda ( lambda x: x[...,3:6], output_shape= ( K.int_shape(x)[1:3], 3 ) ) (x)
        rec_C_AB_t2 = Lambda ( lambda x: x[...,6:9], output_shape= ( K.int_shape(x)[1:3], 3 ) ) (x)

 
        self.G64_view = K.function([warped_A064, warped_B064],[rec_A064, rec_B064, rec_A0B064])
        self.G_view = K.function([warped_A064, warped_B064, real_B64_t0, real_B64_t1, real_B64_t2], [rec_A0, rec_B0, rec_C_AB_t0, rec_C_AB_t1, rec_C_AB_t2])

        if self.is_training_mode:
            loss_AB64 = K.mean( 10 * dssim(kernel_size=5,max_value=1.0) ( real_A064*real_A064m, rec_A064*real_A064m ) ) + \
                        K.mean( 10 * dssim(kernel_size=5,max_value=1.0) ( real_B064*real_B064m, rec_B064*real_B064m ) )
            
            #loss_AB64 = 50 * K.mean(K.abs (real_A064-rec_A064) ) + \
            #            50 * K.mean(K.abs (real_B064-rec_B064) )
            
            weights_AB64 = self.enc.trainable_weights + self.decA64.trainable_weights + self.decB64.trainable_weights


            loss_AB = K.mean( 10 * dssim(kernel_size=int(resolution/11.6),max_value=1.0) ( rec_A0, real_A0*real_A0m + (1-real_A0m)*0.5 ) ) + \
                      K.mean( 10 * dssim(kernel_size=int(resolution/11.6),max_value=1.0) ( rec_B0, real_B0*real_B0m + (1-real_B0m)*0.5) )
            weights_AB = self.enc.trainable_weights + self.decA.trainable_weights + self.decB.trainable_weights

            loss_C = K.mean( 10 * dssim(kernel_size=int(resolution/11.6),max_value=1.0) ( real_A_t0, rec_C_A_t0 ) ) + \
                     K.mean( 10 * dssim(kernel_size=int(resolution/11.6),max_value=1.0) ( real_A_t1, rec_C_A_t1 ) ) + \
                     K.mean( 10 * dssim(kernel_size=int(resolution/11.6),max_value=1.0) ( real_A_t2, rec_C_A_t2 ) ) 
                 
            weights_C = self.C.trainable_weights

            def opt(lr=5e-5):
                return Adam(lr=lr, beta_1=0.5, beta_2=0.999, tf_cpu_mode=2)#, clipnorm=1)

            self.AB64_train = K.function ([warped_A064, real_A064, real_A064m, warped_B064, real_B064, real_B064m], [loss_AB64], opt(lr=5e-5).get_updates(loss_AB64, weights_AB64) )
            self.AB_train = K.function ([warped_A064, real_A0, real_A0m, warped_B064, real_B0, real_B0m],[ loss_AB ], opt(lr=5e-5).get_updates(loss_AB, weights_AB) )
            self.C_train = K.function ([ real_A_t0, real_Am_t0, real_A_t1, real_Am_t1, real_A_t2, real_Am_t2 ],[ loss_C ], opt(lr=5e-5).get_updates(loss_C, weights_C) )
            ###########

            t = SampleProcessor.Types

            output_sample_types=[ {'types': (t.IMG_WARPED_TRANSFORMED, t.FACE_TYPE_FULL_NO_ROTATION, t.MODE_BGR), 'resolution':64},
                                  {'types': (t.IMG_SOURCE, t.FACE_TYPE_FULL_NO_ROTATION, t.MODE_BGR), 'resolution':resolution},
                                  {'types': (t.IMG_SOURCE, t.NONE, t.MODE_BGR), 'resolution':resolution},
                                  {'types': (t.IMG_SOURCE, t.NONE, t.MODE_M), 'resolution':resolution}
                                ]

            self.set_training_data_generators ([
                    SampleGeneratorFace(self.training_data_src_path, debug=self.is_debug(), batch_size=self.batch_size,
                        sample_process_options=SampleProcessor.Options(random_flip=False),
                        output_sample_types=[ {'types': (t.IMG_WARPED_TRANSFORMED, t.FACE_TYPE_FULL_NO_ROTATION, t.MODE_BGR), 'resolution':64},
                                              {'types': (t.IMG_TRANSFORMED, t.FACE_TYPE_FULL_NO_ROTATION, t.MODE_BGR), 'resolution':64},
                                              {'types': (t.IMG_TRANSFORMED, t.FACE_TYPE_FULL_NO_ROTATION, t.MODE_M), 'resolution':64}
                                            ] ),
                    SampleGeneratorFace(self.training_data_dst_path, debug=self.is_debug(), batch_size=self.batch_size,
                        sample_process_options=SampleProcessor.Options(random_flip=False),
                        output_sample_types=[ {'types': (t.IMG_WARPED_TRANSFORMED, t.FACE_TYPE_FULL_NO_ROTATION, t.MODE_BGR), 'resolution':64},
                                              {'types': (t.IMG_TRANSFORMED, t.FACE_TYPE_FULL_NO_ROTATION, t.MODE_BGR), 'resolution':64},
                                              {'types': (t.IMG_TRANSFORMED, t.FACE_TYPE_FULL_NO_ROTATION, t.MODE_M), 'resolution':64}
                                            ] ),
                                           
                    SampleGeneratorFace(self.training_data_src_path, debug=self.is_debug(), batch_size=self.batch_size,
                        sample_process_options=SampleProcessor.Options(random_flip=False, rotation_range=[0,0]),
                        output_sample_types=output_sample_types ),

                    SampleGeneratorFace(self.training_data_dst_path, debug=self.is_debug(), batch_size=self.batch_size,
                        sample_process_options=SampleProcessor.Options(random_flip=False, rotation_range=[0,0]),
                        output_sample_types=output_sample_types ),
                        
                    SampleGeneratorFaceTemporal(self.training_data_src_path, debug=self.is_debug(), batch_size=self.batch_size, 
                        temporal_image_count=3,
                        sample_process_options=SampleProcessor.Options(random_flip=False), 
                        output_sample_types=[{'types': (t.IMG_SOURCE, t.NONE, t.MODE_BGR), 'resolution':resolution},
                                             {'types': (t.IMG_SOURCE, t.NONE, t.MODE_M), 'resolution':resolution}
                                            ] ),
                       
                    SampleGeneratorFaceTemporal(self.training_data_dst_path, debug=self.is_debug(), batch_size=self.batch_size, 
                        temporal_image_count=3,
                        sample_process_options=SampleProcessor.Options(random_flip=False), 
                        output_sample_types=[{'types': (t.IMG_SOURCE, t.FACE_TYPE_FULL_NO_ROTATION, t.MODE_BGR), 'resolution':64},
                                             {'types': (t.IMG_SOURCE, t.FACE_TYPE_FULL_NO_ROTATION, t.MODE_BGR), 'resolution':resolution},
                                            ] ), 
                                            
                   ])
        else:
            self.G_convert = K.function([warped_B064],[rec_C_A0_B0])

    #override
    def onSave(self):
        self.save_weights_safe( [
                                 [self.enc,  'enc.h5'],
                                 [self.decA, 'decA.h5'],
                                 [self.decB, 'decB.h5'],
                                 [self.decA64, 'decA64.h5'],
                                 [self.decB64, 'decB64.h5'],
                                 [self.C,     'C.h5']
                                 ])

    #override
    def onTrainOneIter(self, generators_samples, generators_list):
        warped_src64, src64, src64m = generators_samples[0]
        warped_dst64, dst64, dst64m = generators_samples[1]
        
        warped_src64, _, src, srcm, = generators_samples[2]
        warped_dst64, _, dst, dstm, = generators_samples[3]        
        t_src_0, t_srcm_0, t_src_1, t_srcm_1, t_src_2, t_srcm_2, = generators_samples[4]
        

        loss_AB64, = 0,#= self.AB64_train ( [warped_src64, src64, src64m, warped_dst64, dst64, dst64m] )
        loss_AB, = self.AB_train ( [warped_src64, src, srcm, warped_dst64, dst, dstm] )
        loss_C, = self.C_train ( [ t_src_0, t_srcm_0, t_src_1, t_srcm_1, t_src_2, t_srcm_2 ] )

        return ( ('AB64', loss_AB64), ('AB', loss_AB), ('C', loss_C) ) # ('DA', loss_DA), ('DB', loss_DB) )

    #override
    def onGetPreview(self, sample):
        test_A064w  = sample[0][0][0:4]
        test_A064r  = sample[0][1][0:4]
        test_A064m  = sample[0][2][0:4]
        
        test_B064w  = sample[1][0][0:4]
        test_B064r  = sample[1][1][0:4]
        test_B064m  = sample[1][2][0:4]
        
        test_A064   = sample[2][0][0:4]
        test_A0f  = sample[2][1][0:4]
        test_A0r  = sample[2][2][0:4]

        test_B064  = sample[3][0][0:4]
        test_B0f  = sample[3][1][0:4]
        test_B0r  = sample[3][2][0:4]
        test_B0m  = sample[3][3][0:4]
        
        t_dst64_0 = sample[5][0][0:4]
        t_dst_0   = sample[5][1][0:4]
        t_dst64_1 = sample[5][2][0:4]
        t_dst_1   = sample[5][3][0:4]
        t_dst64_2 = sample[5][4][0:4]
        t_dst_2   = sample[5][5][0:4]
        
        G64_view_result = self.G64_view ([test_A064r, test_B064r])
        test_A064r, test_B064r, rec_A064, rec_B064, rec_AB64 = [ x[0] for x in ([test_A064r, test_B064r] + G64_view_result)  ]
        
        sample64x4 = np.concatenate ([ np.concatenate ( [rec_B064, rec_A064], axis=1 ),
                                       np.concatenate ( [test_B064r, rec_AB64], axis=1) ], axis=0 )
                                       
        #todo sample64x4 = cv2.resize (sample64x4, (self.resolution, self.resolution) )

        G_view_result = self.G_view([test_A064, test_B064, t_dst64_0, t_dst64_1, t_dst64_2 ])

        test_A0f, test_A0r, test_B0f, test_B0r, t_dst_0, t_dst_1, t_dst_2, rec_A0, rec_B0, rec_C_AB_t0, rec_C_AB_t1, rec_C_AB_t2 = [ x[0] for x in ([test_A0f, test_A0r, test_B0f, test_B0r, t_dst_0, t_dst_1, t_dst_2, ] + G_view_result)  ]

        #r = sample64x4
        r = np.concatenate ( (sample64x4, test_B0f, rec_B0, test_A0f, rec_A0, t_dst_0, t_dst_1, t_dst_2, rec_C_AB_t0, rec_C_AB_t1, rec_C_AB_t2 ), axis=1 )

        return [ ('AVATAR', r ) ]

    def predictor_func (self, inp_face_bgr):
        feed = [ inp_face_bgr[np.newaxis,...] ]
        x = self.G_convert (feed)[0]
        return np.clip ( x[0], 0, 1)

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

        use_bias = False
        def XNorm(x):
            return BatchNormalization (axis=-1)(x)
        XConv2D = partial(Conv2D, padding=padding, use_bias=use_bias)

        #def Act(lrelu_alpha=0.1):
        #    return LeakyReLU(alpha=lrelu_alpha)

        #def downscale (dim, **kwargs):
        #    def func(x):
        #        return Act() ( XNormalization(XConv2D(dim, kernel_size=5, strides=2)(x)) )
        #    return func

        #downscale = partial(downscale)
        def downscale (dim):
            def func(x):
                return  LeakyReLU(0.1)( Conv2D(dim, 5, strides=2, padding='same')(x))#BlurPool(filt_size=5)(
            return func

        def upscale (dim):
            def func(x):
                return SubpixelUpscaler()(LeakyReLU(0.1)(Conv2D(dim * 4, 3, strides=1, padding='same')(x)))
            return func
        """
        def self_attn_block(inp, nc, squeeze_factor=8):
            assert nc//squeeze_factor > 0, f"Input channels must be >= {squeeze_factor}, recieved nc={nc}"
            x = inp
            shape_x = x.get_shape().as_list()
            
            f = Conv2D(nc//squeeze_factor, 1, kernel_regularizer=keras.regularizers.l2(1e-4))(x)
            g = Conv2D(nc//squeeze_factor, 1, kernel_regularizer=keras.regularizers.l2(1e-4))(x)
            h = Conv2D(nc, 1, kernel_regularizer=keras.regularizers.l2(1e-4))(x)
            
            shape_f = f.get_shape().as_list()
            shape_g = g.get_shape().as_list()
            shape_h = h.get_shape().as_list()
            flat_f = Reshape( (-1, shape_f[-1]) )(f)
            flat_g = Reshape( (-1, shape_g[-1]) )(g)
            flat_h = Reshape( (-1, shape_h[-1]) )(h)   

            s = Lambda(lambda x: K.batch_dot(x[0], keras.layers.Permute((2,1))(x[1]) ))([flat_g, flat_f])
            beta = keras.layers.Softmax(axis=-1)(s)
            o = Lambda(lambda x: K.batch_dot(x[0], x[1]))([beta, flat_h])
            
            o = Reshape(shape_x[1:])(o)
            o = Scale()(o)
            
            out = Add()([o, inp])
            return out     
        """
               
        def func(input):
            x, = input
            b,h,w,c = K.int_shape(x)
            
            x = downscale(128)(x)
            x = downscale(256)(x)
            x = downscale(512)(x)
            x = downscale(1024)(x)

            x = Dense(256)(Flatten()(x))
            x = Dense(4 * 4 * 512)(x)
            x = Reshape((4, 4, 512))(x) 
            x = upscale(512)(x)   
            return x
            
            x, = input
            b,h,w,c = K.int_shape(x)
            
            x = downscale(128)(x)
            x = downscale(256)(x)
            x = downscale(512)(x)
            x = downscale(1024)(x)
#
            x = Dense(256)(Flatten()(x))
            x = Dense(4 * 4 * 256)(x)
            x = Reshape((4, 4, 256))(x)
            x = upscale(256)(x)
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
    def DFDec64Flow(output_nc, padding='zero', **kwargs):
        exec (nnlib.import_all(), locals(), globals())

        def upscale (dim, padding='zero', norm='', act='', **kwargs):
            def func(x):
                return SubpixelUpscaler()(LeakyReLU(alpha=0.1)(Conv2D(dim * 4, kernel_size=3, strides=1, padding=padding)(x)))
            return func
        def to_bgr (output_nc, **kwargs):
            def func(x):
                return Conv2D(output_nc, kernel_size=5, strides=1, padding='same', activation='sigmoid')(x)
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
                x = LeakyReLU(alpha=0.2)(x)
                x = Conv2D(self.filters, kernel_size=self.kernel_size, padding=self.padding)(x)
                x = Add()([x, inp])
                x = LeakyReLU(alpha=0.2)(x)
                return x
        upscale = partial(upscale)
        to_bgr = partial(to_bgr)

        def func(input):
            x = input[0]                     
                       
            x = upscale(256)( x )
            x = ResidualBlock(256)(x)

            x = upscale(128)( x )
            x = ResidualBlock(128)(x)

            x = upscale(64)( x )
            x = ResidualBlock(64)(x)
            return to_bgr(output_nc) ( x )
            
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

        def upscale (dim, padding='zero', norm='', act='', **kwargs):
            def func(x):
                return SubpixelUpscaler()(LeakyReLU(alpha=0.1)(Conv2D(dim * 4, kernel_size=3, strides=1, padding=padding)(x)))
            return func

        def to_bgr (output_nc, **kwargs):
            def func(x):
                return Conv2D(output_nc, kernel_size=5, padding='same', use_bias=True, activation='sigmoid')(x)
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
                x = LeakyReLU(alpha=0.2)(x)
                x = Conv2D(self.filters, kernel_size=self.kernel_size, padding=self.padding)(x)
                x = Add()([x, inp])
                x = LeakyReLU(alpha=0.2)(x)
                return x

        upscale = partial(upscale)
        to_bgr = partial(to_bgr)

        dims = 64

        def func(input):
            x = input[0]            
            x = upscale(512)( x )
            x = ResidualBlock(512)(x)
            
            x = upscale(256)( x )
            x = ResidualBlock(256)(x)

            x = upscale(128)( x )
            x = ResidualBlock(128)(x)

            x = upscale(64)( x )
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
        
    @staticmethod
    def ResNet(output_nc, use_batch_norm, ngf=64, n_blocks=6, use_dropout=False):
        exec (nnlib.import_all(), locals(), globals())

        if not use_batch_norm:
            use_bias = True
            def XNormalization(x):
                return InstanceNormalization (axis=-1)(x)
        else:
            use_bias = False
            def XNormalization(x):
                return BatchNormalization (axis=-1)(x)
                
        XConv2D = partial(Conv2D, padding='same', use_bias=use_bias)
        XConv2DTranspose = partial(Conv2DTranspose, padding='same', use_bias=use_bias)

        def func(input):


            def ResnetBlock(dim, use_dropout=False):
                def func(input):
                    x = input

                    x = XConv2D(dim, 3, strides=1)(x)
                    x = XNormalization(x)
                    x = ReLU()(x)

                    if use_dropout:
                        x = Dropout(0.5)(x)

                    x = XConv2D(dim, 3, strides=1)(x)
                    x = XNormalization(x)
                    x = ReLU()(x)
                    return Add()([x,input])
                return func

            x = input

            x = ReLU()(XNormalization(XConv2D(ngf, 7, strides=1)(x)))

            x = ReLU()(XNormalization(XConv2D(ngf*2, 3, strides=2)(x)))
            x = ReLU()(XNormalization(XConv2D(ngf*4, 3, strides=2)(x)))

            for i in range(n_blocks):
                x = ResnetBlock(ngf*4, use_dropout=use_dropout)(x)

            x = ReLU()(XNormalization(XConv2DTranspose(ngf*2, 3, strides=2)(x)))
            x = ReLU()(XNormalization(XConv2DTranspose(ngf  , 3, strides=2)(x)))

            x = XConv2D(output_nc, 7, strides=1, activation='sigmoid', use_bias=True)(x)

            return x

        return func
Model = AVATARModel

""" 
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
"""