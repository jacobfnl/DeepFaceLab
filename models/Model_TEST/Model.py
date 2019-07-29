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
        
        self.decA64 = modelify(AVATARModel.DFDec64Flow (out_bgr_shape[2])) (dec_Inputs)
        self.decB64 = modelify(AVATARModel.DFDec64Flow (out_bgr_shape[2])) (dec_Inputs)
        
        self.C = modelify(AVATARModel.ResNet (out_bgr_shape[2], use_batch_norm=False, n_blocks=6, ngf=128, use_dropout=True))(Input(out_bgr_shape))
        

        #self.DA = modelify(AVATARModel.PatchDiscriminator(ndf=ndf) ) (Input(out_bgr_shape))
        #self.DB = modelify(AVATARModel.PatchDiscriminator(ndf=ndf) ) (Input(out_bgr_shape))

        if not self.is_first_run():
            weights_to_load = [
                (self.enc, 'enc.h5'),
                (self.decA, 'decA.h5'),
                (self.decB, 'decB.h5'),
                (self.decA64, 'decA64.h5'),
                (self.decB64, 'decB64.h5'),
                #(self.DA, 'DA.h5'),
                #(self.DB, 'DB.h5'),
                (self.C, 'C.h5')
            ]
            self.load_weights_safe(weights_to_load)

        warped_A064 = Input(in_bgr_shape)
        real_A064 = Input(in_bgr_shape)
        real_A0 = Input(out_bgr_shape)
        real_A0m = Input(mask_shape)
        
        real_A_t0 = Input(out_bgr_shape)
        real_Am_t0 = Input(mask_shape)
        real_A_t1 = Input(out_bgr_shape)
        real_Am_t1 = Input(mask_shape)
        real_A_t2 = Input(out_bgr_shape)
        real_Am_t2 = Input(mask_shape)
        
        warped_B064 = Input(in_bgr_shape)
        warped_B0 = Input(in_bgr_shape)
        real_B064 = Input(in_bgr_shape)
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
        rec_A0_B0 = self.decA ( self.enc (warped_B0) )

        #real_A0_d = self.DA(real_A0)
        #real_A0_d_ones = K.ones_like(real_A0_d)

        #rec_A0_d = self.DA(rec_A0)
        #rec_A0_d_ones = K.ones_like(rec_A0_d)
        #rec_A0_d_zeros = K.zeros_like(rec_A0_d)

        #rec_A0_B0_d = self.DA(rec_A0_B0)
        #rec_A0_B0_d_ones = K.ones_like(rec_A0_B0_d)
        #rec_A0_B0_d_zeros = K.zeros_like(rec_A0_B0_d)
        #
        #real_B0_d = self.DB(real_B0)
        #real_B0_d_ones = K.ones_like(real_B0_d)

        #rec_B0_d = self.DB(rec_B0)
        #rec_B0_d_ones = K.ones_like(rec_B0_d)
        #rec_B0_d_zeros = K.zeros_like(rec_B0_d)

        rec_C_A0 = self.C ( real_A0*real_A0m + (1-real_A0m)*0.5 )

        #import code
        #code.interact(local=dict(globals(), **locals()))

        #x = self.C ( K.concatenate ( [real_A_t0*real_Am_t0 + (1-real_Am_t0)*0.5,
        #                              real_A_t1*real_Am_t1 + (1-real_Am_t1)*0.5,
        #                              real_A_t2*real_Am_t2 + (1-real_Am_t2)*0.5
        #                             ] , axis=-1) )
        #rec_C_A_t0 = Lambda ( lambda x: x[...,0:3], output_shape= ( K.int_shape(x)[1:3], 3 ) ) ()
        #rec_C_A_t1 = Lambda ( lambda x: x[...,3:6], output_shape= ( K.int_shape(x)[1:3], 3 ) ) ()
        #rec_C_A_t2 = Lambda ( lambda x: x[...,6:9], output_shape= ( K.int_shape(x)[1:3], 3 ) ) ()
        #
        #loss_C = K.mean( 10 * dssim(kernel_size=int(resolution/11.6),max_value=1.0) ( real_A_t0, rec_C_A_t0 ) ) + \
        #         K.mean( 10 * dssim(kernel_size=int(resolution/11.6),max_value=1.0) ( real_A_t1, rec_C_A_t1 ) ) + \
        #         K.mean( 10 * dssim(kernel_size=int(resolution/11.6),max_value=1.0) ( real_A_t2, rec_C_A_t2 ) ) 
        
        
        rec_C_A0_B0 = self.C (rec_A0_B0)
        
        self.G64_view = K.function([warped_A064, warped_B064],[rec_A064, rec_B064, rec_A0B064])
        self.G_view = K.function([warped_A064, warped_B064, warped_B0],[rec_A0, rec_B0, rec_A0_B0, rec_C_A0_B0])

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

            loss_A64 = K.mean( 10 * dssim(kernel_size=5,max_value=1.0) ( real_A064, rec_A064 ) )
            loss_B64 = K.mean( 10 * dssim(kernel_size=5,max_value=1.0) ( real_B064, rec_B064 ) )
            
            weights_A64 = self.enc.trainable_weights + self.decA64.trainable_weights
            weights_B64 = self.enc.trainable_weights + self.decB64.trainable_weights
            
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
            

            loss_C = K.mean( 10 * dssim(kernel_size=int(resolution/11.6),max_value=1.0) ( real_A0, rec_C_A0 ) )

            weights_C = self.C.trainable_weights

            def opt(lr=2e-5):
                return Adam(lr=lr, beta_1=0.5, beta_2=0.999, tf_cpu_mode=2)#, clipnorm=1)


            self.A64_train = K.function ([warped_A064, real_A064], [loss_A64], opt(lr=2e-5).get_updates(loss_A64, weights_A64) )
            self.B64_train = K.function ([warped_B064, real_B064], [loss_B64], opt(lr=2e-5).get_updates(loss_B64, weights_B64) )
                                        
            self.A_train = K.function ([warped_A064, real_A0, real_A0m],[ loss_A ],
                                        opt(lr=2e-5).get_updates(loss_A, weights_A) )

            self.B_train = K.function ([warped_B064, real_B0, real_B0m],[ loss_B ],
                                        opt(lr=2e-5).get_updates(loss_B, weights_B) )

            self.C_train = K.function ([real_A0, real_A0m],[ loss_C ],
                                        opt(lr=2e-5).get_updates(loss_C, weights_C) )
            ###########
            """
            loss_DA = ( DLoss(real_A0_d_ones, real_A0_d ) + \
                        DLoss(rec_A0_B0_d_zeros,  rec_A0_B0_d ) ) * 0.5
                        #DLoss(fake_A0_d_zeros, fake_A0_d ) ) * 0.5

            self.DA_train = K.function ([warped_A064, warped_B064, warped_rec_A0],[ loss_DA ],
                                        opt(lr=2e-5).get_updates(loss_DA, self.DA.trainable_weights) )

            ############

            loss_DB = ( DLoss(warped_rec_B0_d_ones, warped_rec_B0_d ) + \
                      ( DLoss(fake_B0_d_zeros, fake_B0_d) + DLoss(rec_A0_B0_d_zeros, rec_A0_B0_d) ) * 0.5  ) * 0.5

            self.DB_train = K.function ([warped_A064, warped_B064, warped_rec_B0],[ loss_DB ],
                                        opt(lr=2e-5).get_updates(loss_DB, self.DB.trainable_weights) )
            """
            ############

            t = SampleProcessor.Types

            output_sample_types=[ {'types': (t.IMG_WARPED_TRANSFORMED, t.FACE_TYPE_FULL_NO_ROTATION, t.MODE_BGR), 'resolution':64},
                                  {'types': (t.IMG_SOURCE, t.FACE_TYPE_FULL_NO_ROTATION, t.MODE_BGR), 'resolution':64},
                                  {'types': (t.IMG_SOURCE, t.FACE_TYPE_FULL_NO_ROTATION, t.MODE_BGR), 'resolution':resolution},
                                  {'types': (t.IMG_SOURCE, t.NONE, t.MODE_BGR), 'resolution':resolution},
                                  {'types': (t.IMG_SOURCE, t.NONE, t.MODE_M), 'resolution':128}
                                ]

            self.set_training_data_generators ([
                    SampleGeneratorFace(self.training_data_src_path, debug=self.is_debug(), batch_size=self.batch_size,
                        sample_process_options=SampleProcessor.Options(rotation_range=[-25,25]),
                        output_sample_types=[ {'types': (t.IMG_WARPED_TRANSFORMED, t.FACE_TYPE_FULL, t.MODE_BGR), 'resolution':64},
                                              {'types': (t.IMG_TRANSFORMED, t.FACE_TYPE_FULL, t.MODE_BGR), 'resolution':64},
                                            ] ),
                    SampleGeneratorFace(self.training_data_dst_path, debug=self.is_debug(), batch_size=self.batch_size,
                        sample_process_options=SampleProcessor.Options(rotation_range=[-25,25]),
                        output_sample_types=[ {'types': (t.IMG_WARPED_TRANSFORMED, t.FACE_TYPE_FULL, t.MODE_BGR), 'resolution':64},
                                              {'types': (t.IMG_TRANSFORMED, t.FACE_TYPE_FULL, t.MODE_BGR), 'resolution':64},
                                            ] ),
                                            
                    SampleGeneratorFace(self.training_data_src_path, debug=self.is_debug(), batch_size=self.batch_size,
                        sample_process_options=SampleProcessor.Options(random_flip=False, rotation_range=[0,0]),
                        output_sample_types=output_sample_types ),
#
                    SampleGeneratorFace(self.training_data_dst_path, debug=self.is_debug(), batch_size=self.batch_size,
                        sample_process_options=SampleProcessor.Options(random_flip=False, rotation_range=[0,0]),
                        output_sample_types=output_sample_types ),
                        
                    #SampleGeneratorFaceTemporal(self.training_data_src_path, debug=self.is_debug(), batch_size=self.batch_size, 
                    #    temporal_image_count=3,
                    #    sample_process_options=SampleProcessor.Options(random_flip=False), 
                    #    output_sample_types=[{'types': (t.IMG_SOURCE, t.NONE, t.MODE_BGR), 'resolution':resolution},
                    #                         {'types': (t.IMG_SOURCE, t.NONE, t.MODE_M), 'resolution':128}
                    #                        ] ),
                        
                    #SampleGeneratorFaceTemporal(self.training_data_dst_path, debug=self.is_debug(), batch_size=self.batch_size, 
                    #    temporal_image_count=3,
                    #    sample_process_options=SampleProcessor.Options(random_flip=False), 
                    #    output_sample_types=[{'types': (t.IMG_SOURCE, t.NONE, t.MODE_BGR), 'resolution':resolution},
                    #                         {'types': (t.IMG_SOURCE, t.NONE, t.MODE_M), 'resolution':128}
                    #                        ] ), ),
                                            
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
                                 #[self.DA,    'DA.h5'],
                                 #[self.DB,    'DB.h5'],
                                 [self.C,     'C.h5']
                                 ])

    #override
    def onTrainOneIter(self, generators_samples, generators_list):
        warped_src64, src64 = generators_samples[0]
        warped_dst64, dst64 = generators_samples[1]
        
        warped_src64, _, _, src, srcm, = generators_samples[2]
        warped_dst64, _, _, dst, dstm, = generators_samples[3]
        
        #t_src_0, t_srcm_0, t_src_1, t_srcm_1, t_src_2, t_srcm_2, = generators_samples[4]
        
        
        loss_A64, = self.A64_train ( [warped_src64, src64] )
        loss_B64, = self.B64_train ( [warped_dst64, dst64] )
        
        loss_A, = self.A_train ( [warped_src64, src, srcm] )
        loss_B, = self.B_train ( [warped_dst64, dst, dstm] )
        loss_C, = self.C_train ( [src, srcm] )
        #loss_DA, = 0,#self.DA_train ( [warped_src64, warped_dst64, src] )
        #loss_DB, = 0,#self.DB_train ( [warped_src64, warped_dst64, dst] )

        return ( ('AB64', (loss_A64+loss_B64)/2 ), ('AB', (loss_A+loss_B) / 2), ('C', loss_C) ) # ('DA', loss_DA), ('DB', loss_DB) )

    #override
    def onGetPreview(self, sample):
        test_A064w  = sample[0][0][0:4]
        test_A064r  = sample[0][1][0:4]
        
        test_B064w  = sample[1][0][0:4]
        test_B064r  = sample[1][1][0:4]
        
        test_A064   = sample[2][0][0:4]
        test_A0   = sample[2][1][0:4]
        test_A0f  = sample[2][2][0:4]
        test_A0r  = sample[2][3][0:4]

        test_B064  = sample[3][0][0:4]
        test_B0   = sample[3][1][0:4]
        test_B0f  = sample[3][2][0:4]
        test_B0r  = sample[3][3][0:4]
        test_B0m  = sample[3][4][0:4]
        
        G64_view_result = self.G64_view ([test_A064w, test_B064w])
        test_A064r, test_B064r, rec_A064, rec_B064, rec_A0B064 = [ x[0] for x in ([test_A064r, test_B064r] + G64_view_result)  ]
        
        sample64x4 = np.concatenate ([ np.concatenate ( [rec_B064, rec_A064], axis=1 ),
                                       np.concatenate ( [test_B064r, rec_A0B064], axis=1) ], axis=0 )
                                       
        #todo sample64x4 = cv2.resize (sample64x4, (self.resolution, self.resolution) )
        
        G_view_result = self.G_view([test_A064, test_B064, test_B0 ])

        test_A0f, test_A0r, test_B0f, test_B0r, rec_A0, rec_B0, rec_A0_B0, rec_C_A0_B0 = [ x[0] for x in ([test_A0f, test_A0r, test_B0f, test_B0r] + G_view_result)  ]
        
        #import code
        #code.interact(local=dict(globals(), **locals()))

        #r = np.concatenate ((np.concatenate ( (test_A0f, test_A0r), axis=1),
        #                     np.concatenate ( (test_B0, rec_B0), axis=1)
        #                     ), axis=0)
        r = np.concatenate ( (sample64x4, test_B0f, rec_B0, test_A0f, rec_A0, rec_A0_B0, rec_C_A0_B0), axis=1 )

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
    def DFDec64Flow(output_nc, padding='zero', **kwargs):
        exec (nnlib.import_all(), locals(), globals())

        def upscale (dim, padding='zero', norm='', act='', **kwargs):
            def func(x):
                return SubpixelUpscaler()(LeakyReLU(alpha=0.2)(Conv2D(dim * 4, kernel_size=3, strides=1, padding=padding)(x)))
            return func
        def to_bgr (output_nc, **kwargs):
            def func(x):
                return Conv2D(output_nc, kernel_size=5, strides=1, padding='same', activation='sigmoid')(x)
            return func

        upscale = partial(upscale)
        to_bgr = partial(to_bgr)

        def func(input):
            x = input[0]            
            x = upscale(256)( x )
            x = upscale(128)( x )
            x = upscale(64)( x )
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
                return SubpixelUpscaler()(LeakyReLU(alpha=0.2)(Conv2D(dim * 4, kernel_size=3, strides=1, padding=padding)(x)))
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