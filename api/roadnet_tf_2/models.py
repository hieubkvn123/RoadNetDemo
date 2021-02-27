### Copied from models_1.py ###
### Changes : ###
'''
    1. No activation at side outputs, only activation at fused output
    2. activation = (sigmoid(logits) - 0.5) / 0.5
'''

import os
import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import GlorotNormal

'''
=======================Changes==========================
    - Add kernel regularizer (l2) to all conv layers
    - Add image enoising to data before fitting
'''

class RoadSurfaceNet(object):
    def __init__(self, input_shape=(128,128,3), name='surface'):
        lambda_ = 1e-4
        self.input_shape=input_shape
        self.name=name
        self.xavier=GlorotNormal()
        self.l2 = l2(lambda_/2)
        
    def selu(self, x):
        alpha = 1.67326324
        scale = 1.05070098

        mask_1 = tf.cast((x > 0), dtype=tf.float32)
        mask_2 = tf.cast((x < 0), dtype=tf.float32)

        selu_ = (x * scale * mask_1) + ((scale * alpha * (tf.math.exp(x) - 1)) * mask_2)

        return selu_

    def get_model(self):
        ### Stage 1 ###
        inputs = Input(shape=self.input_shape)
        conv_1 = Conv2D(8, kernel_size=(3,3), activation='selu', padding='same', 
                kernel_initializer=self.xavier,
                kernel_regularizer=self.l2)(inputs)
        conv_1 = Conv2D(8, kernel_size=(3,3), activation='selu', padding='same', 
                kernel_initializer=self.xavier,
                kernel_regularizer=self.l2)(conv_1)
        pool_1 = MaxPooling2D(pool_size=(2,2))(conv_1)

        ### First output ###
        ### Size = W * H * 1 ###
        side_1_logits = Conv2D(1, kernel_size=(1,1), padding='same', name='surface_output_1', 
                kernel_initializer=self.xavier,
                kernel_regularizer=self.l2)(conv_1)
        side_1 = tf.keras.activations.sigmoid(side_1_logits)

        ### Stage 2 ###
        conv_2 = Conv2D(16, kernel_size=(3,3), activation='selu', padding='same',
                kernel_initializer=self.xavier,
                kernel_regularizer=self.l2)(pool_1)
        conv_2 = Conv2D(16, kernel_size=(3,3), activation='selu', padding='same',
                kernel_initializer=self.xavier,
                kernel_regularizer=self.l2)(conv_2)
        pool_2 = MaxPooling2D(pool_size=(2,2))(conv_2)

        ### Second output ###
        ### Size = W/2 * H/2 * 1 ###
        side_2_logits = Conv2D(1, kernel_size=(1,1), kernel_initializer=self.xavier,
                kernel_regularizer=self.l2)(conv_2)
        side_2_up = UpSampling2D(size=(2,2), interpolation='bilinear')(side_2_logits)
        side_2 = tf.keras.activations.sigmoid(side_2_up)
        
        ### Stage 3 ###
        conv_3 = Conv2D(32, kernel_size=(3,3), activation='selu', padding='same', 
                kernel_initializer=self.xavier,
                kernel_regularizer=self.l2)(pool_2)
        conv_3 = Conv2D(32, kernel_size=(3,3), activation='selu', padding='same', 
                kernel_initializer=self.xavier,
                kernel_regularizer=self.l2)(conv_3)
        conv_3 = Conv2D(32, kernel_size=(3,3), activation='selu', padding='same', 
                kernel_initializer=self.xavier,
                kernel_regularizer=self.l2)(conv_3)
        pool_3 = MaxPooling2D(pool_size=(2,2))(conv_3)

        ### Third output ###
        ### Size = W/4 * H/4 * 1 ###
        side_3_logits = Conv2D(1, kernel_size=(1,1), kernel_initializer=self.xavier,
                kernel_regularizer=self.l2)(conv_3)
        side_3_up = UpSampling2D(size=(4,4), interpolation='bilinear')(side_3_logits)
        side_3 = tf.keras.activations.sigmoid(side_3_up)
        
        ### Stage 4 ###
        conv_4 = Conv2D(64, kernel_size=(3,3), activation='selu', padding='same', 
                kernel_initializer=self.xavier,
                kernel_regularizer=self.l2)(pool_3)
        conv_4 = Conv2D(64, kernel_size=(3,3), activation='selu', padding='same', 
                kernel_initializer=self.xavier,
                kernel_regularizer=self.l2)(conv_4)
        conv_4 = Conv2D(64, kernel_size=(3,3), activation='selu', padding='same', 
                kernel_initializer=self.xavier,
                kernel_regularizer=self.l2)(conv_4)
        pool_4 = MaxPooling2D(pool_size=(2,2))(conv_4)

        ### Fourth output ###
        ### Size = W/8 * H/8 * 1 ###
        side_4_logits = Conv2D(1, kernel_size=(1,1), kernel_initializer=self.xavier,
                kernel_regularizer=self.l2)(conv_4)
        side_4_up = UpSampling2D(size=(8,8), interpolation='bilinear')(side_4_logits)
        side_4 = tf.keras.activations.sigmoid(side_4_up)
        
        ### Stage 5 ###
        conv_5 = Conv2D(128, kernel_size=(3,3), activation='selu', padding='same',
                kernel_initializer=self.xavier,
                kernel_regularizer=self.l2)(pool_4) 
        conv_5 = Conv2D(128, kernel_size=(3,3), activation='selu', padding='same', 
                kernel_initializer=self.xavier,
                kernel_regularizer=self.l2)(conv_5)
        conv_5 = Conv2D(128, kernel_size=(3,3), activation='selu', padding='same', 
                kernel_initializer=self.xavier,
                kernel_regularizer=self.l2)(conv_5)

        ### Fifth output ###
        ### Size = W/16 * H/16 * 1 ###
        side_5_logits = Conv2D(1, kernel_size=(1,1),kernel_initializer=self.xavier,
                kernel_regularizer=self.l2)(conv_5)
        side_5_up = UpSampling2D(size=(16,16), interpolation='bilinear')(side_5_logits)
        side_5 = tf.keras.activations.sigmoid(side_5_up)
        
        ### concatenated output ###
        concat = Concatenate(axis=3)([side_1_logits, side_2_up, side_3_up, side_4_up, side_5_up])
        # concat = tf.keras.layers.concatenate((side_1, side_2, side_3, side_4, side_5), axis=3)
        concat = Conv2D(1, kernel_size=(1,1), name='surface_concat', kernel_regularizer=l2(1e-4), activation='sigmoid', kernel_initializer=self.xavier)(concat)
        # concat = (tf.keras.activations.sigmoid(concat) - 0.5) / 0.5

        ### We need to pass the original inputs and the final concat to centerline and edge networks ###
        model = Model(inputs=inputs, outputs=[side_1,side_2,side_3,side_4,side_5,concat],name=self.name)

        return model

class SideNet(object):
    def __init__(self, input_shape=(128,128,1), name='centerline_net'):
        lambda_ = 1e-4
        self.input_shape=input_shape
        self.name = name
        self.xavier=GlorotNormal()
        self.l2 = l2(lambda_/2)

    def selu(self, x):
        alpha = 1.67326324
        scale = 1.05070098

        mask_1 = tf.cast((x > 0), dtype=tf.float32)
        mask_2 = tf.cast((x < 0), dtype=tf.float32)

        selu_ = (x * scale * mask_1) + ((scale * alpha * (tf.math.exp(x) - 1)) * mask_2)

        return selu_


    def get_model(self):
        ### Original Image ###
        input_1 = Input(shape=(128,128,3))
        ### The segmented map ###
        input_2 = Input(shape=self.input_shape)
        
        ### Concatenate the two inputs along the channels ###
        # concat_input = tf.keras.layers.concatenate((input_1, input_2), axis=3)
        concat_input = Concatenate(axis=3)([input_1, input_2])

        conv_1 = Conv2D(8, kernel_size=(3,3), activation='selu', padding='same',
                kernel_initializer=self.xavier,
                kernel_regularizer=self.l2)(concat_input)
        conv_1 = Conv2D(8, kernel_size=(3,3), activation='selu', padding='same',
                kernel_initializer=self.xavier,
                kernel_regularizer=self.l2)(conv_1)
        pool_1 = MaxPooling2D(pool_size=(2,2))(conv_1)

        ### First output ###
        side_1_logits = Conv2D(1, kernel_size=(1,1), name=self.name+'_output_1',kernel_initializer=self.xavier,
                kernel_regularizer=self.l2)(conv_1)
        side_1 = tf.keras.activations.sigmoid(side_1_logits)

        conv_2 = Conv2D(16, kernel_size=(3,3), activation='selu', padding='same',
                kernel_initializer=self.xavier, 
                kernel_regularizer=self.l2)(pool_1)
        conv_2 = Conv2D(16, kernel_size=(3,3), activation='selu', padding='same',
                kernel_initializer=self.xavier,
                kernel_regularizer=self.l2)(conv_2)
        pool_2 = MaxPooling2D(pool_size=(2,2))(conv_2)

        ### Second output ###
        side_2_logits = Conv2D(1, kernel_size=(1,1),kernel_initializer=self.xavier,
                kernel_regularizer=self.l2)(conv_2)
        side_2_up = UpSampling2D(size=(2,2), interpolation='bilinear')(side_2_logits)
        side_2 = tf.keras.activations.sigmoid(side_2_up)
        
        conv_3 = Conv2D(32, kernel_size=(3,3), activation='selu', padding='same',
                kernel_initializer=self.xavier,
                kernel_regularizer=self.l2)(pool_2)
        conv_3 = Conv2D(32, kernel_size=(3,3), activation='selu', padding='same',
                kernel_initializer=self.xavier,
                kernel_regularizer=self.l2)(conv_3)
        pool_3 = MaxPooling2D(pool_size=(2,2))(conv_3)

        ### Third output ###
        side_3_logits = Conv2D(1, kernel_size=(1,1),kernel_initializer=self.xavier,
                kernel_regularizer=self.l2)(conv_3)
        side_3_up = UpSampling2D(size=(4,4), interpolation='bilinear')(side_3_logits)
        side_3 = tf.keras.activations.sigmoid(side_3_up)
        
        conv_4 = Conv2D(64, kernel_size=(3,3), activation='selu', padding='same',
                kernel_initializer=self.xavier,
                kernel_regularizer=self.l2)(pool_3)
        conv_4 = Conv2D(64, kernel_size=(3,3), activation='selu', padding='same',
                kernel_initializer=self.xavier,
                kernel_regularizer=self.l2)(conv_4)

        ### Fourth output ###
        side_4_logits = Conv2D(1, kernel_size=(1,1),kernel_initializer=self.xavier,
                kernel_regularizer=self.l2)(conv_4)
        side_4_up = UpSampling2D(size=(8,8), interpolation='bilinear')(side_4_logits)
        side_4 = tf.keras.activations.sigmoid(side_4_up)
        
        concat = Concatenate(axis=3)([side_1_logits, side_2_up, side_3_up, side_4_up])
        # concat = tf.keras.layers.concatenate((side_1, side_2, side_3, side_4), axis=3)
        concat = Conv2D(1, kernel_size=(1,1), name=self.name+'_concat', kernel_regularizer=l2(1e-4), activation='sigmoid', kernel_initializer=self.xavier)(concat) 
        # concat = (tf.keras.activations.sigmoid(concat) - 0.5)/0.5

        model = Model(inputs=[input_1, input_2], outputs=[side_1, side_2, side_3, side_4, concat], name=self.name)

        return model
