import os
import numpy as np

import tensorflow as tf

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import activations
from tensorflow.keras import regularizers

### RoadNet but tensorflow 2.x ###
class ConvBlock(object):
	def __init__(self, 
		out_channels,
		input_dim=(128,128,3), 
		pooling=True, 
		upsampling=0,
		use_selu=False,
		batch_norm=False,
		reg_lambda=2e-4):
		
		super(ConvBlock, self).__init__()
		self.input_dim=input_dim
		self.pooling=pooling
		self.upsampling=upsampling
		self.use_selu=use_selu
		self.batch_norm=batch_norm
		self.out_channels=out_channels
		self.reg_lambda=reg_lambda

	def build(self):
		kernel_size = (3,3)
		pool_size = (2,2)
		activation = 'relu'
		if(self.use_selu):
			activation = 'selu'

		inputs = Input(shape=self.input_dim)
		x = Conv2D(self.out_channels[0], 
			kernel_size=kernel_size, 
			activation=activation, 
			use_bias=False, padding='same',
			kernel_regularizer=regularizers.l2(l2=self.reg_lambda/2))(inputs)

		for i, channel in enumerate(self.out_channels[1:]):
			x = Conv2D(channel, kernel_size=kernel_size, 
				activation=activation, 
				use_bias=False, 
				padding='same',
				kernel_regularizer=regularizers.l2(l2=self.reg_lambda/2))(x)
			if(self.batch_norm):
				x = BatchNormalization()(x)

		side_output = x
		output = x

		if(self.pooling):
			output = MaxPooling2D(pool_size=pool_size)(output)

		side_output = Conv2D(1, kernel_size=(1,1), strides=1, activation=None, use_bias=False, 
			padding='same', kernel_regularizer=regularizers.l2(l2=self.reg_lambda/2))(side_output)
		if(self.upsampling > 0):
			side_output = UpSampling2D(size=(self.upsampling,self.upsampling), interpolation='bilinear')(side_output)

		side_output = activations.sigmoid(side_output)

		return Model(inputs = inputs, outputs=[output, side_output])

def _get_roadnet_module_1(input_shape):
	inputs = Input(shape=input_shape)
	out1, side1 = ConvBlock([64, 64], input_dim=inputs.shape[1:], pooling=True, upsampling=0, use_selu=True).build()(inputs)
	out2, side2 = ConvBlock([128, 128], input_dim=out1.shape[1:], pooling=True, upsampling=2, use_selu=True).build()(out1)
	out3, side3 = ConvBlock([256, 256, 256], input_dim=out2.shape[1:], pooling=True, upsampling=4, use_selu=True).build()(out2)
	out4, side4 = ConvBlock([512, 512, 512], input_dim=out3.shape[1:], pooling=True, upsampling=8, use_selu=True).build()(out3)
	out5, side5 = ConvBlock([512, 512, 512], input_dim=out4.shape[1:], pooling=False, upsampling=16, use_selu=True).build()(out4)

	out = Concatenate(axis=3)([side1, side2, side3, side4, side5])
	out = Conv2D(1, kernel_size=(1,1), activation=None, strides=1, use_bias=False, 
		padding='same', kernel_regularizer=regularizers.l2(l2=1e-4))(out)
	out = activations.sigmoid(out)

	return Model(inputs=inputs, outputs=[side1, side2, side3, side4, side5, out])

def _get_roadnet_module_2(input_shape):
	inputs = Input(shape=input_shape)
	out1, side1 = ConvBlock([32,32], input_dim=inputs.shape[1:], pooling=True, upsampling=0, use_selu=True).build()(inputs)
	out2, side2 = ConvBlock([64,64], input_dim=out1.shape[1:], pooling=True, upsampling=2, use_selu=True).build()(out1)
	out3, side3 = ConvBlock([128,128], input_dim=out2.shape[1:], pooling=True, upsampling=4, use_selu=True).build()(out2)
	out4, side4 = ConvBlock([256, 256], input_dim=out3.shape[1:], pooling=False, upsampling=8, use_selu=True).build()(out3)

	out = Concatenate(axis=3)([side1, side2, side3, side4])
	out = Conv2D(1, kernel_size=(1,1), activation=None, strides=1, use_bias=False, 
		padding='same', kernel_regularizer=regularizers.l2(l2=1e-4))(out)
	out = activations.sigmoid(out)

	return Model(inputs=inputs, outputs=[side1, side2, side3, side4, out])
