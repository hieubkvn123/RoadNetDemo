import numpy as np 
import tensorflow as tf 

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model 
from networks import _get_roadnet_module_1, _get_roadnet_module_2

class RoadNet(object):
	def __init__(self, input_shape=(128,128,3)):
		self.input_shape = input_shape

		H, W, C = self.input_shape
		self._segment_net = _get_roadnet_module_1((H, W, C))
		self._centerline_net = _get_roadnet_module_2((H, W, C+1))
		self._edge_net = _get_roadnet_module_2((H, W, C+1))

	def bce_with_logits(self, y_true, y_pred):
		pass

	def build(self):
		inputs = Input(shape=self.input_shape)

		side_seg1, side_seg2, side_seg3, side_seg4, side_seg5, seg = self._segment_net(inputs)
		inputs2 = Concatenate(axis=3)([inputs, seg])

		side_line1, side_line2, side_line3, side_line4, line = self._centerline_net(inputs2)
		side_edge1, side_edge2, side_edge3, side_edge4, edge = self._edge_net(inputs2)

		side_seg1 = Lambda(lambda x : x, name='side_seg_1')(side_seg1)
		side_seg2 = Lambda(lambda x : x, name='side_seg_2')(side_seg2)
		side_seg3 = Lambda(lambda x : x, name='side_seg_3')(side_seg3)
		side_seg4 = Lambda(lambda x : x, name='side_seg_4')(side_seg4)
		side_seg5 = Lambda(lambda x : x, name='side_seg_5')(side_seg5)
		seg       = Lambda(lambda x : x, name='seg')(seg)

		side_line1 = Lambda(lambda x : x, name='side_line_1')(side_line1)
		side_line2 = Lambda(lambda x : x, name='side_line_2')(side_line2)
		side_line3 = Lambda(lambda x : x, name='side_line_3')(side_line3)
		side_line4 = Lambda(lambda x : x, name='side_line_4')(side_line4)
		line       = Lambda(lambda x : x, name='line')(line)

		side_edge1 = Lambda(lambda x : x, name='side_edge_1')(side_edge1)
		side_edge2 = Lambda(lambda x : x, name='side_edge_2')(side_edge2)
		side_edge3 = Lambda(lambda x : x, name='side_edge_3')(side_edge3)
		side_edge4 = Lambda(lambda x : x, name='side_edge_4')(side_edge4)
		edge       = Lambda(lambda x : x, name='edge')(edge)

		model = Model(inputs=inputs, outputs=[side_seg1, side_seg2, side_seg3, side_seg4, side_seg5, seg,
			side_line1, side_line2, side_line3, side_line4,
			side_edge1, side_edge2, side_edge3, side_edge4])

		return model