import os
import numpy as np

import tensorflow as tf

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

class RoadNetModule1(Model):
    def __init__(self, input_shape=(128,128)):
        super(RoadNetModule1, self).__init__()
        self.input_shape=input_shape

    def build():

