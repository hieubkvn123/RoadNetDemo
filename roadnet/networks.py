import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class ConvBlock(nn.Module):
    def __init__(self, in_channels, 
            out_channels, 
            pooling, 
            upsampling,
            use_selu=False, 
            batch_norm=False):
        super(ConvBlock, self).__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.pooling=pooling
        self.upsampling=upsampling

        kernel_size=(3,3)
        pool_size=(2,2)
        
        self.conv_module = nn.Sequential()
        self.conv_module.add_module("conv1", nn.Conv2d(in_channels=self.in_channels, 
                out_channels=self.out_channels[0], 
                kernel_size=kernel_size, 
                stride=1,
                padding=kernel_size[0] - 2,
                bias=False))

        # Activation layer
        if(use_selu):
            self.conv_module.add_module("selu1", nn.SELU(True))
        else:
            self.conv_module.add_module("relu1", nn.ReLU(True))

        if(batch_norm):
            self.conv_module.add_module("batch_norm1", nn.BatchNorm2d(num_features=self.out_channels[0]))

        for i, _filter in enumerate(self.out_channels[1:]):
            self.conv_module.add_module("conv" + str(i+2), nn.Conv2d(in_channels=self.out_channels[i],
                    out_channels=self.out_channels[i+1],
                    kernel_size=kernel_size,
                    stride=1,
                    padding=kernel_size[0] - 2,
                    bias=False))

            if(use_selu):
                self.conv_module.add_module("selu" + str(i+2), nn.SELU(True))
            else:
                self.conv_module.add_module("relu" + str(i+2), nn.ReLU(True))

            if(batch_norm):
                self.conv_module.add_module("batch_norm" + str(i+2), nn.BatchNorm2d(num_features=self.out_channels[i+1]))

        self.maxpool = nn.MaxPool2d(kernel_size=pool_size,
                stride=2)
        self.final_conv = nn.Conv2d(in_channels=self.out_channels[-1],
                out_channels=1,
                kernel_size=(1,1),
                stride=1,
                padding=0,
                bias=False)

    def forward(self, inputs):
        x = self.conv_module(inputs)
        output = x 

        if(self.pooling):
            output = self.maxpool(output)

        side_output = self.final_conv(x)
        if(self.upsampling is not None):
            side_output = F.interpolate(side_output,
                    size=self.upsampling,
                    mode='bilinear',
                    align_corners=True)

        return output, side_output

class RoadNetModule1(nn.Module):
    def __init__(self, input_shape=(3, 128, 128)):
        super(RoadNetModule1, self).__init__()
        self.input_shape=input_shape

        C, H, W = self.input_shape
        self._conv_1 = ConvBlock(C, [64, 64], True, None, use_selu=True, batch_norm=False)
        self._conv_2 = ConvBlock(64, [128, 128], True, (H, W), use_selu=True, batch_norm=False)
        self._conv_3 = ConvBlock(128, [256, 256, 256], True, (H, W), use_selu=True, batch_norm=False)
        self._conv_4 = ConvBlock(256, [512, 512, 512], True, (H, W), use_selu=True, batch_norm=False)
        self._conv_5 = ConvBlock(512, [512, 512, 512], False, (H, W), use_selu=True, batch_norm=False)
        self._final_conv = nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1, stride=1, bias=False)

    def forward(self, inputs):
        out1, side1 = self._conv_1(inputs)
        out2, side2 = self._conv_2(out1)
        out3, side3 = self._conv_3(out2)
        out4, side4 = self._conv_4(out3)
        out5, side5 = self._conv_5(out4)

        concat = torch.cat([side1, side2, side3, side4, side5], dim=1)
        out = self._final_conv(concat)
        return side1, side2, side3, side4, side5, out

class RoadNetModule2(nn.Module):
    ### Input Shape = 4 because we concatenate input images (3 channels)
    ### And segmentation layer (1 channel) together ###
    def __init__(self, input_shape=(4, 128, 128)):
        super(RoadNetModule2, self).__init__()
        self.input_shape=input_shape
        
        C, H, W = self.input_shape
        self._conv_1 = ConvBlock(C, [32, 32], True, None, use_selu=True, batch_norm=False)
        self._conv_2 = ConvBlock(32, [64, 64], True, (H, W), use_selu=True, batch_norm=False)
        self._conv_3 = ConvBlock(64, [128, 128], True, (H, W), use_selu=True, batch_norm=False)
        self._conv_4 = ConvBlock(128, [256, 256], False, (H, W), use_selu=True, batch_norm=False)
        self._final_conv = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=1, stride=1, bias=False)

    def forward(self, inputs):
        out1, side1 = self._conv_1(inputs)
        out2, side2 = self._conv_2(out1)
        out3, side3 = self._conv_3(out2)
        out4, side4 = self._conv_4(out3)

        concat = torch.cat([side1, side2, side3, side4], dim=1)
        out = self._final_conv(concat)
        return side1, side2, side3, side4, out
