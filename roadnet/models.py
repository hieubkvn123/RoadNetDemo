import torch
import torch.nn as nn
import torch.functional as F

from networks import RoadNetModule1, RoadNetModule2

class RoadNet(nn.Module):
    def __init__(self, input_shape=(128, 128)):
        super(RoadNet, self).__init__()
        self.input_shape = input_shape
        self.height = input_shape[1]
        self.width  = input_shape[0]

        self._segment_net = RoadNetModule1(input_shape=(3, self.height, self.width))
        self._centerline_net = RoadNetModule2(input_shape=(4, self.height, self.width))
        self._edge_net = RoadNetModule2(input_shape=(4, self.height, self.width))

    def _segment_forward(self, inputs):
        side_segment1, side_segment2, side_segment3, side_segment4, side_segment5, segment = self._segment_net(inputs)
        
        return [side_segment1, side_segment2, side_segment3, side_segment4, side_segment5, segment]

    def _centerline_forward(self, inputs):
        side_centerline1, side_centerline2, side_centerline3, side_centerline4, centerline = self._centerline_net(inputs)

        return [side_centerline1, side_centerline2, side_centerline3, side_centerline4, centerline]

    def _edge_forward(self, inputs):
        side_edge1, side_edge2, side_edge3, side_edge4, edge = self._edge_net(inputs)
        
        return [side_edge1, side_edge2, side_edge3, side_edge4, edge]

    def forward(self, inputs):
        segments = self._segment_forward(inputs)
        segment = segments[-1]

        inputs2 = torch.cat([inputs, segment], dim=1)
        centerlines = self._centerline_forward(inputs2)
        edges = self._edge_forward(inputs2)

        return segments, centerlines, edges

