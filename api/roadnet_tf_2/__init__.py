import os 
import cv2
import numpy as np
import matplotlib.pyplot as plt

from argparse import ArgumentParser

from roadnet_tf_2.roadnet import RoadNet
from tensorflow.keras.models import Model

MODEL_CHECKPOINT = 'roadnet_tf_2/models/model_4.weights.hdf5'
TEST_IMG = '../data/1/Ottawa-1.tif'

original = cv2.imread(TEST_IMG)
#gt_segment = cv2.imread(TEST_LBL)

model = RoadNet().get_model()
model.load_weights(MODEL_CHECKPOINT)
model = Model(model.inputs, [
    model.get_layer('surface_final_output').output,
    model.get_layer('line_final_output').output,
    model.get_layer('edge_final_output').output
])

def crop_image(img, crop_size=(128,128)):
    crops = []

    H, W = img.shape[0], img.shape[1]

    ### Get the ratio to resize ###
    ratio_h = int(H/crop_size[0])
    ratio_w = int(W/crop_size[1])

    ### get the refined resize dimensions ###
    resized_dimensions = (crop_size[1] * ratio_w , crop_size[0] * ratio_h)

    ### resize the image ###
    img_resize = cv2.resize(img, resized_dimensions)

    ### Divide the images into chunks of 128 x 128 squares ###
    for i in range(ratio_h):
        for j in range(ratio_w):
            crop = img_resize[i*crop_size[0]: (i+1)*crop_size[0], j*crop_size[1]:(j+1)*crop_size[1]]

            crops.append(crop)

    return crops, ratio_h, ratio_w

# img = cv2.imread(TEST_IMG)
def make_prediction(img):
    crops, ratio_h, ratio_w = crop_image(img)

    crops = np.array(crops)

    print('[INFO] Running full prediction on segmentation')
    full_image = None
    full_image_line = None
    full_image_edge = None

    def parse_to_binary_map(map_):
        ### Make foreground black and background white ###
        ''' Easier to parse to geojson later on'''
        map_[map_ > 0.5] = 1
        map_[map_ < 0.5] = 0

        return map_

    for i in range(ratio_h):
        horizontal_image = None
        horizontal_image_line = None
        horizontal_image_edge = None
        for j in range(ratio_w):
            index = ratio_w * i + j

            map_, line, edge = model.predict(np.array([crops[index]]))
            map_ = parse_to_binary_map(map_[0])
            line = parse_to_binary_map(line[0])
            edge = parse_to_binary_map(edge[0])

            if(j == 0):
                horizontal_image = map_
                horizontal_image_line = line
                horizontal_image_edge = edge
            else:
                horizontal_image = cv2.hconcat([horizontal_image, map_])
                horizontal_image_line = cv2.hconcat([horizontal_image_line, line])
                horizontal_image_edge = cv2.hconcat([horizontal_image_edge, edge])

        if(i == 0):
            full_image = horizontal_image
            full_image_line = horizontal_image_line
            full_image_edge = horizontal_image_edge
        else:
            full_image = cv2.vconcat([full_image, horizontal_image])
            full_image_line = cv2.vconcat([full_image_line, horizontal_image_line])
            full_image_edge = cv2.vconcat([full_image_edge, horizontal_image_edge])

    basename = os.path.basename(TEST_IMG)
    filename = basename.split('.')[0]
    save_file_name = "sample_predictions/full_prediction_" + filename + ".jpg"
    line_file_name = "sample_predictions/centerline_" + filename + ".jpg"

    full_image *= 255
    full_image = full_image.astype(np.uint8)

    full_image_line_3d = np.zeros((full_image.shape[0], full_image.shape[1], 3))
    full_image_line_3d[full_image_line == 0] = [0, 0, 0]
    full_image_line_3d[full_image_line == 1] = [255,255,255]

    ### dilate image abit for readability ###
    #kernel = np.ones((5,5), np.uint8)
    #full_image_line_3d = cv2.dilate(full_image_line_3d, kernel, iterations=1)
    full_image_line_3d = 255 - full_image_line_3d

    full_image_edge_3d = np.zeros((full_image.shape[0], full_image.shape[1], 3))
    full_image_edge_3d[full_image_edge == 0] = [0, 0, 0]
    full_image_edge_3d[full_image_edge == 1] = [255,255,255]

    ### dilate image abit for readability ###
    kernel = np.ones((5,5), np.uint8)
    full_image_edge_3d = cv2.dilate(full_image_edge_3d, kernel, iterations=2)
    full_image_edge_3d = cv2.erode(full_image_edge_3d, kernel, iterations=1)
    full_image_edge_3d = 255 - full_image_edge_3d ### Invert the image ###

    print('[INFO] Saving prediction result in %s' % save_file_name)
    # cv2.imwrite(line_file_name, full_image_line_3d)
    return full_image
