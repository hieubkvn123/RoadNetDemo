import os
import cv2
import numpy as np

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader

### For ottawa dataset ###
train_data = [2,3,4,5,6,7,8,9,10,11,12,13,14,15]
test_data = [1, 16, 17, 18, 19, 20]

def crop_images(img, crop_size=(128, 128)):
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

    return crops

### Currently specific to Ottawa dataset ###
def load_dataset_from_dir(folder_name, train_labels, test_labels, crop_size=(128, 128)):
    '''
        The data folder must have the following structure :
        |- <DATA_FOLDER>
            |-<CLASS_1>
                |-image.png
                |-segmentation.png
                |-centerline.png
                |-edge.png
                |-extra.png
            |-<CLASS_2>
            ...
    '''
    ### initialize arrays ###
    train_images = []
    test_images = []

    train_segments = []
    train_centerlines = []
    train_edges = []

    test_segments = []
    test_centerlines = []
    test_edges = []

    labels = train_labels + test_labels

    for entry in labels:
        abs_img_path = folder_name + ("/%d/" % entry) + ("Ottawa-%d.tif" % entry)
        abs_surface_path = folder_name + ("/%d/" % entry) + "segmentation.png"
        abs_edge_path = folder_name + ("/%d/" % entry) + "edge.png"
        abs_centerline_path = folder_name + ("/%d/" % entry) + "centerline.png"

        print(abs_img_path)
        img = cv2.imread(abs_img_path)
        edge = cv2.cvtColor(cv2.imread(abs_edge_path), cv2.COLOR_BGR2GRAY)
        surface = cv2.cvtColor(cv2.imread(abs_surface_path), cv2.COLOR_BGR2GRAY)
        centerline = cv2.cvtColor(cv2.imread(abs_centerline_path), cv2.COLOR_BGR2GRAY)

        ### Convert all label images to binary ###
        edge[edge < 250] = 1
        edge[edge >= 250] = 0

        surface[surface < 250] = 1
        surface[surface >= 250] = 0

        centerline[centerline < 250] = 1
        centerline[centerline >= 250] = 0

        img_crops = crop_images(img, crop_size=crop_size)
        surface_crops = crop_images(surface, crop_size=crop_size) 
        edge_crops = crop_images(edge, crop_size=crop_size)
        centerline_crops = crop_images(centerline, crop_size=crop_size)

        if(entry in train_labels):
            print("[INFO] Processing training image with id %d ..." % entry)
            train_images.extend(img_crops)
            train_segments.extend(surface_crops)
            train_edges.extend(edge_crops)
            train_centerlines.extend(centerline_crops)
        elif(entry in test_labels):
            print("[INFO] Processing testing image with id %d ..." % entry)
            test_images.extend(img_crops)
            test_segments.extend(surface_crops)
            test_edges.extend(edge_crops)
            test_centerlines.extend(centerline_crops)

    train_images = np.array(train_images).reshape(-1, 3, crop_size[0], crop_size[1])
    test_images  = np.array(test_images).reshape(-1, 3, crop_size[0], crop_size[1])

    train_segments = np.array(train_segments).reshape(-1, 1, crop_size[0], crop_size[1])
    test_segments  = np.array(test_segments).reshape(-1, 1, crop_size[0], crop_size[1])
    
    train_centerlines = np.array(train_centerlines).reshape(-1, 1, crop_size[0], crop_size[1])
    test_centerlines  = np.array(test_centerlines).reshape(-1, 1, crop_size[0], crop_size[1])

    train_edges = np.array(train_edges).reshape(-1, 1, crop_size[0], crop_size[1])
    test_edges  = np.array(test_edges).reshape(-1, 1, crop_size[0], crop_size[1])

    print('[*] Train images size : ', train_images.shape[0])
    print('[*] Train segments images size : ', train_segments.shape[0])
    print('[*] Train centerlines images size : ', train_centerlines.shape[0])
    print('[*] Train edges images size : ', train_edges.shape[0])

    print()
    print('[*] Test images size : ', test_images.shape[0])
    print('[*] Test segments images size : ', test_segments.shape[0])
    print('[*] Test centerlines images size : ', test_centerlines.shape[0])
    print('[*] Test edges images size : ', test_edges.shape[0])

    ### Convert everything to tensors ###
    train_images = torch.from_numpy(train_images).float()
    train_segments = torch.from_numpy(train_segments).float()
    train_centerlines = torch.from_numpy(train_centerlines).float()
    train_edges = torch.from_numpy(train_edges).float()

    test_images = torch.from_numpy(test_images).float()
    test_segments = torch.from_numpy(test_segments).float()
    test_centerlines = torch.from_numpy(test_centerlines).float()
    test_edges = torch.from_numpy(test_edges).float()

    train_dataset = TensorDataset(train_images, train_segments, train_centerlines, train_edges)
    test_dataset = TensorDataset(test_images, test_segments, test_centerlines, test_edges)

    train_loader = DataLoader(train_dataset)
    test_loader = DataLoader(test_dataset)

    return train_loader, test_loader