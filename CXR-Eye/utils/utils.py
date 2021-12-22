import os
import logging
import math
import torch
import time
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from .visualization import VisdomLinePlotter, plot_roc_curve
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
import cv2

logger = logging.getLogger('cxr')

N_list = [2,3,4]
proposalN = sum(N_list)  # proposal window num
iou_threshs = [0.25, 0.25, 0.25]
stride = 2
ratios = [[180, 190], [190, 180],
          [123, 135], [135, 123], [123, 123], [135, 135],
          [87, 95], [95, 87], [87, 87], [95, 95]
          ] 

input_size = 224

def compute_window_nums(ratios, stride, input_size):
    window_nums = []

    for _, ratio in enumerate(ratios):
        window_nums.append(int((input_size - ratio[0])/stride + 1) * int((input_size - ratio[1])/stride + 1))
    return window_nums

def ComputeCoordinate(image_size, stride, indice, ratio):
    column_window_num = int((image_size - ratio[1])/stride + 1)
    x_indice = indice // column_window_num
    y_indice = indice % column_window_num
    x_lefttop = x_indice * stride - 1
    y_lefttop = y_indice * stride - 1
    x_rightlow = x_lefttop + ratio[0]
    y_rightlow = y_lefttop + ratio[1]
    # for image
    if x_lefttop < 0:
        x_lefttop = 0
    if y_lefttop < 0:
        y_lefttop = 0
    coordinate = np.array((x_lefttop, y_lefttop, x_rightlow, y_rightlow), dtype=object).reshape(1, 4)

    return coordinate


def indices2coordinates(indices, stride, image_size, ratio):
    batch, _ = indices.shape
    coordinates = []

    for j, indice in enumerate(indices):
        coordinates.append(ComputeCoordinate(image_size, stride, indice, ratio))

    coordinates = np.array(coordinates).reshape(batch,4).astype(int)  
    return coordinates

'''indice2coordinates'''
window_nums = compute_window_nums(ratios, stride, input_size)
indices_ndarrays = [np.arange(0,window_num).reshape(-1,1) for window_num in window_nums]
coordinates = [indices2coordinates(indices_ndarray, stride, input_size, ratios[i]) for i, indices_ndarray in enumerate(indices_ndarrays)] # 每个window在image上的坐标
coordinates_cat = np.concatenate(coordinates, 0)
window_milestones = [sum(window_nums[:i+1]) for i in range(len(window_nums))]
window_nums_sum = [0, sum(window_nums[:2]), sum(window_nums[2:6]), sum(window_nums[6:])]

def image_with_boxes(image, coordinates=None, color=None):
    '''
    :param image: image array(CHW) tensor
    :param coordinate: bounding boxs coordinate, coordinates.shape = [proposalN, 4], coordinates[0] = (x0, y0, x1, y1)
    :return:image with bounding box(HWC)
    '''

    if type(image) is not np.ndarray:
        image = image.clone().detach()
        rgbN = [(255, 0, 0), (255, 0, 0), (255, 165, 0), (255, 165, 0), (255, 165, 0), (0, 255, 0), (0, 255, 0), (0, 255, 0), (0, 255, 0)]

        # Anti-normalization
        std = [0.229, 0.224, 0.225]
        mean = [0.485, 0.456, 0.406]
        image[0] = image[0] * std[0] + mean[0]
        image[1] = image[1] * std[1] + mean[1]
        image[2] = image[2].mul(std[2]) + mean[2]
        image = image.mul(255).byte()

        image = image.data.cpu().numpy()

        image.astype(np.uint8)

        image = np.transpose(image, (1, 2, 0))  # CHW --> HWC
        image = image.copy()
        

    if coordinates is not None:
        for i, coordinate in enumerate(coordinates):
            if color:
                image = cv2.rectangle(image, (int(coordinate[1]), int(coordinate[0])),
                                      (int(coordinate[3]), int(coordinate[2])),
                                      color, 2)
            else:
                if i < proposalN:
                # coordinates(x, y) is reverse in numpy
                    image = cv2.rectangle(image, (int(coordinate[1]), int(coordinate[0])), (int(coordinate[3]), int(coordinate[2])),
                                          rgbN[i], 2)
                else:
                    image = cv2.rectangle(image, (int(coordinate[1]), int(coordinate[0])),
                                          (int(coordinate[3]), int(coordinate[2])),
                                          (255, 255, 255), 2)
    return image

def nms(scores_np, proposalN, iou_threshs, coordinates, IND_RANDOM=None):
    if not (type(scores_np).__module__ == 'numpy' and len(scores_np.shape) == 2 and scores_np.shape[1] == 1):
        raise TypeError('score_np is not right')

    windows_num = scores_np.shape[0]
    indices_coordinates = np.concatenate((scores_np, coordinates), 1)
    # # indices = np.arange(windows_num)
    # # np.random.seed(1)
    # # np.random.shuffle(indices)
    if IND_RANDOM is not None:
        indices = IND_RANDOM
    else:
        indices = np.argsort(indices_coordinates[:, 0])
    indices_coordinates = np.concatenate((indices_coordinates, np.arange(0,windows_num).reshape(windows_num,1)), 1)[indices]                  #[339,6]
    indices_results = []

    res = indices_coordinates

    while res.any():
        indice_coordinates = res[-1]
        indices_results.append(indice_coordinates[5])

        if len(indices_results) == proposalN:
            return np.array(indices_results).reshape(1,proposalN).astype(np.int)
        res = res[:-1]

        # Exclude anchor boxes with selected anchor box whose iou is greater than the threshold
        start_max = np.maximum(res[:, 1:3], indice_coordinates[1:3])
        end_min = np.minimum(res[:, 3:5], indice_coordinates[3:5])
        lengths = end_min - start_max + 1
        intersec_map = lengths[:, 0] * lengths[:, 1]
        intersec_map[np.logical_or(lengths[:, 0] < 0, lengths[:, 1] < 0)] = 0
        iou_map_cur = intersec_map / ((res[:, 3] - res[:, 1] + 1) * (res[:, 4] - res[:, 2] + 1) +
                                      (indice_coordinates[3] - indice_coordinates[1] + 1) *
                                      (indice_coordinates[4] - indice_coordinates[2] + 1) - intersec_map)
        res = res[iou_map_cur <= iou_threshs]

    while len(indices_results) != proposalN:
        indices_results.append(indice_coordinates[5])

    return np.array(indices_results).reshape(1, -1).astype(np.int)

def load_model(model_name, model_dir, model):
    output_weights_name = os.path.join(model_dir, model_name)
    logger.info(f'MODEL FILE --- {output_weights_name}')
    if os.path.isfile(output_weights_name) == False:
        logger.info(f'No such model file: {output_weights_name}')
        return False
    model.load_state_dict(torch.load(output_weights_name))
    return model
