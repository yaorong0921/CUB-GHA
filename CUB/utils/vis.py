import numpy as np
import cv2
from config import proposalN, model_path, model_name
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
import os

def image_with_boxes(image, coordinates=None, color=None):
    '''
    :param image: image array(CHW) tensor
    :param coordinate: bounding boxs coordinate, coordinates.shape = [proposalN, 4], coordinates[0] = (x0, y0, x1, y1)
    :return:image with bounding box(HWC)
    '''

    if type(image) is not np.ndarray:
        image = image.clone().detach()

        # rgbN = [(255, 0, 0), (255, 165, 0), (255, 255, 0), (0, 255, 0), (0, 255, 0), (0, 255, 0), (0, 255, 0)]
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


def denormalize(s):
    invTrans = transforms.Compose([transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                               ])

    inv_tensor = invTrans(s)
    return inv_tensor

def normalizeData(data):
    eps = 1e-07
    return (data - torch.min(data)) / (torch.max(data) - torch.min(data) + eps)

def imshow(img, image_name):
    # img = normalize_map(img)
    img = denormalize(img) # unnormalize
    img = normalizeData(img)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    save_path = save_path = os.path.join(model_path, model_name, 'viz')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig('%s/%s.jpg'%(save_path, image_name))

def heatmap_show(img, mask, image_name):
    img = denormalize(img) # unnormalize
    img = normalizeData(img)
    img = np.transpose(img.numpy(), (1, 2, 0))
    save_path = save_path = os.path.join(model_path, model_name, 'viz')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    mask = np.transpose(mask.numpy(), (1, 2, 0))
    plt.imshow(img)
    plt.imshow(mask, cmap='jet', alpha=0.5)
    plt.axis('off')
    # heatmap_image_npy = np.multiply(heatmap_image_npy, 255.0)
    # heatmap_image_npy = np.rollaxis(heatmap_image_npy, 0, 3)
    # heatmap_image_npy = cv2.applyColorMap(np.uint8(heatmap_image_npy), cv2.COLORMAP_JET)
    plt.colorbar(cmap='jet', fraction=0.046, pad=0.04)
    plt.savefig('%s/%s.jpg'%(save_path, image_name))