# -- Generator
import os
import re
import torch
import random
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import csv
import torchvision.transforms as transforms
import torch.nn.functional as F

def normalizeData(data):
    eps = 1e-07
    return (data - torch.min(data)) / (torch.max(data) - torch.min(data) + eps)



def denormalize(s):
    invTrans = transforms.Compose([transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                               ])

    inv_tensor = invTrans(s)
    return inv_tensor

def imshow(img, image_name):
    img = denormalize(img) # unnormalize
    img = normalizeData(img)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    # plt.show()
    save_path = './viz'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig('%s/%s.jpg'%(save_path, image_name))


def heatmap_show(img, mask, image_name):
    img = denormalize(img) # unnormalize
    img = normalizeData(img)
    img = np.transpose(img.numpy(), (1, 2, 0))
    save_path = './viz'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    mask = np.transpose(mask.numpy(), (1, 2, 0))
    plt.imshow(img)
    plt.imshow(mask, cmap='jet', alpha=0.5)
    plt.axis('off')
    plt.colorbar(cmap='jet', fraction=0.046, pad=0.04)
    plt.savefig('%s/%s.jpg'%(save_path, image_name))

def split_dataset(file_path, random_state=10):
    df = pd.read_csv(file_path)
    # -- Split after uniquing the patient ids so that it does not get split across the different test, dev, test
    pid = list(df['patient_id'].unique())
    random.seed(random_state)
    random.shuffle(pid)
    train_patient_count = round(len(pid) * 0.8)
    not_train = len(pid) - train_patient_count
    # --- Split this remaining equally into dev and test.
    dev_patient_count = round(not_train * 0.5)
    train = df[df['patient_id'].isin(pid[:train_patient_count])]
    dev = df[df['patient_id'].isin(pid[train_patient_count:train_patient_count+dev_patient_count])]
    test = df[df['patient_id'].isin(pid[train_patient_count+dev_patient_count:])]
    return train, dev, test


def nfold_split_dataset(file_path, nFold=5, random_state=1):
    df = pd.read_csv(file_path)
    # -- Split after uniquing the patient ids so that it does not get split across the different test, dev, test
    pid = list(df['patient_id'].unique())
    random.seed(random_state)
    random.shuffle(pid)
    fold_count = round(len(pid)/nFold)+1
    split_dict = {}
    split_list = [pid[i:i+fold_count] for i in range(0, len(pid), fold_count)]
    for K in range(nFold):
        train_ids = []
        for i in range(nFold):
            if i == K:
                test_ids = split_list[i]
            else:
                train_ids.extend(split_list[i])
        test = df[df['patient_id'].isin(test_ids)]
        train = df[df['patient_id'].isin(train_ids)]
        split_dict[K] = {'train':train, 'test':test}
    # write the split to .csv file
    dir_path = file_path.replace("/master_sheet.csv","")
    with open(os.path.join(dir_path,"%d_fold_split.csv"%nFold),'w') as f:
        wr = csv.writer(f)
        wr.writerow(split_list)
    return split_dict

def read_cxrjpg(image_path):
    image = Image.open(image_path).convert("RGB")
    return image

class EyegazeDataset(Dataset):
    def __init__(self, csv_file, image_path_name, class_names, static_heatmap_path=None, heatmap_static_transform=None, image_transform=None, heatmaps_threshold=None):
        self.csv_file = csv_file
        self.path_name = image_path_name
        self.image_transform = image_transform
        self.heatmap_static_transform = heatmap_static_transform
        self.class_names = class_names
        self.static_heatmap_path = static_heatmap_path
        self.heatmaps_threshold = heatmaps_threshold

    def __len__(self):
        return len(self.csv_file)

    def get_image(self, idx):
        # -- Query the index location of the required file
        image_path = os.path.join(self.path_name, self.csv_file['path'].iloc[idx])
        image_path = image_path.replace('.dcm', '.jpg')
        image_pil = read_cxrjpg(image_path)

        ### multi-label
        truth_labels = [self.csv_file[labels].iloc[idx] for labels in self.class_names]
        y_label = np.array(truth_labels, dtype=np.int64).tolist()
        y_label = torch.from_numpy(np.array(y_label)).float()

        ##### one hot label
        one_label = ((y_label == 1.0).nonzero(as_tuple=True)[0]).squeeze()

        ###### attributes
        healthy = 'no_finding__chx'
        attributes = ['atelectasis__chx', 'cardiomegaly__chx', 'consolidation__chx', 'edema__chx', \
                       'enlarged_cardiomediastinum__chx', 'fracture__chx',  'lung_lesion__chx',  \
                       'lung_opacity__chx', 'pleural_effusion__chx',  'pleural_other__chx', \
                       'pneumonia__chx', 'pneumothorax__chx',  'support_devices__chx']
        if self.csv_file[healthy].iloc[idx] == 1:
            attr_labels = [0,0,0,0,0,0,0,0,0,0,0,0,0]
        else:
            attr_labels = [self.csv_file[attr].iloc[idx] for attr in attributes]
            attr_labels = np.asarray(attr_labels, dtype=np.float32).T
            attr_labels[attr_labels == -1.] = 0.0
            # make all the -1 values into nans (0.0) to keep things simple
            attr_labels = np.nan_to_num(attr_labels)
        attr_labels = torch.from_numpy(np.array(attr_labels)).float()

        if self.image_transform:
            image = self.image_transform(image_pil)
        return image_pil, image.float(), one_label, attr_labels

    def num_sort(self, filename):
        not_num = re.compile("\D")
        return int(not_num.sub("", filename))

    def __getitem__(self, idx):
        image_name = self.csv_file['dicom_id'].iloc[idx]
        y_hm = []

        if self.static_heatmap_path:
            heat_path = os.path.join(self.static_heatmap_path, image_name)
            # ground_truth static heatmap
            if not os.path.exists(heat_path + '/heatmap.png'):
                raise FileNotFoundError(f'static heatmaps not found for {heat_path}')
            y_hm_pil = Image.open(heat_path + '/heatmap.png').convert('L')
            y_hm = self.heatmap_static_transform(y_hm_pil)
            if self.heatmaps_threshold:
                y_hm = y_hm > self.heatmaps_threshold
        image_pil, image, y_label, attr_labels = self.get_image(idx)

        y_hm_n = normalizeData(y_hm)
        y_hm_n = y_hm_n > 0.3 # filter with 0.3 after normalization
        y_hm = y_hm * y_hm_n
        gaze_img = image * y_hm_n
        gaze_img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(gaze_img)
        image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
        # # for viz
        # imshow(gaze_img, image_name)
        # exit()

        return image, y_label, image_name, y_hm, gaze_img, attr_labels
