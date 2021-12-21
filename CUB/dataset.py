import numpy as np
import imageio
import os
from PIL import Image
from torchvision import transforms
import torch
import random
from config import GHA_dir

def get_image_dict(file):
    image2index = {}
    f = open(file, 'r')
    for item in f.readlines():
        item_list = item.split(" ")
        image_name = item_list[1].strip("\n")
        image2index[image_name] = item_list[0]
    return image2index
def normalizeData(data):
    eps = 1e-07
    return (data - torch.min(data)) / (torch.max(data) - torch.min(data))

class CUB():
    def __init__(self, input_size, root, is_train=True, data_len=None, model_type='kfn'):
        self.input_size = input_size
        self.root = root
        self.is_train = is_train
        self.model_type = model_type
        self.img_dict = get_image_dict(os.path.join(self.root, "images.txt"))
        img_txt_file = open(os.path.join(self.root, 'images.txt'))
        label_txt_file = open(os.path.join(self.root, 'image_class_labels.txt'))
        train_val_file = open(os.path.join(self.root, 'train_test_split.txt'))
        img_name_list = []
        for line in img_txt_file:
            img_name_list.append(line[:-1].split(' ')[-1])
        label_list = []
        for line in label_txt_file:
            label_list.append(int(line[:-1].split(' ')[-1]) - 1)
        train_test_list = []
        for line in train_val_file:
            train_test_list.append(int(line[:-1].split(' ')[-1]))

        train_file_list = [x for i, x in zip(train_test_list, img_name_list) if i]
        test_file_list = [x for i, x in zip(train_test_list, img_name_list) if not i]

        if self.is_train:
            self.train_img = [os.path.join(self.root, 'images', train_file) for train_file in
                              train_file_list[:data_len]]
            self.train_label = [x for i, x in zip(train_test_list, label_list) if i][:data_len]
        if not self.is_train:
            self.test_img = [os.path.join(self.root, 'images', test_file) for test_file in
                             test_file_list[:data_len]]
            self.test_label = [x for i, x in zip(train_test_list, label_list) if not i][:data_len]

    def __getitem__(self, index):
        if self.model_type == 'kfn':
            if self.is_train:
                img, target = imageio.imread(self.train_img[index]), self.train_label[index]
                if len(img.shape) == 2:
                    img = np.stack([img] * 3, 2)
                img = Image.fromarray(img, mode='RGB')

                # compute scaling
                height, width = img.height, img.width
                height_scale = self.input_size / height
                width_scale = self.input_size / width

                #### add attention maps
                img_path = self.train_img[index].split('images/')[1]
                att_path = os.path.join(self.root, GHA_dir, '%s.jpg'%self.img_dict[img_path])
                img_att = Image.open(att_path).convert("L").resize((width, height))

                img_mask = transforms.ToTensor()(img_att)
                img_ori = transforms.ToTensor()(img)

                img_pil = img_ori * img_mask
                img_pil = transforms.ToPILImage()(img_pil)
                transform = transforms.Compose([transforms.Resize(448), 
                                     transforms.RandomCrop(448), 
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                     ])

                img_att= transform(img_pil)
                img = transform(img)
                scale = torch.tensor([height_scale, width_scale])
            else:
                img, target = imageio.imread(self.test_img[index]), self.test_label[index]
                if len(img.shape) == 2:
                    img = np.stack([img] * 3, 2)
                img = Image.fromarray(img, mode='RGB')

                # compute scaling
                height, width = img.height, img.width
                height_scale = self.input_size / height
                width_scale = self.input_size / width

                #### add attention maps
                img_path = self.test_img[index].split('images/')[1]
                att_path = os.path.join(self.root, GHA_dir, '%s.jpg'%self.img_dict[img_path])
                img_att = Image.open(att_path).convert("L").resize((width, height))

                img_mask = transforms.ToTensor()(img_att)
                img_ori = transforms.ToTensor()(img)

                img_pil = img_ori * img_mask
                img_pil = transforms.ToPILImage()(img_pil)
                transform = transforms.Compose([transforms.Resize(448), 
                                     transforms.CenterCrop(448),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                     ])
                img_att= transform(img_pil)
                img = transform(img)
                scale = torch.tensor([height_scale, width_scale])
        else:
            if self.is_train:
                img, target = imageio.imread(self.train_img[index]), self.train_label[index]
                if len(img.shape) == 2:
                    img = np.stack([img] * 3, 2)
                img = Image.fromarray(img, mode='RGB')

                # compute scaling
                height, width = img.height, img.width
                height_scale = self.input_size / height
                width_scale = self.input_size / width

                #### add attention maps
                img_path = self.train_img[index].split('images/')[1]
                att_path = os.path.join(self.root, GHA_dir, '%s.jpg'%self.img_dict[img_path])
                img_att = Image.open(att_path).convert("L").resize((width, height))

                img = transforms.Resize(self.input_size)(img) 
                img_att = transforms.Resize(self.input_size)(img_att) 
                #########
                crop = transforms.RandomCrop(448)
                params = crop.get_params(img, output_size=[448,448])
                img = transforms.functional.crop(img, *params)
                img_att = transforms.functional.crop(img_att, *params)

                if random.random() > 0.5:
                    img = transforms.functional.hflip(img)
                    img_att = transforms.functional.hflip(img_att)

                img = transforms.ColorJitter(brightness=0.2, contrast=0.2)(img)

                img = transforms.ToTensor()(img)
                img_att = transforms.ToTensor()(img_att)
                img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
                img_att = normalizeData(img_att)

            else:
                img, target = imageio.imread(self.test_img[index]), self.test_label[index]
                if len(img.shape) == 2:
                    img = np.stack([img] * 3, 2)
                img = Image.fromarray(img, mode='RGB')

                # compute scaling
                height, width = img.height, img.width
                height_scale = self.input_size / height
                width_scale = self.input_size / width

                #### add attention maps
                img_path = self.test_img[index].split('images/')[1]
                att_path = os.path.join(self.root, GHA_dir, '%s.jpg'%self.img_dict[img_path])
                img_att = Image.open(att_path).convert("L").resize((width, height))

                img = transforms.Resize(self.input_size)(img) 
                img_att = transforms.Resize(self.input_size)(img_att) #
                img = transforms.CenterCrop(448)(img)
                img_att = transforms.CenterCrop(448)(img_att)
                img = transforms.ToTensor()(img)
                img_att = transforms.ToTensor()(img_att)
                img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
                img_att = normalizeData(img_att)

        return img, target, img_att

    def __len__(self):
        if self.is_train:
            return len(self.train_label)
        else:
            return len(self.test_label)
