#coding=utf-8
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
import shutil
import time
from config import num_classes, model_name, model_path, lr_milestones, lr_decay_rate, input_size, \
    root, end_epoch, save_interval, init_lr, batch_size, CUDA_VISIBLE_DEVICES, weight_decay, \
    proposalN, channels, model_type
from utils.train_model import train
from utils.auto_laod_resume import auto_load_resume
from networks.model import MainNet, TwoBranch
from dataset import CUB

import os
import numpy as np
import random

# random seed
os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES
torch.manual_seed(1142)
random.seed(1142)
np.random.seed(1142)

def main():

    # load the CUB dataset
    print('Loading CUB trainset')
    trainset = CUB(input_size=input_size, root=root, is_train=True, model_type=model_type)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=8, drop_last=False)
    print('Loading CUB testset')
    testset = CUB(input_size=input_size, root=root, is_train=False, model_type=model_type)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=8, drop_last=False)

    # load the model
    if model_type == 'kfn':
      print('Knowledge Fusion Network')
      model = TwoBranch(num_classes=num_classes, channels=channels)

    else:
      print('Gaze Augmentation Training')
      model = MainNet(proposalN=proposalN, num_classes=num_classes, channels=channels)

    criterion = nn.CrossEntropyLoss()
    parameters = model.parameters()

    # load checkpoint
    save_path = os.path.join(model_path, model_name)
    if os.path.exists(save_path):
        start_epoch, lr = auto_load_resume(model, save_path, status='train')
        assert start_epoch < end_epoch
    else:
        os.makedirs(save_path)
        start_epoch = 0
        lr = init_lr
    # mk dir for the saving the best model
    if not os.path.exists(os.path.join(save_path, 'best')):
      os.makedirs(os.path.join(save_path, 'best'))

    # define optimizers
    optimizer = torch.optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=weight_decay)

    model = model.cuda() 

    scheduler = MultiStepLR(optimizer, milestones=lr_milestones, gamma=lr_decay_rate)

    # save config for this training
    time_str = time.strftime("%Y%m%d-%H%M%S")
    shutil.copy('./config.py', os.path.join(save_path, "{}config.py".format(time_str)))

    # start training
    train(model=model,
          model_type=model_type,
          trainloader=trainloader,
          testloader=testloader,
          criterion=criterion,
          optimizer=optimizer,
          scheduler=scheduler,
          save_path=save_path,
          start_epoch=start_epoch,
          end_epoch=end_epoch,
          save_interval=save_interval)


if __name__ == '__main__':
    main()
