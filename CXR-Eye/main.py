import os
import sys
import argparse
import logging
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torchvision import transforms, models
from datetime import datetime
from models.classifier import Two_Branch, Classifier_with_Augmentation
from utils.gradcam_utils import GradCam, visualize_gcam
from utils.dataset import split_dataset, EyegazeDataset, nfold_split_dataset, normalizeData
from utils.utils import proposalN, image_with_boxes
import csv
plt.rcParams['figure.figsize'] = [25, 10]


logging.basicConfig(stream=sys.stdout, level=logging.INFO) #format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s', datefmt='%H:%M:%S',
logger = logging.getLogger('cxr')

def make_parser():
    parser = argparse.ArgumentParser(description='PyTorch CXR Eye Gaze')

    # Data
    parser.add_argument('--data_path', type=str, default='resources/master_sheet.csv', help='Data path')
    parser.add_argument('--image_path', type=str, default='/data/MIMIC/MIMIC-IV/cxr_v2/physionet.org/files/mimic-cxr/2.0.0', help='image_path')
    parser.add_argument('--heatmaps_path', type=str, default='/storage/rong/CXR-JPG/egd-cxr/fixation_heatmaps', help='human attention heatmap path')
    parser.add_argument('--output_dir', type=str, default='checkpoint', help='Output directory')
    parser.add_argument('--class_names', type=list, default=['Normal', 'CHF', 'pneumonia'], help='Label names for classification')
    parser.add_argument('--num_workers', type=int, default=16, help='number of workers')
    parser.add_argument('--resize', type=int, default=224, help='Resizing images')

    # Training
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
    parser.add_argument('--scheduler', default=False, action='store_true', help='[USE] scheduler')
    parser.add_argument('--step_size', type=int, default=5, help='scheduler step size')
    parser.add_argument('--lr_milestones', type=list, default = [50],help='scheduler decay step')
    parser.add_argument('--weight_decay', type=float, default = 1e-4, help='weight decay')

    ## UNET Specific arguments.
    parser.add_argument('--model_type', default='unet', choices=['kfn', 'gat'], help='kfn, gat')
    parser.add_argument('--heatmaps_threshold', type=float, default=None, help='set the threshold value for the heatmap to be used with unet.')
    parser.add_argument('--pretrained_dir', type=str, default=None, help='path to model pretrained with gat')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout')

    # Misc
    parser.add_argument('--gpus', type=str, default='7', help='Which gpus to use, -1 for CPU')
    parser.add_argument('--viz', default=False, action='store_true', help='[USE] Vizdom')
    parser.add_argument('--gcam_viz', default=False, action='store_true', help='[USE] Used for displaying the GradCam results')
    parser.add_argument('--test', default=False, action='store_true', help='[USE] flag for testing only')
    parser.add_argument('--testdir', type=str, default=None, help='model to test [same as train if not set]')
    parser.add_argument('--rseed', type=int, default=42, help='Seed for reproducibility')
    parser.add_argument('--crossval', default=False, action='store_true', help='Use N-fold cross valiation')
    return parser


def load_data(model_type, data_path, image_path, heatmaps_path, input_size, class_names, batch_size, num_workers, rseed, heatmaps_threshold, nfold=None):
    # ImageNet normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if nfold is None:
        train_file, valid_file, test_file = split_dataset(data_path, random_state=rseed)

        image_transform = transforms.Compose([transforms.Resize((input_size, input_size)), transforms.ToTensor()])

        heatmap_static_transform = transforms.Compose([transforms.Resize([input_size, input_size]),
                                                   transforms.Grayscale(num_output_channels=1),
                                                   transforms.ToTensor()])
        static_heatmap_path = heatmaps_path
        train_dataset = EyegazeDataset(train_file, image_path, class_names, static_heatmap_path=static_heatmap_path,
                                   heatmaps_threshold=heatmaps_threshold, heatmap_static_transform=heatmap_static_transform,
                                   image_transform=image_transform)
        valid_dataset = EyegazeDataset(valid_file, image_path, class_names, static_heatmap_path=static_heatmap_path,
                                   heatmaps_threshold = heatmaps_threshold, heatmap_static_transform=heatmap_static_transform,
                                   image_transform=image_transform)
        train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=num_workers)
        valid_dl = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=num_workers)

        test_dataset = EyegazeDataset(test_file, image_path, class_names, static_heatmap_path=static_heatmap_path,
                                  heatmaps_threshold = heatmaps_threshold, heatmap_static_transform=heatmap_static_transform,
                                  image_transform=image_transform)
        test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=32)

        return train_dl, valid_dl, test_dl
    else:
        train_file = nfold['train']
        valid_file = nfold['test']

        image_transform = transforms.Compose([transforms.Resize((input_size, input_size)), transforms.ToTensor()]) #transforms.Normalize(mean=mean, std=std)

        heatmap_static_transform = transforms.Compose([transforms.Resize([input_size, input_size]),
                                                   transforms.Grayscale(num_output_channels=1),
                                                   transforms.ToTensor()])
        static_heatmap_path = heatmaps_path
        train_dataset = EyegazeDataset(train_file, image_path, class_names, static_heatmap_path=static_heatmap_path,
                                   heatmaps_threshold=heatmaps_threshold, heatmap_static_transform=heatmap_static_transform,
                                   image_transform=image_transform)
        valid_dataset = EyegazeDataset(valid_file, image_path, class_names, static_heatmap_path=static_heatmap_path,
                                   heatmaps_threshold = heatmaps_threshold, heatmap_static_transform=heatmap_static_transform,
                                   image_transform=image_transform)
        train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=num_workers)
        valid_dl = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)

        return train_dl, valid_dl, None

def eval_net(model, loader, classifier_criterion, model_type):
    # evaluate networks
    model.eval()
    correct = 0
    cls_loss = 0.0
    for images, labels, idx, y_hm, gaze_img, attributes in (loader):
        images = images.cuda()
        labels = labels.long()
        labels = labels.cuda()
        y_hm = y_hm.cuda()
        gaze_img = gaze_img.cuda()

        with torch.no_grad():
            if model_type == 'gat':
                gaze_img = gaze_img.cuda()
                y_pred, _, _, _, _, coordinates = model(images, gaze_img, status='test') 
            elif model_type == 'kfn':
                y_pred = model(images, gaze_img)

            l_classifier = classifier_criterion(y_pred, labels)
            cls_loss += l_classifier.item()
            pred = y_pred.max(1, keepdim=True)[1]
            correct += pred.eq(labels.view_as(pred)).sum().item()

    model.train()
    return cls_loss/len(loader.dataset), correct/len(loader.dataset)


def train_net(args, model, train_dl, valid_dl, output_model_path, comment):
    # train networks
    if args.viz: writer = SummaryWriter(log_dir=os.path.join(output_model_path, 'log'), comment=comment)
    global_step = 0

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=0.1)
    classifier_criterion = nn.CrossEntropyLoss()
    best_acc = 0.0

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        correct = 0.0
        for i, (images, labels, idx, y_hm, gaze_img, attributes) in enumerate(train_dl):
            images = images.cuda()
            labels = labels.long()
            labels = labels.cuda()
            y_hm = y_hm.cuda()
            if args.model_type == 'gat':
                y_pred, _, proposalN_windows_logits, _, _, coordinates = model(images, y_hm, status='train')
            elif args.model_type == 'kfn':
                gaze_img = gaze_img.cuda()
                y_pred = model(images, gaze_img)
            loss_classifier = classifier_criterion(y_pred, labels)

            if args.model_type == 'gat':
                windowscls_loss = classifier_criterion(proposalN_windows_logits, labels.unsqueeze(1).repeat(1, proposalN).view(-1))
                if epoch < 2:
                    total_loss = loss_classifier
                else:
                    total_loss = loss_classifier + windowscls_loss
                if i == 1:
                    cat_imgs = []
                    for j, coordinate_ndarray in enumerate(coordinates):
                        img = image_with_boxes(images[j], coordinate_ndarray)
                        cat_imgs.append(img)
                    cat_imgs = np.concatenate(cat_imgs, axis=1)
                    writer.add_images('train' + '/' + 'cut_image_with_windows', cat_imgs, epoch, dataformats='HWC')
            else:
                total_loss =  loss_classifier
            epoch_loss += total_loss.item()
            pred = y_pred.max(1, keepdim=True)[1]
            correct += pred.eq(labels.view_as(pred)).sum().item()

            if args.viz:
                writer.add_scalar('Classifier_Loss', loss_classifier.item(), global_step)
                writer.add_scalar('Loss/Train', total_loss.item(), global_step)
                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            global_step += 1
        logging.info(f'Epoch: {epoch+1} Training_Loss: {epoch_loss/len(train_dl.dataset)}')
        logging.info(f'Epoch: {epoch+1} Accuracy: {correct/len(train_dl.dataset)}')

        with torch.no_grad():
            val_loss, val_acc = eval_net(model, valid_dl, classifier_criterion, args.model_type)

        if args.viz:
            writer.add_scalar('Validation_Loss', val_loss, global_step)
            writer.add_scalar('Validation_ACC', val_acc, global_step)
    
        model.train()
        if args.scheduler: scheduler.step()
        try:
            os.makedirs(output_model_path)
            logger.info('Created Checkpoint directory')
        except OSError:
            pass
        if best_acc < val_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), output_model_path + f"/best.pth")
            logger.info(f"Checkpoint saved !")
        logging.info(f'Best Validation_Acc_scores: {best_acc}')
    if args.viz: writer.close()
    return best_acc


def display_gcam(args, val_dl, model_dir, model_name, aux_params):
    # visualize the gradcam of kfn branches
    if args.model_type == 'kfn':
        model = Two_Branch(len(args.class_names),model_type='efficientnet')
        candidate_layers = ['branch1.blocks.6', 'branch2.blocks.6'] 
    else:
        print('Not implemented yet.')
        exit()

    output_weights_name = os.path.join(model_dir, model_name)
    logger.debug(f'Loading Model: {output_weights_name}')
    if len(args.gpus.split(',')) > 1:
        print(f"Using {len(args.gpus.split(','))} GPUs!")
        device_ids = [int(i) for i in args.gpus.split(',')]
        model = nn.DataParallel(model, device_ids=device_ids)
    model.load_state_dict(torch.load(output_weights_name))
    model.cuda()

    gcam = GradCam(model=model, candidate_layers=candidate_layers)
    if args.model_type == 'kfn':
        ## for branch1
        target_layer = ['branch1.blocks.6', 'branch2.blocks.6'] # ['branch1.0.denseblock4.denselayer16', 'branch2.0.denseblock4.denselayer16']
        visualize_gcam(args, model, val_dl, gcam, target_layer=target_layer, model_dir=model_dir)      
    else:
        print('Not implemented yet.')
        exit()
    exit()


if __name__ == '__main__':
    args = make_parser().parse_args()
    random.seed(args.rseed)
    np.random.seed(args.rseed)
    torch.manual_seed(args.rseed)
    cuda = torch.cuda.is_available() and args.gpus != '-1'
    if cuda:
        torch.cuda.manual_seed(args.rseed)
        torch.cuda.manual_seed_all(args.rseed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    torch.cuda.set_device("cuda:"+ args.gpus)
    args.device = torch.device("cuda:"+ args.gpus) if cuda else torch.device('cpu')
    logger.info(torch.cuda.get_device_name(args.device))

    # Create saving dir, all useful variables
    comment_variable = ''
    timestamp = str(datetime.now()).replace(" ", "").split('.')[0]
    for arg in vars(args):
        if arg not in ['data_path', 'heatmaps_path', 'image_path', 'class_names', 'gpus', 'viz', 'device',
                       'test', 'pretrained_dir', 'testdir', 'output_dir', 'num_workers', 'rseed']:
            comment_variable += f'{arg}{str(getattr(args, arg)).replace(" ", "")}_' \
                if arg != 'model_type' else f'{str(getattr(args, arg))}_'
    comment_variable += f'{timestamp}'

    logger.info(f"Comment Variable: {comment_variable}")
    output_model_path = os.path.join(args.output_dir, args.model_type+'_crossv-'+str(args.crossval) + '_rseed'+str(args.rseed))
    logger.info(f"[Arguments]:{args}")

    if not args.crossval:
        # no cross validation is used
        nfold = None
        train_dl, valid_dl, test_dl = load_data(args.model_type, args.data_path, args.image_path, args.heatmaps_path, args.resize,
                                            args.class_names, args.batch_size, args.num_workers, args.rseed, args.heatmaps_threshold, nfold=nfold)
    n_classes = len(args.class_names) # Classifier classes
    n_segments = 1 # Number of segmentation classes
    aux_params = dict(
        pooling='avg',  # one of 'avg', 'max'
        dropout=args.dropout,  # dropout ratio, default is None
        activation=None,  # activation function, default is None
        classes=n_classes,  # define number of output labels
    )

    if not args.test:  # training
        val_acc = 0.0
        if args.crossval:
            logger.info('-- TRAIN THE NETWORK WITH CROSS-VALIDATION --')
            split_dict = nfold_split_dataset(args.data_path, random_state=args.rseed)
            for n in range(5):
                train_dl, valid_dl, test_dl = load_data(args.model_type, args.data_path, args.image_path, args.heatmaps_path, args.resize,
                                            args.class_names, args.batch_size, args.num_workers, args.rseed, args.heatmaps_threshold, nfold=split_dict[n])

                if args.model_type == 'kfn':
                    if args.pretrained_dir is not None:
                        pretrain_path = os.path.join(args.pretrained_dir, '%d_fold'%n, 'best.pth')
                    else:
                        pretrain_path = args.pretrained_dir
                    model = Two_Branch(n_classes, model_type='efficientnet', pretrain_path=pretrain_path)

                elif args.model_type == 'gat':
                    model = Classifier_with_Augmentation(encoder_name='timm-efficientnet-b0', proposalN=proposalN,
                                           aux_params=aux_params).to(device=args.device)
                else:
                    print('Not implemented yet.')
                    exit()

                total_params_net = sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])
                logger.info(f'Number of parameters: {total_params_net} ')
                if len(args.gpus.split(',')) > 1:
                   print(f"Using {len(args.gpus.split(','))} GPUs!")
                   device_ids = [int(i) for i in args.gpus.split(',')]
                   model = nn.DataParallel(model, device_ids=device_ids)
                else:
                   model = model.cuda()
                logger.info(f"Comment:{comment_variable}")
                val_acc += train_net(args, model, train_dl, valid_dl, output_model_path+'/%d_fold'%n, comment_variable)
            avg_acc = val_acc/5
            logger.info(f"Cross Validation Average Acc:{avg_acc}")
        else:
            if args.model_type == 'kfn':
                model = Two_Branch(n_classes, model_type='efficientnet')
            elif args.model_type == 'gat':
                model = Classifier_with_Augmentation(encoder_name='timm-efficientnet-b0', proposalN=proposalN,
                                           aux_params=aux_params).to(device=args.device) 
            else:
                print('Not implemented yet.')
                exit()

            total_params_net = sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])
            logger.info(f'Number of parameters: {total_params_net} ')
            if len(args.gpus.split(',')) > 1:
                print(f"Using {len(args.gpus.split(','))} GPUs!")
                device_ids = [int(i) for i in args.gpus.split(',')]
                model = nn.DataParallel(model, device_ids=device_ids)
            else:
                model = model.cuda()
            logger.info(f"Comment:{comment_variable}")
            train_net(args, model, train_dl, valid_dl, output_model_path, comment_variable)

    else:
        if not args.crossval:
            logger.info('-- TESTING THE NETWORK OUTPUT --')
            model_dir = args.testdir if args.testdir else output_model_path
            model_name = f'best.pth'
            if args.model_type == 'kfn':
                model = Two_Branch(n_classes, model_type='efficientnet', pretrain_path=None)

            elif args.model_type == 'gat':
                    model = Classifier_with_Augmentation(encoder_name='timm-efficientnet-b0', proposalN=proposalN,
                                           aux_params=aux_params).to(device=args.device) 
            else:
                print('Not implemented yet.')
                exit()
            output_weights_name = os.path.join(model_dir, model_name)
            logger.info(f'Loading Model: {output_weights_name}')
            if len(args.gpus.split(',')) > 1:
                print(f"Using {len(args.gpus.split(','))} GPUs!")
                device_ids = [int(i) for i in args.gpus.split(',')]
                model = nn.DataParallel(model, device_ids=device_ids)
            else:
                model = model.cuda()
            model.load_state_dict(torch.load(output_weights_name))
            classifier_criterion = nn.CrossEntropyLoss()
            _, model_acc = eval_net(model, test_dl, classifier_criterion, args.model_type)
            logger.info(f"Test ACC:{model_acc}")
        else:
            #### use validation set in cross-val:
            logger.info('-- TESTING THE NETWORK OUTPUT IN CROSS VALIDATION--')
            if args.model_type == 'kfn':
                    model = Two_Branch(n_classes, model_type='efficientnet', pretrain_path=None)
            elif args.model_type == 'gat':
                    model = Classifier_with_Augmentation(encoder_name='timm-efficientnet-b0',  proposalN=proposalN,
                                            aux_params=aux_params).to(device=args.device)
            else:
                print('Not implemented yet.')
                exit()
            
            split_dict = nfold_split_dataset(args.data_path, random_state=args.rseed)

            acc_val = 0.0
            for n in range(5):
                train_dl, valid_dl, test_dl = load_data(args.model_type, args.data_path, args.image_path, args.heatmaps_path, args.resize,
                                            args.class_names, args.batch_size, args.num_workers, args.rseed, args.heatmaps_threshold, nfold=split_dict[n])

                model_dir = args.testdir if args.testdir else output_model_path
                model_name = 'best.pth'
                output_weights_name = os.path.join(model_dir,'%d_fold'%n, model_name)
                if len(args.gpus.split(',')) > 1:
                    print(f"Using {len(args.gpus.split(','))} GPUs!")
                    device_ids = [int(i) for i in args.gpus.split(',')]
                    model = nn.DataParallel(model, device_ids=device_ids)
                else:
                    model = model.cuda()
                model.load_state_dict(torch.load(output_weights_name))
                logger.info(f'Loading Model: {output_weights_name}')

                classifier_criterion = nn.CrossEntropyLoss()
                _, model_acc = eval_net(model, valid_dl, classifier_criterion, args.model_type)
                acc_val += model_acc
            logger.info(f"Test fold ACC:{acc_val/5}")

        if args.gcam_viz:
            logger.info('-- VISUALIZE THE NETWORK GRADCAM AND HA PREDICTIONS --')
            if not args.crossval: 
                print('Only visualize for n=[1,2,3,4,5] in cross-validation.')
                exit()
            else:
                n = 0
                split_dict = nfold_split_dataset(args.data_path, random_state=args.rseed)
                train_dl, test_dl, _ = load_data(args.model_type, args.data_path, args.image_path, args.heatmaps_path, args.resize,
                                                args.class_names, args.batch_size, args.num_workers, args.rseed, args.heatmaps_threshold, nfold=split_dict[n])
                model_dir = args.testdir if args.testdir else output_model_path
                model_name = f'best.pth'
                display_gcam(args, test_dl, os.path.join(model_dir,'%d_fold'%n), model_name, aux_params)

