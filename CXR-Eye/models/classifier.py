from typing import Optional
import torch
import torch.nn as nn
from torch.autograd import Variable
from segmentation_models_pytorch.base import ClassificationHead, SegmentationHead
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import initialization as init
import torchvision
import torch.nn.functional as F
from utils.utils import N_list, stride, iou_threshs, coordinates_cat, window_nums_sum, ratios, nms
import numpy as np
from utils.dataset import heatmap_show, imshow
import collections

class APPM(nn.Module):
    # function for attention area proposal for GAT
    def __init__(self):
        super(APPM, self).__init__()
        # self.avgpools = [nn.AvgPool2d(ratios[i], 1) for i in range(len(ratios))]
        self.avgpools = [nn.AvgPool2d(ratios[i], stride) for i in range(len(ratios))]

    def forward(self, proposalN, x, ratios, window_nums_sum, N_list, iou_threshs, DEVICE='cuda'):
        batch, channels, _, _ = x.size()
        avgs = [self.avgpools[i](x) for i in range(len(ratios))]

        # feature map sum
        # fm_sum = [torch.sum(avgs[i], dim=1) for i in range(len(ratios))]

        all_scores = torch.cat([avgs[i].view(batch, -1, 1) for i in range(len(ratios))], dim=1)
        windows_scores_np = all_scores.data.cpu().numpy()
        window_scores = torch.from_numpy(windows_scores_np).to(DEVICE).reshape(batch, -1)

        # nms
        proposalN_indices = []
        for i, scores in enumerate(windows_scores_np):
            indices_results = []
            for j in range(len(window_nums_sum)-1):
                indices_results.append(nms(scores[sum(window_nums_sum[:j+1]):sum(window_nums_sum[:j+2])], proposalN=N_list[j], iou_threshs=iou_threshs[j],
                                           coordinates=coordinates_cat[sum(window_nums_sum[:j+1]):sum(window_nums_sum[:j+2])]) + sum(window_nums_sum[:j+1]))
            # indices_results.reverse()
            proposalN_indices.append(np.concatenate(indices_results, 1))   # reverse

        proposalN_indices = np.array(proposalN_indices).reshape(batch, proposalN)
        proposalN_indices = torch.from_numpy(proposalN_indices).to(DEVICE)
        proposalN_windows_scores = torch.cat(
            [torch.index_select(all_score, dim=0, index=proposalN_indices[i]) for i, all_score in enumerate(all_scores)], 0).reshape(
            batch, proposalN)

        return proposalN_indices, proposalN_windows_scores, window_scores


class Classifier_with_Augmentation(nn.Module):
    # GAT network
    def __init__(self,
                 encoder_name: str = "resnet34",
                 encoder_depth: int = 5,
                 encoder_weights: str = "imagenet",
                 in_channels: int = 3,
                 proposalN: int = 7,
                 aux_params: Optional[dict] = None,
                 ):
        super().__init__()
        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )
        self.classification_head = ClassificationHead(
            in_channels=self.encoder.out_channels[-1], **aux_params
        )
        self.num_classes = aux_params['classes']
        self.name = "c-{}".format(encoder_name)
        self.proposalN = proposalN
        init.initialize_head(self.classification_head)

        self.APPM = APPM()

    def forward(self, x, att, status='test', DEVICE='cuda'):
        batch_size, _,_,_ = x.shape
        ### Pass model through the encoder and the classifier part of it.
        features = self.encoder(x)
        labels = self.classification_head(features[-1])
        proposalN_indices, proposalN_windows_scores, window_scores \
            = self.APPM(self.proposalN, att, ratios, window_nums_sum, N_list, iou_threshs, DEVICE)
        coordinates = []

        if status == "train":
            # window_imgs cls
            window_imgs = torch.zeros([batch_size, self.proposalN, 3, 112, 112]).to(DEVICE)
            for i in range(batch_size):
                coord_tensor = torch.zeros((self.proposalN, 4), dtype=torch.int16)
                for j in range(self.proposalN):
                    [x0, y0, x1, y1] = coordinates_cat[proposalN_indices[i, j]]
                    coord_tensor[j,:] = torch.as_tensor([x0, y0, x1, y1])
                    window_imgs[i:i + 1, j] = F.interpolate(x[i:i + 1, :, x0:(x1 + 1), y0:(y1 + 1)], size=(112, 112),
                                                                mode='bilinear',
                                                                align_corners=True) 
                coordinates.append(coord_tensor)
            # # viz
            # m=-1
            # heatmap_show(x[m,:].cpu(), att[m,:].cpu(), 'overlay')
            # for j in range(self.proposalN):
            #     imshow(window_imgs[m,j].cpu(),str(j))
            # exit()

            window_imgs = window_imgs.reshape(batch_size * self.proposalN, 3, 112, 112)  # [N*4, 3, 224, 224]
            window_embeddings = self.encoder(window_imgs.detach())  # [N*4, 2048]
            proposalN_windows_logits = self.classification_head(window_embeddings[-1])  # [N* 4, 200]
        else:
            proposalN_windows_logits = torch.zeros([batch_size * self.proposalN, self.num_classes]).to(DEVICE)
        # -- For compatibility with the rest of the code. Output a zero mask region.
        return labels, proposalN_windows_scores, proposalN_windows_logits, proposalN_indices, \
               window_scores, coordinates



class Two_Branch(nn.Module):
    # KFN model
    def __init__(self, n_classes, model_type='densenet', pretrain_path=None):
        super(Two_Branch, self).__init__()
        # densenet and resnet are not trained or evluated in the main script
        if model_type == 'densenet':
            branch1 = torchvision.models.densenet121(pretrained=True)
            branch2 = torchvision.models.densenet121(pretrained=True)
            num_ftrs = branch1.classifier.in_features

            if pretrain_path is not None:
                branch1.classifier = nn.Linear(num_ftrs, 13)
                branch1.load_state_dict(torch.load(pretrain_path), strict=False)
                branch2.classifier = nn.Linear(num_ftrs, 13)
                branch2.load_state_dict(torch.load(pretrain_path), strict=False)
            self.branch1 = nn.Sequential(*list(branch1.children())[:-1])
            self.branch2 = nn.Sequential(*list(branch2.children())[:-1])

        elif model_type == 'resnet':
            branch1 = torchvision.models.resnet50(pretrained=True)
            branch2 = torchvision.models.resnet50(pretrained=True)
            num_ftrs = branch1.fc.in_features

            if pretrain_path is not None:
                branch1.fc = nn.Linear(num_ftrs, 13)
                branch1.load_state_dict(torch.load(pretrain_path), strict=False)
                branch2.fc = nn.Linear(num_ftrs, 13)
                branch2.load_state_dict(torch.load(pretrain_path), strict=False)

            self.branch1 = nn.Sequential(*list(branch1.children())[:-1])
            self.branch2 = nn.Sequential(*list(branch2.children())[:-1])

        # efficient net is used in training and evaluatioin.
        elif model_type == 'efficientnet':
            if pretrain_path is None:
                self.branch1 = get_encoder(
                    'timm-efficientnet-b0',
                    in_channels=3,
                    depth=5,
                    weights="imagenet")
                self.branch2 = get_encoder(
                    'timm-efficientnet-b0',
                    in_channels=3,
                    depth=5,
                    weights="imagenet")
                num_ftrs = self.branch1.out_channels[-1]
            else:
                self.branch1 = get_encoder(
                    'timm-efficientnet-b0',
                    in_channels=3,
                    depth=5)
                # weights="imagenet")
                self.branch2 = get_encoder(
                    'timm-efficientnet-b0',
                    in_channels=3,
                    depth=5)
                # weights="imagenet")
                num_ftrs = self.branch1.out_channels[-1]
                new_dict = collections.OrderedDict()
                pretrained_dict = torch.load(pretrain_path)
                for k,v in pretrained_dict.items():
                    if 'encoder.' in k:
                        name = k[8:]
                        new_dict[name] = v

                #### add the redundant key 'classifier.bias' and 'classifier.weight' into state_dict
                redundant_keys = {"classifier.bias": None, "classifier.weight": None}
                new_dict.update(redundant_keys)
                self.branch1.load_state_dict(new_dict, strict=False)
                new_dict.update(redundant_keys)
                self.branch2.load_state_dict(new_dict, strict=False)
                # for param in self.branch1.parameters():
                #     param.requires_grad = False

        self.classifier = nn.Linear(num_ftrs*2, n_classes)

    def forward (self, x, heatmap):
        x1 = self.branch1(x)[-1]
        x2 = self.branch2(heatmap)[-1]

        x1 = F.relu(x1, inplace=True)
        x1 = F.adaptive_avg_pool2d(x1, (1, 1))
        x1 = torch.flatten(x1, 1)
        x2 = F.relu(x2, inplace=True)
        x2 = F.adaptive_avg_pool2d(x2, (1, 1))
        x2 = torch.flatten(x2, 1)

        out = torch.cat((x1,x2), dim=-1)
        # out = self.dropout(out)
        out = self.classifier(out)

        return out

