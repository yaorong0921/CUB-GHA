import torch
from torch import nn
import torch.nn.functional as F
from networks import resnet
from torchvision import models
from config import pretrain_path, N_list, stride,  iou_threshs, coordinates_cat, window_nums_sum, ratios, GAT_pretrained
import numpy as np
from utils.vis import imshow, heatmap_show
import collections

def nms(scores_np, proposalN, iou_threshs, coordinates, IND=None):
    if not (type(scores_np).__module__ == 'numpy' and len(scores_np.shape) == 2 and scores_np.shape[1] == 1):
        raise TypeError('score_np is not right')

    windows_num = scores_np.shape[0]
    indices_coordinates = np.concatenate((scores_np, coordinates), 1)

    if IND is not None:
        indices = IND
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

class APPM(nn.Module):
    def __init__(self):
        super(APPM, self).__init__()

        self.avgpools = [nn.AvgPool2d(ratios[i], stride) for i in range(len(ratios))]

    def forward(self, proposalN, x, ratios, window_nums_sum, N_list, iou_threshs, DEVICE='cuda'):
        batch, channels, _, _ = x.size()
        avgs = [self.avgpools[i](x) for i in range(len(ratios))]

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

class MainNet(nn.Module):
    def __init__(self, proposalN, num_classes, channels):
        super(MainNet, self).__init__()
        self.num_classes = num_classes
        self.proposalN = proposalN
        self.pretrained_model = resnet.resnet50(pretrained=True, pth_path=pretrain_path)
        self.cls_net = nn.Linear(channels, num_classes)
        self.APPM = APPM()

    def forward(self, x, att, epoch, batch_idx, status='test', DEVICE='cuda'):
        fm, embedding, conv5_b = self.pretrained_model(x)
        batch_size, channel_size, side_size, _ = fm.shape
        assert channel_size == 2048

        logits = self.cls_net(embedding)
        coordinates = []
        proposalN_indices, proposalN_windows_scores, window_scores \
            = self.APPM(self.proposalN, att, ratios, window_nums_sum, N_list, iou_threshs, DEVICE)

        if status == "train":
            # window_imgs cls
            window_imgs = torch.zeros([batch_size, self.proposalN, 3, 224, 224]).to(DEVICE)
            for i in range(batch_size):
                coord_tensor = torch.zeros((self.proposalN, 4), dtype=torch.int16)
                for j in range(self.proposalN):
                    [x0, y0, x1, y1] = coordinates_cat[proposalN_indices[i, j]]
                    coord_tensor[j,:] = torch.as_tensor([x0, y0, x1, y1])

                    window_imgs[i:i + 1, j] = F.interpolate(x[i:i + 1, :, x0:(x1 + 1), y0:(y1 + 1)], size=(224, 224),
                                                                mode='bilinear',
                                                                align_corners=True)
                coordinates.append(coord_tensor)
            ###  this is for visualization
            # m=3
            # heatmap_show(x[m,:].cpu(), att[m,:].cpu(), 'overlay')
            # for j in range(self.proposalN):
            #     imshow(window_imgs[m,j].cpu(),str(j))
            # exit()
            window_imgs = window_imgs.reshape(batch_size * self.proposalN, 3, 224, 224) 
            _, window_embeddings, _ = self.pretrained_model(window_imgs.detach())
            proposalN_windows_logits = self.cls_net(window_embeddings)
        else:
            proposalN_windows_logits = torch.zeros([batch_size * self.proposalN, self.num_classes]).to(DEVICE)

        return proposalN_windows_scores, proposalN_windows_logits, proposalN_indices, window_scores, coordinates, logits 

class TwoBranch(nn.Module):
    def __init__(self, num_classes, channels):
        # nn.Module
        super(TwoBranch, self).__init__()
        self.num_classes = num_classes
        model1 = models.resnet50(pretrained=True)
        model2 = models.resnet50(pretrained=True)

## load pretrain model
        if GAT_pretrained:
            new_dict = collections.OrderedDict()
            pretrained_dict = torch.load(pretrain_path)['model_state_dict']
            for k,v in pretrained_dict.items():
                if 'pretrained_model.' in k:
                    name = k[17:]
                    new_dict[name] = v
            model1.load_state_dict(new_dict)
            model2.load_state_dict(new_dict)
#############################
        self.pretrained_model1 = nn.Sequential(*list(model1.children())[:-1])
        self.pretrained_model2 = nn.Sequential(*list(model2.children())[:-1])
        self.cls_net = nn.Linear(channels*2, num_classes)

    def forward(self, x, att, DEVICE='cuda'):
        x1 = self.pretrained_model1(x)
        x2 = self.pretrained_model2(att)
        x1 = torch.flatten(x1, 1)
        x2 = torch.flatten(x2, 1)

        out = torch.cat((x1, x2), dim=-1)
        out = self.cls_net(out)
        return out
