import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy
from collections import OrderedDict


# -- Code modified from source: https://github.com/kazuto1011/grad-cam-pytorch
class _BaseWrapper(object):
    """
    Please modify forward() and backward() depending on your task.
    """
    def __init__(self, model):
        super(_BaseWrapper, self).__init__()
        self.device = next(model.parameters()).device
        self.model = model
        self.handlers = []  # a set of hook function handlers

    def generate(self):
        raise NotImplementedError

    def forward(self, image):
        """
        Simple classification
        """
        self.model.zero_grad()
        self.logits = self.model(image)
        self.probs = F.softmax(self.logits, dim=1)
        return list(zip(*self.probs.sort(0, True)))  # element: (probability, index)

class GradCam(_BaseWrapper):
    def __init__(self, model, candidate_layers=[]):
        super(GradCam, self).__init__(model)
        self.fmap_pool = OrderedDict()
        self.grad_pool = OrderedDict()
        self.candidate_layers = candidate_layers
        # for module in self.model.named_modules():
        #     print(module[0])

        def forward_hook(module, input, output):
            self.fmap_pool[id(module)] = output.detach()


        def backward_hook(module, grad_in, grad_out):
            self.grad_pool[id(module)] = grad_out[0].detach()

        for module in self.model.named_modules():
            if len(self.candidate_layers) == 0 or module[0] in self.candidate_layers:
                self.handlers.append(module[1].register_forward_hook(forward_hook))
                self.handlers.append(module[1].register_backward_hook(backward_hook))

    def find(self, pool, target_layer):
        # --- Query the right layer and return it's value.
        for key, value in pool.items():
            for module in self.model.named_modules():
                # print(module[0], id(module[1]), key)
                if id(module[1]) == key:
                    if module[0] == target_layer:
                        return value
        raise ValueError(f"Invalid Layer Name: {target_layer}")

    def normalize(self, grads):
        l2_norm = torch.sqrt(torch.mean(torch.pow(grads ,2))) + 1e-5
        return grads /l2_norm

    def compute_grad_weights(self, grads):
        grads = self.normalize(grads)
        return F.adaptive_avg_pool2d(grads, 1)


    def generate(self, target_layer):
        fmaps = self.find(self.fmap_pool, target_layer)
        grads = self.find(self.grad_pool, target_layer)
        weights = self.compute_grad_weights(grads)

        gcam = (fmaps[0] * weights[0]).sum(dim=0)
        gcam = torch.clamp(gcam, min=0.0)

        gcam -= gcam.min()
        gcam /= gcam.max()
        return gcam

def compute_gradCAM(probs, labels, gcam, testing_labels, target_layer='layer4'):
    # --- one hot encode this:
    # one_hot = torch.zeros((labels.shape[0], labels.shape[1])).float()
    one_hot = torch.zeros((probs.shape[0], probs.shape[1])).float()
    max_int = torch.max(torch.nn.Sigmoid()(probs), 1)[1]

    if testing_labels:
        for i in range(one_hot.shape[0]):
            one_hot[i][max_int[i]] = 1.0

    else:
        for i in range(one_hot.shape[0]):
            one_hot[i][torch.max(labels, 1)[1][i]] = 1.0

    probs.backward(gradient=one_hot.cuda(), retain_graph=True)
    fmaps = gcam.find(gcam.fmap_pool, target_layer)
    grads = gcam.find(gcam.grad_pool, target_layer)

    weights = torch.nn.functional.adaptive_avg_pool2d(grads, 1)
    gcam = torch.mul(fmaps, weights).sum(dim=1, keepdim=True)
    gcam_out = torch.nn.functional.relu(gcam)
    return probs, gcam_out, one_hot


def get_mask(gcam):
    for i in range(gcam.shape[0]):
        temp_loc = -1
        if gcam[i][:].sum() != 0:
            gcam[i][:] = gcam[i][:]
        else:
            temp_loc = i

        # if temp_loc != -1:
            # print('#--Zero SUM Error for image idx %d--#'%i)

    # gcam = torch.nn.functional.interpolate(gcam, size=(224,224), mode='bilinear', align_corners=False)
    B, C, H, W = gcam.shape
    gcam = gcam.view(B, -1)
    gcam -= gcam.min(dim=1, keepdim=True)[0]
    gcam /= (gcam.max(dim=1, keepdim=True)[0])
    mask = gcam.view(B, C, H, W)

    return mask