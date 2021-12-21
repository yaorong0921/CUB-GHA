from utils.indices2coordinates import indices2coordinates
from utils.compute_window_nums import compute_window_nums
import numpy as np

CUDA_VISIBLE_DEVICES = '0'

# please use 'gat' to run gat, and 'kfn' to run 'kfn'
model_type = 'gat'

# data input dir
root = '/storage/rong/CUB_200_2011/CUB_200_2011' # pth to the CUB dataset
GHA_dir = 'CUB_GHA' # the dir name of GHA images (the dir should be under the root)

# save model
model_path = './checkpoint'  # pth save path
model_name = 'gat'   # please give a name to the dir for saving the model, e.g. gat or kfn

# The pth path of pretrained model
pretrain_path, GAT_pretrained = './pretrained/resnet50-19c8e357.pth', False

# To use other pretrained backbone in KFN, e.g. GAT pretrained
# pretrain_path, GAT_pretrained = '/checkpoint/gat/best/best.pth', True

# training configs
batch_size = 6
eval_trainset = False  # Whether or not evaluate trainset
save_interval = 1
max_checkpoint_num = 1
end_epoch = 150
init_lr = 0.001
lr_milestones = [50, 100]
lr_decay_rate = 0.1
weight_decay = 1e-4
channels = 2048
input_size = 448
num_classes = 200

# windows info in GAT are followings:
stride = 8
N_list = [2, 3, 4] # change the number of windows in (L,M,S) here.
proposalN = sum(N_list)  # proposal window num
window_side = [128, 192, 256]
iou_threshs = [0.25, 0.25, 0.25]

ratios = [[246, 269], [269, 246],
            [174, 190], [190, 174], [174, 174], [190, 190],
            [123, 134], [134, 123], [123, 123], [134, 134]]

# indice2coordinates
window_nums = compute_window_nums(ratios, stride, input_size)
indices_ndarrays = [np.arange(0,window_num).reshape(-1,1) for window_num in window_nums]
coordinates = [indices2coordinates(indices_ndarray, stride, input_size, ratios[i]) for i, indices_ndarray in enumerate(indices_ndarrays)] # 每个window在image上的坐标
coordinates_cat = np.concatenate(coordinates, 0)
window_milestones = [sum(window_nums[:i+1]) for i in range(len(window_nums))]
window_nums_sum = [0, sum(window_nums[:2]), sum(window_nums[2:6]), sum(window_nums[6:])]

