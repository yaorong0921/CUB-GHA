## CUB Experiments
#### Requirement:
1. packages: pytorch, numpy, TensorboardX, tqdm, imageio, pillow
2. CUB_200_2011 and CUB_GHA datasets organized as:
	```
	CUB_200_2011/
		└── images/ 
		└── CUB_GHA/
	```

3. Download an ImageNet-pretrained model from the folder `pretrained` from [here](https://drive.google.com/drive/folders/1Oc6oLSHO5xELa5Qy2i7hMOF9_wwTzvdb?usp=sharing).

#### Code Structure:
In the folder CUB, you can find the code for training on CUB.

`train.py`: main script to train.

`dataset.py`: read CUB images together with human attention from CUB-GHA.

`utils`: functions for training, evaluating the model, as well as for GAT training/ visualization. `utils/gradcam.py, vis.py` is used for visualizing GradCAM of models but not in the training phase.

`networks`: code for our model, GAT and KFN.

`pretrained`: the pretrained model is stored here.

`config.py`: settings for training. More details see below.

#### Gaze Augmentation Training
1. Set the following parameters in the `config.py`:
	```
	model_type = 'gat'
	root = '/storage/CUB_200_2011/' # pth to the CUB dataset
	GHA_dir = 'CUB_GHA' # the dir name of GHA images (the dir should be under the root)
	model_path = './checkpoint' # dir pth save path
	model_name = 'gat' # please give a name to the dir for saving the model, e.g. gat or kfn
	pretrain_path, GAT_pretrained = './pretrained/resnet50-19c8e357.pth', False
	batch_size = 6
	```
2. Go to the folder CUB and run:
		`python train.py`

3. A ResNet-50 trained with GAT (accuracy 88.00%) can be found in the folder `models` from [here](https://drive.google.com/drive/folders/1Oc6oLSHO5xELa5Qy2i7hMOF9_wwTzvdb?usp=sharing).

#### Knowledge Fusion Network
1. Set the following parameters in the `config.py`:
	```
	model_type = 'kfn'
	root = '/storage/CUB_200_2011/' # pth to the CUB dataset
	GHA_dir = 'CUB_GHA' # the dir name of GHA images (the dir should be under the root)
	model_path = './checkpoint' # dir pth save path
	model_name = 'kfn' # please give a name to the dir for saving the model, e.g. gat or kfn
	pretrain_path, GAT_pretrained = './pretrained/resnet50-19c8e357.pth', False
	batch_size = 8
	```
	To train KFN with the GAT-pretrained backbone, please change `pretrain_path` and set `GAT_pretrained`to True , e.g.:
	```
	pretrain_path, GAT_pretrained = '/checkpoint/gat/best/best.pth', True
	```
2. Go to the folder CUB and run:
		`python train.py`

3. A KFN (two branches of ResNet-50) using GAT-pretrained backbone (accuracy 88.66%) can be found in the folder `models` from [here](https://drive.google.com/drive/folders/1Oc6oLSHO5xELa5Qy2i7hMOF9_wwTzvdb?usp=sharing).   


## 
If you use the CUB-GHA dataset or code in this repo in your research, please cite

```
@article{rong2021human,
title={Human Attention in Fine-grained Classification},
author={Rong, Yao and Xu, Wenjia and Akata, Zeynep and Kasneci, Enkelejda},
journal={arXiv preprint arXiv:2111.01628},
year={2021}
}
```

Contact me (yao.rong@uni-tuebingen.de) if you have any questions or suggestions.
