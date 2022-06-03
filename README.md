# HA-in-Fine-Grained-Classification

This repo includes the CUB-GHA (Gaze-based Human Attention) dataset and code of the paper ["Human Attention in Fine-grained Classification"](https://arxiv.org/pdf/2111.01628.pdf) accepted to BMVC 2021.

## CUB-GHA Dataset
To get the CUB-GHA (heatmap for each image) as shown in the paper, you can download from [here](https://drive.google.com/drive/folders/1Oc6oLSHO5xELa5Qy2i7hMOF9_wwTzvdb?usp=sharing) (CUB-GHA.zip). Every image is saved under its index, and the index can be found in `images.txt` in CUB_200_2011.

 
If you would like to generate GHA by yourself. You need:
(1) CUB-200-2011, which can be downloaded [here](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html).
(2) some python packages: numpy, matplotlib, scipy, PIL, tqdm.

1. To generate the all fixation points in one heatmap for each image, as shown in the example below, please run the command:
	`python generate_heatmap.py --CUB_dir_path <path_to_CUB> --CUB_GHA_save_path <path_to_save_CUB_GHA> --gaze_file_path ./Fixation.txt`

	![](./examples/all.jpg)

2. To get single fixation heatmaps for each image, as shown in the example below, please run the command. Fixation belonging to one image will be saved under a directory named with its index.
	`python generate_heatmap.py --single_fixation --CUB_dir_path <path_to_CUB> --CUB_GHA_save_path <path_to_save_CUB_GHA> --gaze_file_path ./Fixation.txt`
	
	![](./examples/single.jpg)


	More settings can be found in the comments in the script.
	*Please note that the fixation duration will not effect the fixation heatmaps in this mode.*

 
3. Some comments of `Fixation.txt`:

	In "Fixation.txt", gaze data of each image in CUB can be found.
	Each line contains the following information:

	> img_id, original_img_width, original_img_height, img_width_on_display, img_height_on_display, x_img_on_display, y_img_on_display, x_gaze_0, y_gaze_0, duration_gaze_0, x_gaze_1, y_gaze_1, duration_gaze_1, .... x_gaze_N, y_gaze_N, duration_gaze_N.

	"Fixation.txt" includes all gaze data from five runs of the data collection (after filtering gaze duration <0.1s). Inside the folder "data_5runs", you will find five files and each contains fixation in one run of the collection.


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
  

## CXR-Eye Experiments
#### Requirement:
1. packages: pytorch, numpy, TensorboardX, imageio, pillow, [segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch), pandas.
2. MIMIC-CXR-JPG dataset and CXR-Eye dataset. 
   MIMIC-CXR-JPG can be downloaded from [here](https://physionet.org/content/mimic-cxr-jpg/2.0.0/#methods). To download the CXR-Eye dataset, please check the repo [cxr-eye-gaze](https://github.com/cxr-eye-gaze/eye-gaze-dataset)

   The two datasets organized as:
	```
	CXR-JPG/
		└── egd-cxr/
		      └── fixation_heatmaps
		      └── 1.0.0/
			    └── master_sheet.csv	
		└── files/
		     └── mimic-cxr-2.0.0-metadata.csv, ...
		     └── files/
			      └── p10
			      └── p11
			      └── ...

	```
	where `egd-cxr` is from [cxr-eye-gaze](https://github.com/cxr-eye-gaze/eye-gaze-dataset) and CXR-JPG is from [MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/2.0.0/#methods).

#### Code Structure:
In the folder CXR-Eye, you can find the code for training on CXR-Eye.

`main.py`: main script, including functions to train and evaluate.

`utils/dataset.py`: read CXR images together with human attention from CXR-Eye.

`utils/gradcam_utils.py, visualization.py`: functions for visualizing GradCAM of models but not in the training phase. 

`utils/utils.py`: functions for GAT training and others.

`models`: code for our model, GAT and KFN.

`run_train.sh`: to train the model. `run_eval.sh`: to evalate the model.

#### Gaze Augmentation Training
To train the model with GAT, set `--model_type gat` in `run_train.sh`.

Change the path of `--data_path`, `--image_path` and `--heatmaps_path` where you stored them. 

If you want to use 5-fold cross validation, please set the flag `--crossval`, otherwise, random train and test splits will be used. 

If you use `--rseed 1`, you will use the same 5-fold cross validation as in the paper.

Go to the folder CXR-Eye and run `sh run_train.sh` to start training.
After the training, the models are saved and the final average accuracy is printed. 

If you want to run the evaluation only, please go to `run_eval.sh` and set `--test_dir` to the path of saved models and run `sh run_eval.sh`.

#### Knowledge Fusion Network
To train the KFN, set `--model_type kfn` in `run_train.sh`.

Change the path of `--data_path`, `--image_path` and `--heatmaps_path` where you stored them. 

If you want to use 5-fold cross validation, please set the flag `--crossval`, otherwise, random train and test splits will be used. 

If you use `--rseed 1`, you will use the same 5-fold cross validation for each running for fairly comparing different settings (as used in the paper).

If you want to train KFN with the GAT-pretrained backbone, please set `--pretrained_dir` to the path where gat-trained backbonses are saved, e.g. `./checkpoint/gat_crossv-True_rseed1`. 

Go to the folder CXR-Eye and run `sh run_train.sh` to start training.
After the training, the models are saved and the final average accuracy is printed. 

If you want to run the evaluation only, please go to `run_eval.sh` and set `--test_dir` to the path of saved models. For KFN, GradCAM visualization is enable in evaluation if you set the flag `--gcam_viz`.


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

We thank the following repos:

1.  [GazePointHeatMap](https://github.com/TobiasRoeddiger/GazePointHeatMap) for providing some functions of gaze visualization.

2.  [MMAL-Net](https://github.com/ZF4444/MMAL-Net) for providing functions of training CUB.

3.  [cxr-eye-gaze](https://github.com/cxr-eye-gaze/eye-gaze-dataset) for providing the dataset and functions of training on CXR-eye.
