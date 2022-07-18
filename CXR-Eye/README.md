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

`main.py`: main script, including functions to train and evaluate.

`utils/dataset.py`: read CXR images together with human attention from CXR-Eye.

`utils/gradcam_utils.py, visualization.py`: functions for visualizing GradCAM of models but not in the training phase. 

`utils/utils.py`: functions for GAT training and others.

`models`: code for our model, GAT and KFN.

`run_train.sh`: to train the model. `run_eval.sh`: to evalate the model.

#### Gaze Augmentation Training
1. Preparation:

    To train the model with GAT, set `--model_type gat` in `run_train.sh`.

    Change the path of `--data_path`, `--image_path` and `--heatmaps_path` where you stored them. 

    If you want to use 5-fold cross validation, please set the flag `--crossval`, otherwise, random train and test splits will be used. 

    If you use `--rseed 1`, you will use the same 5-fold cross validation as in the paper.
    
2. Run the training script:

    Go to the folder CXR-Eye and run `sh run_train.sh` to start training.
    After the training, the models are saved and the final average accuracy is printed. 


3. Run the eval script:

    If you want to run the evaluation only, please go to `run_eval.sh` and set `--test_dir` to the path of saved models and run `sh run_eval.sh`.

#### Knowledge Fusion Network
1. Preparation:

   To train the KFN, set `--model_type kfn` in `run_train.sh`.

    Change the path of `--data_path`, `--image_path` and `--heatmaps_path` where you stored them. 

    If you want to use 5-fold cross validation, please set the flag `--crossval`, otherwise, random train and test splits will be used. 

    If you use `--rseed 1`, you will use the same 5-fold cross validation for each running for fairly comparing different settings (as used in the paper).

    If you want to train KFN with the GAT-pretrained backbone, please set `--pretrained_dir` to the path where gat-trained backbonses are saved, e.g. `./checkpoint/gat_crossv-True_rseed1`. 

2. Run the training script:

    Go to the folder CXR-Eye and run `sh run_train.sh` to start training.
    
    After the training, the models are saved and the final average accuracy is printed. 

3. Run the eval script:

    If you want to run the evaluation only, please go to `run_eval.sh` and set `--test_dir` to the path of saved models. 
    
    For KFN, GradCAM visualization is enable in evaluation if you set the flag `--gcam_viz`.
