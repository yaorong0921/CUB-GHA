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

## Experiment Code
In the folder CUB, you can find the code and instructions for experiments on CUB-200-2011.

In the folder CXR-Eye, you can find code and instructions for experiments on CXR-Eye.


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
