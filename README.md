# HA-in-Fine-Grained-Classification
This repo includes the CUB-GHA (Gaze-based Human Attention) dataset and code of the paper ["Human Attention in Fine-grained Classification"](https://arxiv.org/pdf/2111.01628.pdf) accepted to BMVC 2021.


## CUB-GHA Dataset
To get the dataset, you need:

(1) CUB-200-2011, which can be downloaded [here](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html).

(2) some python packages: numpy, matplotlib, scipy, PIL, tqdm.

"Fixation.txt" includes all gaze data from 5 runs of the data collection (after filtering gaze duration <0.1s). Inside the directory "data_5runs", you will find 5 files and each stands for 1 run of the collection. 

1. To get the all fixation points in one heatmap for each image, as shown in the example below, you can download from [here](https://drive.google.com/file/d/1EC4H7N_l2UEtMxFZYOhZCBGz29QSVl2Q/view?usp=sharing). 
   
   Every image is saved under its index which can be found in the file `images.txt` in CUB.
   
   Or you can run the command and generate heatmaps:

   `python generate_heatmap.py --CUB_dir_path <path_to_CUB> --CUB_GHA_save_path <path_to_save_CUB_GHA> --gaze_file_path ./Fixation.txt`

    ![](./examples/all.jpg)

2. To get single fixation heatmaps for each image, as shown in the example below, please run the command:

   `python generate_heatmap.py --single_fixation --CUB_dir_path <path_to_CUB> --CUB_GHA_save_path <path_to_save_CUB_GHA> --gaze_file_path ./Fixation.txt`

	![](./examples/single.jpg)
    
     Fixation belonging to one image will be saved under a directory named with its index.
     
     More settings can be found in the comments in the script.
     
    *Please note that the fixation duration will not effect the fixation heatmaps in this mode.*

If you use this dataset in your research, please cite
```
@article{rong2021human,
  title={Human Attention in Fine-grained Classification},
  author={Rong, Yao and Xu, Wenjia and Akata, Zeynep and Kasneci, Enkelejda},
  journal={arXiv preprint arXiv:2111.01628},
  year={2021}
}
```
Contact me (yao.rong@uni-tuebingen.de) if you have any questions or suggestions.

We thank this [repo](https://github.com/TobiasRoeddiger/GazePointHeatMap) for providing some functions of gaze visualization.

