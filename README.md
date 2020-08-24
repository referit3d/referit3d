# ReferIt3D: Neural Listeners for Fine-Grained 3D Object Identification in Real-World Scenes *[ECCV 2020 (Oral)]*  
[![Website Badge](images/project_website_badge.svg)](https://referit3d.github.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
<!--[![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg?style=plastic)](https://arxiv.org/abs/1234.56789)-->
Created by: Panos Achlioptas, Ahmed Abdelreheem, Fei Xia, Mohamed Elhoseiny, Leonidas Guibas  
  
 
## Introduction
This work is based on our ECCV-2020 [paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460409.pdf). There, we proposed the novel task of identifying a 3D object in a real-world scene given discriminative language, created two relevant datasets (Nr3D and Sr3D) and proposed a 3D neural listener (ReferIt3DNet) for solving this task. The bulk of the provided code serves the training & evaluation of ReferIt3DNet in our data. For more information please visit our project's [webpage](https://referit3d.github.io).

![](images/draft_teaser_gif.gif)

## ReferIt3DNet
![ReferIt3DNet](https://referit3d.github.io/img/method.png)

## Code-Dependencies
1. Python 3.x with numpy, pandas, matplotlib (and a few more *common* packages - please see setup.py)
2. Pytorch 1.x

Our code is tested with Python 3.6.9, Pytorch 1.4 and CUDA 10.0, on Ubuntu 14.04.

## Installation
- (recommended) you are advised to create a new anaconda environment, please use the following commands to create a new one. 
```Console
    conda create -n referit3d_env python=3.6.9 cudatoolkit=10.0
    conda activate referit3d_env
    conda install pytorch torchvision -c pytorch
```

- Install the referit3d python package using
```Console
    cd referit3d
    pip install -e .
```

- To use a PointNet++ visual-encoder you need to compile its CUDA layers for [PointNet++](http://arxiv.org/abs/1706.02413):
```Note: To do this compilation also need: gcc5.4 or later.```
```Console
    cd external_tools/pointnet2
    python setup.py install
```

## Dataset

### ScanNet
First you must download the train/val scans of ScanNet if you do not have them locally. To do so, please refer to the [ScanNet Dataset](referit3d/data/scannet/README.md) for more details.

### Our Linguistic Data
* **Nr3D** you can dowloaded Nr3D [here](https://drive.google.com/file/d/1qswKclq4BlnHSGMSgzLmUu8iqdUXD8ZC/view?usp=sharing) (10.7MB)
* **Sr3D / Sr3D+** you can dowloaded Sr3D/Sr3D+ [here](https://drive.google.com/drive/folders/1DS4uQq7fCmbJHeE-rEbO8G1-XatGEqNV?usp=sharing) (19MB / 20MB)

Since Sr3d is a synthetic dataset, you can change the hyper-parameters to create a version customized to your needs. please see ``referit3d/data_generation/sr3d/``  

## Training
* To train on either Nr3d or Sr3d dataset, use the following commands
```Console
    cd referit3d/scripts/
    python train_referit3d.py -scannet-file the_processed_scannet_file -referit3d-file dataset_file.csv --log-dir dir_to_log --n-workers 4
```
feel free to change the number of workers to match your #CPUs and RAM size.

* To train nr3d in joint with sr3d, add the following argument
```Console
    --augment-with-sr3d sr3d_dataset_file.csv
``` 

## Evaluation
* To evaluate on either Nr3d or Sr3d dataset, use the following commands
```Console
    cd referit3d/scripts/
    python train_referit3d.py --mode evaluate -scannet-file the_processed_scannet_file -referit3d-file dataset_file.csv --resume-path the_path_to_the_best_model.pth  --n-workers 4 
```
* To evaluate on joint trained model, add the following argument to the above command
```Console
    --augment-with-sr3d sr3d_dataset_file.csv
``` 

## Pretrained model
you can download a pretrained ReferIt3DNet model on Nr3D [here](https://drive.google.com/drive/folders/1v50Bwq224Cj4Y4h-OX8mzDaKCQzMeFLl?usp=sharing). please extract the zip file and then copy the extracted folder to referit3d/log folder. you can run the following the command to evaluate:
```
cd referit3d/scripts
python train_referit3d.py --mode evaluate -scannet-file path_to_keep_all_points_00_view_with_global_scan_alignment.pkl  -referit3D-file path_to_nr3d.csv  --resume-path ../log/pretrained_nr3d/checkpoints/best_model.pth
```

## LeaderBoard
   Coming soon!

## Citation
```
@article{achlioptas2020referit_3d,
    title={ReferIt3D: Neural Listeners for Fine-Grained 3D Object Identification in Real-World Scenes},
    author={Achlioptas, Panos and Abdelreheem, Ahmed and Xia, Fei and Elhoseiny, Mohamed and Guibas, Leonidas},
    journal={16th European Conference on Computer Vision (ECCV)},
    year={2020}
}
```

## License
The code is licensed under MIT license.  
Panos Achlioptas, Ahmed Abdelreheem, Fei Xia, Mohamed Elhoseiny, Leonidas Guibas
