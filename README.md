# ReferIt3D:  Neural Listeners for Fine-Grained 3D Object Identification in Real-World Scenes
[![Website Badge](images/project_website_badge.svg)](https://referit3d.github.io)
[![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg?style=plastic)](https://arxiv.org/abs/1234.56789)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
<!---[![GitHub stars](https://img.shields.io/github/stars/Naereen/StrapDown.js.svg?style=social&label=Star&maxAge=2592000)](https://GitHub.com/Naereen/StrapDown.js/stargazers/)--->
Panos Achlioptas, Ahmed Abdelreheem, Fei Xia, Mohamed Elhoseiny, Leonidas Guibas  
**ECCV 2020 (Oral)**  
  
 
## Introduction
In this work we study the problem of using referential language to identify common objects in real-world 3D scenes. We focus on a challenging setup where the referred object belongs to a fine-grained object class and the underlying scene contains multiple object instances of that class. Due to the scarcity and unsuitability of existent 3D-oriented linguistic resources for this task, we first develop two large-scale and complementary visio-linguistic datasets: i) Sr3D, which contains 83.5K template-based utterances leveraging spatial relations among fine-grained object classes to localize a referred object in a scene, and ii) Nr3D which contains 41.5K natural, free-form, utterances collected by deploying a 2-player object reference game in 3D scenes. Using utterances of either datasets, human listeners can recognize the referred object with high (>86%, 92% resp.) accuracy. By tapping on this data, we develop novel neural listeners that can comprehend object-centric natural language and identify the referred object directly in a 3D scene. Our key technical contribution is designing an approach for combining linguistic and geometric information (in the form of 3D point clouds) and creating multi-modal (3D) neural listeners. We also show that architectures which promote object-to-object communication via graph neural networks outperform less context-aware alternatives, and that fine-grained object classification is a bottleneck for language-assisted 3D object identification.
  
![](images/draft_teaser_gif.gif)

## ReferIt3DNet
![ReferIt3DNet](https://referit3d.github.io/img/method.png)

## Dependencies
```
- gcc5.4 or later
```

- (optional) you are advised to create a new anaconda environment, please use the following commands to create a new one. The code is tested using Python 3.6.9, Pytorch 1.4 and CUDA 10.0
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

- You will need to compile the CUDA layers for [PointNet++](http://arxiv.org/abs/1706.02413):
```Console
    cd external_tools/pointnet2
    python setup.py install
```

## Dataset
You need to download the train/val scans of ScanNet dataset. To do so, Please refer to the [ScanNet Dataset](referit3d/data/scannet/README.md) for more details.

**Nr3D** is found in ```data/language/nr3d/nr3d_tokenized.csv```  
**Sr3D** is found in ```data/language/sr3d/tokenized/sr3d.csv```  
**Sr3D+** is found in ```data/language/sr3d/tokenized/sr3d+.csv```  

Since Sr3d is synthetically made, you change the code/hyper-parameters to create a new one from scratch.  

## Train
* To train on either Nr3d or Sr3d dataset, use the following command
```Console
    cd referit3d/scripts/
    python train_referit3d.py -scannet-file the_processed_scannet_file -referit3d-file dataset_file.csv --log-dir dir_to_log --n-workers 4
```
feel free to change the number of workers to match your #CPUs and RAM size.

* To train nr3d in joint with sr3d, add the following argument
```Console
    --augment-with-sr3d sr3d_dataset_file.csv
``` 

## Eval
* To evaluate on either Nr3d or Sr3d dataset, use the following command
```Console
    cd referit3d/scripts/
    python train_referit3d.py --mode evaluate -scannet-file the_processed_scannet_file -referit3d-file dataset_file.csv --resume-path the_path_to_the_best_model.pth  --n-workers 4 
```
* To evaluate on joint trained model, add the following argument to the above command
```Console
    --augment-with-sr3d sr3d_dataset_file.csv
``` 

## (Optional) Building Datasets [Sr3D, Nr3D] 
- For Sr3D, use directly data/language/sr3d.csv or sr3d+.csv, OR, generate with your own settings: 
    ```Console
    python data_generation/sr3D/create_data.py --arguments
    ```
   The generation with the default settings takes 3 hours and produces an output sr3d_references.csv
   containing the synthetic grounded referential language over Scannet.  
   
- For Nr3D, use directly data/nr3d.csv

then preprocess the Sr3D / Nr3D to be ready to be used in training by runnung the following command
    ```Console
    python scripts/prepare_referential_data.py
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
