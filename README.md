# TSGM: Topological Semantic Graph Memory

This repository contains a Pytorch implementation of our CoRL 2022 **<span style="color: rgb(255, 95, 0)">oral</span>** paper:

### *[Topological Semantic Graph Memory for Image Goal Navigation](https://arxiv.org/pdf/2209.08274.pdf)* <br>
Nuri Kim, Obin Kwon, Hwiyeon Yoo, Yunho Choi, Jeongho Park, Songhwai Oh <br>
Seoul National University

Project website: [https://bareblackfoot.github.io/TopologicalSemanticGraphMemory](https://bareblackfoot.github.io/TopologicalSemanticGraphMemory/)

## Abstract
This work proposes an approach to incrementally collect a landmark-based semantic graph memory and use the
collected memory for image goal navigation.
Given a target image to search, an embodied robot utilizes the semantic memory to find the target in an unknown
environment.
We present a method for incorporating object graphs into topological graphs, called
<span style="font-weight:bold; font-style: italic">Topological Semantic Graph Memory (TSGM)</span>.
Although TSGM does not use position information, it can estimate 3D spatial topological information about objects.
<br>
<br>TSGM consists of <br>
(1) <span style="color: rgb(37, 181, 210)">Graph builder</span> that takes the observed RGB-D image to construct
a topological semantic graph. <br>
(2) <span style="color: rgba(132, 37, 210, 0.699)">Cross graph mixer</span> that takes the collected memory to
get contextual information. <br>
(3) <span style="color: rgba(210, 37, 77, 0.535)">Memory decoder</span> that takes the contextual memory as an
input to find an action to the target.<br>
<br>
On the task of an image goal navigation, TSGM significantly outperforms competitive baselines by +5.0-9.0% on
the success rate and +7.0-23.5% on SPL, which means that the TSGM finds <span  style="font-style: italic">efficient</span> paths.

## Demonstration

To visualize the TSGM generation, run the jupyter notebook [build_tsgm_demo](demo/build_tsgm_demo.ipynb).
This notebook will show the online TSGM generation during *w/a/s/d control* on the simulator.
The rendering window will show the generated TSGM and the observations as follows:
![tsgm_demo](demo/tsgm_demo.gif)
Note that the top-down map and pose information are only used for visualization, not for the graph generation. 

To check the effectiveness of the object encoder, run the jupyter notebook [object_encoder](demo/object_encoder.ipynb).

## Installation
The source code is developed and tested in the following setting. 
- Python 3.7
- pytorch 1.10
- detectron2
- habitat-sim 0.2.1
- habitat 0.2.1

Please refer to [habitat-sim](https://github.com/facebookresearch/habitat-sim.git) and [habitat-lab](https://github.com/facebookresearch/habitat-lab.git) for installation.

To start, we prefer creating the environment using conda:

```
conda env create -f environment.yml
conda activate tsgm
conda install habitat-sim==0.2.1 withbullet headless -c conda-forge -c aihabitat
cd 
mkdir programs
cd programs
git clone --branch stable https://github.com/facebookresearch/habitat-lab.git habitat-lab-v21
cd habitat-lab-v21
git checkout tags/v0.2.1
pip install -e .
conda activate tsgm
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
```

### Gibson Env Setup

Most of the scripts in this code build the environments assuming that the **gibson dataset** is in **habitat-lab/data/** folder.

The recommended folder structure of habitat-lab:
```
habitat-lab 
  └── data
      └── datasets
      │   └── pointnav
      │       └── gibson
      │           └── v1
      │               └── train
      │               └── val
      └── scene_datasets
          └── gibson
              └── *.glb, *.navmeshs  
```

## Download Data

You can download the whole data [here](https://mysnu-my.sharepoint.com/:f:/g/personal/blackfoot_seoul_ac_kr/EoYVFXO2_3lNhho_7LdmAU4B9A669tgjNbSd0ukc_AKXtQ?e=eRheUZ).


## Creating Datasets
1. Data Generation for Imitation Learning 
    ```
    python collect_il_data.py --ep-per-env 200 --num-procs 4 --split train --data-dir IL_data/gibson
    ```
    This will generate the data for imitation learning. (takes around ~24hours)
    You can find some examples of the collected data in *IL_data/gibson* folder, and look into them with  *show_IL_data.ipynb*.
    You can also download the collected il data from [here](https://mysnu-my.sharepoint.com/:f:/g/personal/blackfoot_seoul_ac_kr/EkGTdtVgaMVCvPMHNDjsxlcBoSN2wwzn83gXeF7vT2_Dfg?e=4lBlfZ).

2. Collect Topological Semantic Graph for Imitation Learning 
    ```
    python collect_graph.py ./configs/TSGM.yaml --data-dir IL_data/gibson --record-dir IL_data/gibson_graph --split train --num-procs 16
    ```
    This will generate the graph data for training the TSGM model. (takes around ~3hours)
    You can find some examples of the collected graph data in *IL_data/gibson_graph* folder, and look into them with  *show_graph_data.ipynb*.
    You can also download the collected graph data from [here](https://mysnu-my.sharepoint.com/:f:/g/personal/blackfoot_seoul_ac_kr/EmvaMrQID5NKoQ7SA04eu-gBSIgiDESRznpR7qLw2zjmJQ?e=Ed2alj).

## Training
1. Imitation Learning
    ```
    python train_il.py --policy TSGMPolicy --config configs/TSGM.yaml --version exp_name --data-dir IL_data/gibson --prebuild-path IL_data/gibson_graph
    ```
    This will train the imitation learning model. The model will be saved in *./checkpoints/exp_name*.

2. Reinforcement Learning
The reinforcement learning code is highly based on [habitat-lab/habitat_baselines](https://github.com/facebookresearch/habitat-lab/tree/master/habitat_baselines).
To train the agent with reinforcement learning (PPO), run:
    ```
    python train_rl.py --policy TSGMPolicy --config configs/TSGM.yaml --version exp_name --diff hard --use-detector --strict-stop --task imggoalnav --gpu 0,1
    ```
    This will train the agent with the imitation learning model in *./checkpoints/exp_name*.
    The trained model will be saved in *./checkpoints/exp_name*.

## Evaluation
To evaluate the trained model, run:
```
python evaluate.py --config configs/TSGM.yaml --version version_name --diff hard --gpu 0
```
This will evaluate the trained model in *./data/checkpoints/{$version_name}_{$task}*.

Or, you can evaluate the pretrained model with:
```
python evaluate.py --config configs/TSGM.yaml --version version_name --diff hard --eval-ckpt ./data/best_ckpts/tsgm_rl.pth --gpu 0
```

### Results

Expected results for TSGM from running the code

|  Model  |                              Test set                               | Easy (SR) | Easy (SPL) | Medium (SR) | Medium (SPL) | Hard (SR) | Hard (SPL) | Overall (SR) | Overall (SPL) |
|:-------:|:-------------------------------------------------------------------:|:---------:|:----------:|:-----------:|:------------:|:---------:|:----------:|:------------:|:-------------:|
| TSGM-IL |      [VGM](https://rllab-snu.github.io/projects/vgm/doc.html)       |   76.76   |   59.54    |    72.99    |    53.67     |   63.16   |   45.21    |    70.97     |     52.81     |
| TSGM-RL |      [VGM](https://rllab-snu.github.io/projects/vgm/doc.html)       |   91.86   |   85.25    |    82.03    |    68.10     |   70.30   |   49.98    |     81.4     |     67.78     |
| TSGM-RL |         [NRNS-straight](https://github.com/meera1hahn/NRNS)         |   94.40   |   92.19    |    92.60    |    84.35     |   70.35   |   62.85    |    85.78     |     79.80     |
| TSGM-RL |          [NRNS-curved](https://github.com/meera1hahn/NRNS)          |   93.60   |   91.04    |    89.70    |    77.88     |   64.20   |   55.03    |    82.50     |     74.13     |
| TSGM-RL |  [Meta](https://github.com/facebookresearch/image-goal-nav-dataset) |   90.79   |   86.93    |    85.43    |    72.89     |   63.43   |   56.28    |    79.88     |     72.03     |

### Visualize the Results
To visualize the TSGM from the recorded output from the evaluate (test with --record 3), please run the following command:
```
python visualize_tsgm.py --config-file configs/TSGM.yaml --eval-ckpt <checkpoint_path>
```

We release pre-trained models from the experiments in our paper:

|      Method       |                    Train                    |       Checkpoints       |
|:-----------------:|:-------------------------------------------:|:-----------------------:|
|       TSGM        |             Imitation Learning              |     [tsgm_il.pth](https://mysnu-my.sharepoint.com/:u:/g/personal/blackfoot_seoul_ac_kr/EfCG8qKkuIFCjQrj2LIqIKYBKXuJ1NV2dzsrV9HuXIsADQ?e=abto3d)     |
|       TSGM        | Imitation Learning + Reinforcement Learning |     [tsgm_rl.pth](https://mysnu-my.sharepoint.com/:u:/g/personal/blackfoot_seoul_ac_kr/EbYjg9RZoH5JoZvr941A2mMBvwlGyJAexdychHVQUAYirw?e=3BZENR)     |
| Image Classifier  |         Self-supervised Clustering          | [Img_encoder.pth.tar](https://mysnu-my.sharepoint.com/:u:/g/personal/blackfoot_seoul_ac_kr/Eb6Xa9AdaLVJmqbIrujfIeMBfor6bA_svioih4R2XGPIkA?e=ZyIJcU) |
| Object Classifier |            Supervised Clustering            | [Obj_encoder.pth.tar](https://mysnu-my.sharepoint.com/:u:/g/personal/blackfoot_seoul_ac_kr/ESJwR8YHqgRNpErupUo2DVEBf9k7csRECl6hyDA3Xs5seQ?e=gJbbok) |

## Citation
If you find this code useful for your research, please consider citing:
```Bibtex
@inproceedings{TSGM,
      title={{Topological Semantic Graph Memory for Image Goal Navigation}},
      author={Nuri Kim and Obin Kwon and Hwiyeon Yoo and Yunho Choi and Jeongho Park and Songhawi Oh},
      year={2022},
      booktitle={CoRL}
}
```


## Acknowledgements
In our work, we used parts of [VGM](https://rllab-snu.github.io/projects/vgm/doc.html), 
and [Habitat-Lab](https://github.com/facebookresearch/habitat-lab) repos and extended them.

## Related Work
- [VGM](https://rllab-snu.github.io/projects/vgm/doc.html)
- [NeuralTopologicalSLAM](https://devendrachaplot.github.io/projects/neural-topological-slam.html)
- [ActiveNeuralSLAM](https://devendrachaplot.github.io/projects/neural-slam.html)

## License
This project is released under the MIT license, as found in the [LICENSE](LICENSE) file.
