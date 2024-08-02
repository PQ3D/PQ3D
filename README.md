## <img src='https://pq3d.github.io/file/pq3d-logo.png' width=3%> Unifying 3D Vision-Language Understanding via Promptable Queries

<p align="left">
    <a href='https://arxiv.org/abs/2405.11442'>
      <img src='https://img.shields.io/badge/Paper-PDF-red?style=plastic&logo=adobeacrobatreader&logoColor=red' alt='Paper PDF'>
    </a>
    <a href='https://arxiv.org/abs/2405.11442'>
      <img src='https://img.shields.io/badge/Paper-arXiv-green?style=plastic&logo=arXiv&logoColor=green' alt='Paper arXiv'>
    </a>
    <a href='https://pq3d.github.io'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=plastic&logo=Google%20chrome&logoColor=blue' alt='Project Page'>
    </a>
    <a href='https://huggingface.co/spaces/li-qing/PQ3D-Demo'>
      <img src='https://img.shields.io/badge/Demo-HuggingFace-yellow?style=plastic&logo=AirPlay%20Video&logoColor=yellow' alt='HuggigFace'>
    </a>
    <a href='https://drive.google.com/drive/folders/1MDt9yaco_TllGfqqt76UOxMV1JMMVsji?usp=share_link'>
      <img src='https://img.shields.io/badge/Model-Checkpoints-orange?style=plastic&logo=Google%20Drive&logoColor=orange' alt='Checkpoints(TODO)'>
    </a>
</p>

[Ziyu Zhu](https://zhuziyu-edward.github.io/), 
[Zhuofan Zhang](https://tongclass.ac.cn/author/zhuofan-zhang/), 
[Xiaojian Ma](https://jeasinema.github.io/), 
[Xuesong Niu](https://scholar.google.com/citations?user=iuPSV-0AAAAJ&hl=en),
[Yixin Chen](https://yixchen.github.io/),
[Baoxiong Jia](https://buzz-beater.github.io),
[Zhidong Deng](https://www.cs.tsinghua.edu.cn/csen/info/1165/4052.htm)📧,
[Siyuan Huang](https://siyuanhuang.com/)📧,
[Qing Li](https://liqing-ustc.github.io/)📧

This repository is the official implementation of the ECCV 2024 paper "Unifying 3D Vision-Language Understanding via Promptable Queries".

[Paper](https://arxiv.org/abs/2405.11442) |
[arXiv](https://arxiv.org/abs/2405.11442) |
[Project](https://pq3d.github.io) |
[HuggingFace Demo](https://huggingface.co/spaces/li-qing/PQ3D-Demo) |
[Checkpoints](https://drive.google.com/drive/folders/1MDt9yaco_TllGfqqt76UOxMV1JMMVsji?usp=share_link)

<div align=center>
<img src='https://pq3d.github.io/file/teaser.png' width=100%>
</div>

### News
- [ 2024.08 ] Release training and evaluation.
- [ 2024.07 ] Our huggingface DEMO is here [DEMO](https://huggingface.co/spaces/li-qing/PQ3D-Demo), welcome to try our model!
- [ 2024.07 ] Release codes of model! TODO: Clean up training and evaluation

### Abstract

A unified model for 3D vision-language (3D-VL) understanding is expected to take various scene representations and perform a wide range of tasks in a 3D scene. However, a considerable gap exists between existing methods and such a unified model, due to the independent application of representation and insufficient exploration of 3D multi-task training. In this paper, we introduce PQ3D, a unified model capable of using Promptable Queries to tackle a wide range of 3D-VL tasks, from low-level instance segmentation to high-level reasoning and planning.Tested across ten diverse 3D-VL datasets, This is achieved through three key innovations: (1) unifying various 3D scene representations (i.e., voxels, point clouds, multi-view im- ages) into a shared 3D coordinate space by segment-level grouping, (2) an attention-based query decoder for task-specific information retrieval guided by prompts, and (3) universal output heads for different tasks to support multi-task training. PQ3D demonstrates impressive performance on these tasks, setting new records on most benchmarks. Particularly, PQ3D improves the state- of-the-art on ScanNet200 by 4.9% (AP25), ScanRefer by 5.4% (acc@0.5), Multi3DRefer by 11.7% (F1@0.5), and Scan2Cap by 13.4% (CIDEr@0.5).Moreover, PQ3D supports flexible inference with individual or combined forms of available 3D representations, e.g., solely voxel input

### Install
1. Install conda package
```
conda env create --name envname 
pip3 install torch==2.0.0
pip3 install torchvision==0.15.1
pip3 install -r requirements.txt
```

2. install pointnet2
```
cd modules/third_party
# PointNet2
cd pointnet2
python setup.py install
cd ..
```

3. Install Minkowski Engine
```
git clone https://github.com/NVIDIA/MinkowskiEngine.git
conda install openblas-devel -c anaconda
cd MinkowskiEngine
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas
```

### Prepare data
1. download sceneverse data  from [scene_verse_base](https://github.com/scene-verse/sceneverse?tab=readme-ov-file) and change `data.scene_verse_base` to sceneverse data directory.
2. download segment level data from [scene_ver_aux](https://drive.google.com/drive/folders/1em0G5S4aH4lCIfnjLxyFO3DV3Rwuwnqh?usp=share_link) and change `data.scene_verse_aux` to download data directory.
3. download other data from [scene_verse_pred](https://drive.google.com/drive/folders/12BjbhXzV7lON4X0tx3e7DdvZHhI1Q1f2?usp=share_link) and change `data.scene_verse_pred` to download data directory.

### Prepare checkpoints
1. download PointNet++ from [pointnet](https://drive.google.com/drive/folders/1KSu2z9cOMocQg5yYKUTlk0TlDwJZQMKs?usp=share_link) and change `pretrained_weights_dir` to downloaded directory,
2. download checkpoint from [stage1](https://drive.google.com/drive/folders/1KHayAzF7XlE3R3cz4YUPoHY_1VJkKYpd?usp=share_link), [stage2](https://drive.google.com/drive/folders/1MDt9yaco_TllGfqqt76UOxMV1JMMVsji?usp=share_link) and change pretrain_ckpt_path to certain checkpoint weight.

### Run PQ3D
Stage 1 training for instance segmentation
```
python3 run.py --config-path configs --config-name instseg_sceneverse_gt.yaml
```
```
python3 run.py --config-path configs --config-name instseg_sceneverse.yaml pretrain_ckpt={ckpt_from_instseg_sceneverser_gt}
```
Stage 1 evaluation
```
python3 run.py --config-path configs --config-name instseg_sceneverse.yaml mode=test pretrain_ckp_path={pretrain_ckpt_path}
```
Stage 2 training for vl tasks
```
python3 run.py --config-path configs --config-name unified_tasks_sceneverse.yaml
```
Stage 2 evaluation
```
python3 run.py --config-path configs --config-name unified_tasks_sceneverse.yaml mode=test pretrain_ckpt_path={pretrain_ckpt_path}
```
For multi-gpu training usage, we use four GPU in our experiments.
```
python launch.py --mode ${launch_mode} \
    --qos=${qos} --partition=${partition} --gpu_per_node=4 --port=29512 --mem_per_gpu=80 \
    --config {config}  \
```

### Acknowledgement
We would like to thank the authors of [Vil3dref](https://github.com/cshizhe/vil3dref), [Mask3d](https://github.com/JonasSchult/Mask3D), [Openscene](https://github.com/pengsongyou/openscene), [Xdecoder](https://github.com/microsoft/X-Decoder), and [3D-VisTA](https://github.com/3d-vista/3D-VisTA) for their open-source release.


### Citation:
```
@article{zhu2024unifying,
    title={Unifying 3D Vision-Language Understanding via Promptable Queries},
    author={Zhu, Ziyu and Zhang, Zhuofan and Ma, Xiaojian and Niu, Xuesong and Chen, Yixin and Jia, Baoxiong and Deng, Zhidong and Huang, Siyuan and Li, Qing},
    journal={arXiv preprint arXiv:2405.11442},
    year={2024}
}
```
