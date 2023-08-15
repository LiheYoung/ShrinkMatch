# ShrinkMatch for CIFAR-10 and CIFAR-100

This codebase provides the official PyTorch implementation of our ICCV 2023 paper:

> **[Shrinking Class Space for Enhanced Certainty in Semi-Supervised Learning](https://arxiv.org/abs/2308.06777)**</br>
> Lihe Yang, Zhen Zhao, Lei Qi, Yu Qiao, Yinghuan Shi, Hengshuang Zhao</br>
> *In International Conference on Computer Vision (ICCV), 2023*</br>

Note: This codebase is based on SimMatch for CIFAR-10 and CIFAR-100. Our baseline is FixMatch + DA (Distribution Alignment), same as SimMatch.


## Results

**We provide [all training logs](../training-logs). You can refer to them when reproducing.**

### CIFAR-10 @40 labels

|       Seed      |   0   |   1   |   2   |   3   |   4   |    Mean   |
|:---------------:|:-----:|:-----:|:-----:|:-----:|:-----:|:---------:|
|    SimMatch     | 95.34 | 95.16 | 92.63 | 93.76 | 95.10 |   94.39   |
| **ShrinkMatch** | 95.09 | 94.66 | 95.12 | 94.78 | 94.95 | **94.92** |


### CIFAR-100 @400 labels

|       Seed      |   0   |   1   |   2   |   3   |   4   |    Mean   |
|:---------------:|:-----:|:-----:|:-----:|:-----:|:-----:|:---------:|
|    SimMatch     | 62.06 | 60.19 | 59.89 | 64.88 | 63.92 |   62.19   |
| **ShrinkMatch** | 65.00 | 63.47 | 63.77 | 66.42 | 64.52 | **64.64** |


## Installation

```
conda create -n shrinkmatch python=3.9.2
conda activate shrinkmatch
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
```

## Usage

```
CUDA_VISIBLE_DEVICES=<gpu_id> sh script/dist_train.sh <dataset> <labels_per_class> <seed> <port>
```

E.g.,

```
CUDA_VISIBLE_DEVICES=0 sh script/dist_train.sh cifar100 4 0 23456
```

Data will be automatically downloaded.


## Citation

If you find this project useful, please consider citing:

```bibtex
@inproceedings{shrinkmatch,
  title={Shrinking Class Space for Enhanced Certainty in Semi-Supervised Learning},
  author={Yang, Lihe and Zhao, Zhen and Qi, Lei and Qiao, Yu and Shi, Yinghuan and Zhao, Hengshuang},
  booktitle={ICCV},
  year={2023}
}
```