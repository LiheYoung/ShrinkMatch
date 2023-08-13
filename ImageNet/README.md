# ShrinkMatch for ImageNet-1K

This codebase provides the official PyTorch implementation of our ICCV 2023 paper:

> **[Shrinking Class Space for Enhanced Certainty in Semi-Supervised Learning]()**</br>
> Lihe Yang, Zhen Zhao, Lei Qi, Yu Qiao, Yinghuan Shi, Hengshuang Zhao</br>
> *In International Conference on Computer Vision (ICCV), 2023*</br>

Note: This codebase is based on SimMatch. Our baseline is SimMatch.


## Results

**We provide all training logs [here](../training-logs). You can refer to them when reproducing.**

### ImageNet-1K

|     Accuracy    | Top-1 @1% labels | Top-1 @10% labels | Top-5 @1% labels | Top-5 @10% labels |
|:---------------:|:----------------:|:----------------:|:----------------:|:----------------:|
|    SimMatch*    |        67.0      |       74.1       |       86.9       |       91.5       |
| **ShrinkMatch** |      **67.5**    |     **74.5**     |     **87.4**     |     **91.9**     |

*Reproduced in our environment

## Installation

```
conda create -n shrinkmatch python=3.9.2
conda activate shrinkmatch
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
```


## Data Preparation

Please follow [this script](https://gist.github.com/BIGBALLON/8a71d225eff18d88e469e6ea9b39cef4) to download and organize the ImageNet-1K dataset.

## Usage

```
sh script/dist_train.sh <data_path> <split> <num_gpus> <batch_size> <port>
```
where ``split`` can be 0.01 or 0.1. To reproduce our results, the total batch size of ``num_gpus`` $\times$ ``batch_size`` should be 64. We use 8 V100 GPUs, and each GPU has a batch size of 8.

E.g.,

```
sh script/dist_train.sh ./data/ImageNet 0.01 8 8 23456
```



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