# ShrinkMatch for STL-10 and SVHN

This codebase provides the official PyTorch implementation of our ICCV 2023 paper:

> **[Shrinking Class Space for Enhanced Certainty in Semi-Supervised Learning]()**</br>
> Lihe Yang, Zhen Zhao, Lei Qi, Yu Qiao, Yinghuan Shi, Hengshuang Zhao</br>
> *In International Conference on Computer Vision (ICCV), 2023*</br>

Note: This codebase is based on TorchSSL. Our baseline is FixMatch + DA (Distribution Alignment) on STL-10, while is solely FixMatch on SVHN.


## Results

**We provide all training logs [here](../training-logs). You can refer to them when reproducing.**

### STL-10 @40 labels

|       Seed      |   0   |   1   |   2   |    Mean   |
|:---------------:|:-----:|:-----:|:-----:|:---------:|
|    FlexMatch    | 76.71 | 68.28 | 67.55 |   70.85   |
| **ShrinkMatch** | 85.75 | 85.64 | 86.55 | **85.98** |


### SVHN @40 labels

|       Seed      |   0   |   1   |   2   |    Mean   |
|:---------------:|:-----:|:-----:|:-----:|:---------:|
|    FlexMatch    | 89.19 | 89.93 | 96.32 |   91.81   |
|    FixMatch     | 94.53 | 96.90 | 97.14 |   96.19   |
| **ShrinkMatch** | 97.96 | 97.81 | 96.70 | **97.49** |


## Installation

```
conda create -n shrinkmatch python=3.8
conda activate shrinkmatch
pip install -r requirements.txt
```

## Usage

```
python shrinkmatch.py --c <config> 
```

E.g.,

```
python shrinkmatch.py --c config/shrinkmatch_stl10_40labels_seed0.yaml
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