# FastDA

## Introduction

FastDA is a simple framework for domain adaptation training.  

**FastDA** relies on [MMCV](https://github.com/open-mmlab/mmcv) via borrowing a lot of useful tools and mechanisms (*e.g.,* [Config](https://mmcv.readthedocs.io/en/latest/understand_mmcv/config.html), [Registry](https://mmcv.readthedocs.io/en/latest/understand_mmcv/registry.html), [Hook](https://mmcv.readthedocs.io/en/latest/)). MMCV acts as a foundational library for computer vision research and supports many projects such as [MMClassification](https://github.com/open-mmlab/mmclassification), [MMSegmentation](https://github.com/open-mmlab/mmsegmentation), [MMDetection](https://github.com/open-mmlab/mmdetection). 

We hope FastDA can also support domain adaptation training for different research areas.

## Design idea

To make FastDA more suitable for domain adaptation, we first review the differences between traditional supervised training process and domain adaptation training process.

|          | Training in mmcls/mmseg/mmdet      | Training in domain adaptation                                |
| -------- | ---------------------------------- | ------------------------------------------------------------ |
| Datasets | single train dataset               | multiple train datasets                                      |
| Models   | single model with single optimizer | multiple models (*e.g.*, base model and domain classifier) <br/>sometime different models has different optimizers |

MMCV use the [Runner](https://mmcv.readthedocs.io/en/latest/understand_mmcv/runner.html) class to control the training and testing process, and they provide two kinds of runners, namely [**EpochBasedRunner**]() and [**IterBasedRunner** ](). The latter one is more natural for domain adaptation training since the number of samples in the source and target datasets are always different and it is hard to define an epoch. Actually, most methods report their implementation details based on iterations but not epochs.

We do not directly adopt the IterBasedRunner, but maintain a minimum Trainer and Validator which control the training and validation (testing) process respectively.

**Note**: For a more concise and easy implementation, we impose some restrictions as follows:

- We only consider training and testing based on GPU device.
- Use DistributedDataParallel instead of original model or DataParallel. More specifically, we use MMDistributedDataParallel in mmcv which is more consistent with the DataContainer.

## Installation

1. Prepare environment: Install [pytorch](https://pytorch.org/) and [mmcv](https://mmcv.readthedocs.io/en/latest/get_started/installation.html) . 

```bash
pip3 install torch
pip3 install mmcv
```

Note: Since MMCV requires Python 3.6+, our FastDA also maintains this requirements.

2. Install FastDA

```bash
pip3 install fastda
```



## Train procedure

![img](./docs/figures/train.svg)



## License

This project is released under the [MIT License](LICENSE).
