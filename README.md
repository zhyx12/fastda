# FastDA

## Introduction

FastDA is a simple framework for domain adaptation training.  

It relies on [MMCV](https://github.com/open-mmlab/mmcv) via borrowing a lot of useful tools and mechanisms (*e.g.,* [Config](https://mmcv.readthedocs.io/en/latest/understand_mmcv/config.html), [Registry](https://mmcv.readthedocs.io/en/latest/understand_mmcv/registry.html), [Hook](https://mmcv.readthedocs.io/en/latest/api.html#mmcv.runner.BaseRunner.register_training_hooks)). MMCV acts as a foundational library for computer vision research and supports many projects such as [MMClassification](https://github.com/open-mmlab/mmclassification), [MMSegmentation](https://github.com/open-mmlab/mmsegmentation), [MMDetection](https://github.com/open-mmlab/mmdetection). 

We hope **FastDA** can also support domain adaptation training for different research areas.

## Design idea

To make FastDA more suitable for domain adaptation, we first review the differences between traditional supervised training and domain adaptation training.

|           | Training in mmcls/mmseg/mmdet                                | Training in domain adaptation                                |
| :-------- | :----------------------------------------------------------- | ------------------------------------------------------------ |
| Dataset   | single train dataset                                         | multiple train datasets                                      |
| Model     | single model with single optimizer                           | multiple models (*e.g.*, base model and domain classifier) <br/>sometimes different models has different optimizer parameters |
| Optimizer | single loss (summation of all losses)<br/>call loss.backward() once in each iteration | multiple loss.backward() for calculating gradients of <br/>different models |

It can be seen that the training process in domain adaptation are more complicated. 

### 1. Datasets and Models

For dataset and model which are the basic parts for training, we define a set of new rules when writing dataset and model configs.  By parsing the dataset config, we can get multiples training datasets. By parsing the model config, we can get a [ModuleDict](https://pytorch.org/docs/stable/generated/torch.nn.ModuleDict.html?highlight=moduledict#torch.nn.ModuleDict) where the key is an user defined name and the value contains a instantiated model and its optimizer and learning rate scheduler.

### 2. Trainer and Validator

For the training and testing process, MMCV use the [Runner](https://mmcv.readthedocs.io/en/latest/understand_mmcv/runner.html) class to control them. Two kinds of runners are provided, namely [**EpochBasedRunner**](https://mmcv.readthedocs.io/en/latest/understand_mmcv/runner.html#epochbasedrunner) and [**IterBasedRunner** ](https://mmcv.readthedocs.io/en/latest/understand_mmcv/runner.html#iterbasedrunner). The latter one is more natural for domain adaptation training since the number of samples in the source and target datasets are always different and it is hard to define an epoch. Actually, most methods report their implementation details based on iterations but not epochs.

Instead of directly using IterBasedRunner where the whole training process are wrapped in a single top-level model 's train_step() function, we put interaction among different models in a [Trainer](./fastda/runner/trainer.py). It can be regarded as a *minimum* implementation of basic running process, which can take care of  text logging, tensorboard logging, checkpoint saving, resume training, scheduler step etc. This is achieved by registering predefined [hooks](./fastda/hooks/training_hooks.py) to trainer.

Besides Trainer, we also introduce [Validator](./fastda/runner/validator.py) to control the validation (testing) process. There is not default hook for validator. When building your own project, you should create a task-specific evaluation hook and register it to validator.



**Note**: For a more concise and easy implementation, we impose some restrictions as follows:

- We only consider training and testing based on GPU device.
- Use DistributedDataParallel instead of original model or DataParallel. More specifically, we use MMDistributedDataParallel in mmcv which is more consistent with the DataContainer.
- In the BaseTrainer class, we do not impose operations of optimizer such as optimizer.zero_grad() and optimizer.step(). You can either register OptimizerHook in your trainer's init function or manipulate it yourself.

## Installation

1. Prepare environment: Install [pytorch](https://pytorch.org/) and [mmcv](https://mmcv.readthedocs.io/en/latest/get_started/installation.html) . 

```bash
pip3 install torch
pip3 install mmcv
```

Note: Since MMCV requires Python 3.6+, FastDA also maintains this requirement.

2. Install FastDA

```bash
pip3 install fastda
```

## Train procedure

![img](./docs/figures/train.svg)

## Build your own project



## License

This project is released under the [MIT License](LICENSE).
