# FastDA

## Introduction

This is a simple framework for domain adaptation training. You can use it to build your own training process. It heavily relies on [MMCV](https://github.com/open-mmlab/mmcv) since we use a lot of useful tools (e.g., [Config](https://mmcv.readthedocs.io/en/latest/understand_mmcv/config.html), [Registry](https://mmcv.readthedocs.io/en/latest/understand_mmcv/registry.html), [Hook]()). The main difference between FastDA and MMCV is the [Runner](https://mmcv.readthedocs.io/en/latest/understand_mmcv/runner.html) class. MMCV provides two kinds of runners to control the training and validation process, namely [**EpochBasedRunner**]() and [**IterBasedRunner** ]().



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



## License

This project is released under the [MIT License](LICENSE).
