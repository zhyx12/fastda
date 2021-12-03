# Author: Vincent Zhang
# Mail: zhyx12@gmail.com
# ----------------------------------------------
from mmcv.utils import Registry, build_from_cfg

TRAINER = Registry('fastda_trainer')
VALIDATOR = Registry('fastda_validator')


def build_trainer(cfg, default_args=None):
    return build_from_cfg(cfg, TRAINER, default_args=default_args)


def build_validator(cfg, default_args=None):
    return build_from_cfg(cfg, VALIDATOR, default_args=default_args)
