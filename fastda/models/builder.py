# Author: Vincent Zhang
# Mail: zhyx12@gmail.com
# ----------------------------------------------
import torch
import torch.nn as nn
import copy
from ..utils import move_models_to_gpu
from mmcv.utils import Registry, build_from_cfg
from .optimizers import build_model_defined_optimizer
from .schedulers import build_scheduler


MODELS = Registry('fastda_models')


def build_models(cfg, default_args=None):
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, MODELS, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, MODELS, default_args)


def parse_args_for_one_model(model_args, scheduler_args, find_unused_parameters=False,
                             max_card=1, sync_bn=False):
    """
    输入带名字的字典，
    :param model_args: 类型是字典，名字就是model，optimizer，scheduler的名字的前者
    :param scheduler_args:
    :param logger:
    :return:
    """
    assert 'optimizer' in model_args.keys(), 'model args should have optimizer args'
    model_args = copy.deepcopy(model_args)
    # 获取参数
    optimizer_params = model_args['optimizer']
    model_args.pop('optimizer')
    scheduler_params = model_args.get('scheduler', None)
    if scheduler_params is not None:
        model_args.pop('scheduler')
    else:
        scheduler_params = scheduler_args
    #
    device_params = model_args.get('device', 0)
    if 'device' in model_args.keys():
        model_args.pop(device_params)
    # 构造模型
    temp_model = build_models(model_args)
    if sync_bn:
        print('Using sync_bn mode')
        temp_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(temp_model)
    # move model to gpu
    temp_model = move_models_to_gpu(temp_model, device_params, max_card=max_card,
                                    find_unused_parameters=find_unused_parameters)
    if optimizer_params is not None:
        temp_optimizer = build_model_defined_optimizer(temp_model, optimizer_params)
        temp_scheduler = build_scheduler(temp_optimizer, scheduler_params)
    else:
        temp_optimizer = None
        temp_scheduler = None

    return temp_model, temp_optimizer, temp_scheduler


def parse_args_for_models(model_args, find_unused_parameters=False, sync_bn=False):
    shared_lr_scheduler_param = model_args['lr_scheduler']
    model_args.pop('lr_scheduler')
    model_dict = nn.ModuleDict()
    optimizer_dict = {}
    scheduler_dict = {}
    # 获得一个进行最多需要多少块卡（model parallel）
    max_need_card = 0
    for key in model_args:
        tmp_device = model_args[key].get('device', 0)
        if tmp_device > max_need_card:
            max_need_card = tmp_device
    max_need_card += 1
    #
    for key in model_args:
        temp_res = parse_args_for_one_model(model_args[key], shared_lr_scheduler_param,
                                            find_unused_parameters=find_unused_parameters, max_card=max_need_card,
                                            sync_bn=sync_bn)
        model_dict[key] = temp_res[0]
        if temp_res[1] is not None:
            optimizer_dict[key] = temp_res[1]
            scheduler_dict[key] = temp_res[2]
    return model_dict, optimizer_dict, scheduler_dict
