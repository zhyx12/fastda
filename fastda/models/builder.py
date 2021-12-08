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
from ..utils import get_root_logger

MODELS = Registry('fastda_models')


def build_models(cfg, default_args=None):
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, MODELS, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, MODELS, default_args)


def parse_args_for_one_model(model_args, optimizer_args, scheduler_args, find_unused_parameters=None,
                             broadcast_buffers=None, max_card=1, sync_bn=None):
    """
    输入带名字的字典，
    :param model_args: 类型是字典，名字就是model，optimizer，scheduler的名字的前者
    :param scheduler_args:
    :param logger:
    :return:
    """
    logger = get_root_logger()
    model_args = copy.deepcopy(model_args)
    #
    tmp_optimizer_args = model_args.get('optimizer', None)
    final_optimizer_args = tmp_optimizer_args if tmp_optimizer_args is not None else optimizer_args
    tmp_lr_scheduler_args = model_args.get('lr_scheduler', None)
    final_lr_scheduler_args = tmp_lr_scheduler_args if tmp_lr_scheduler_args is not None else scheduler_args
    #
    if final_optimizer_args is None and final_lr_scheduler_args is None:
        logger.warning('model have no optimizer and lr scheudler')
    if (final_optimizer_args is None and final_lr_scheduler_args is not None) or \
            (final_lr_scheduler_args is None and final_optimizer_args is not None):
        raise RuntimeError('You should specify optimizer and scheduler simultaneously')
    #
    if 'optimizer' in model_args:
        model_args.pop('optimizer')
    if 'lr_scheduler' in model_args:
        model_args.pop('lr_scheduler')
    #
    device_params = model_args.get('device', 0)
    if 'device' in model_args.keys():
        model_args.pop(device_params)
    # 构造模型
    temp_model = build_models(model_args)
    tmp_sync_bn = model_args.get('sync_bn', None)
    final_sync_bn = tmp_sync_bn if tmp_sync_bn is not None else sync_bn
    if final_sync_bn:
        logger.info('Use SyncBatchNorm Mode')
        temp_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(temp_model)
    #
    tmp_find_unused_parameters = model_args.get('find_unused_parameters', None)
    final_find_unused_parameters = tmp_find_unused_parameters if tmp_find_unused_parameters is not None else find_unused_parameters
    tmp_broadcast_buffers = model_args.get('broadcast_buffers', None)
    final_broadcast_buffers = tmp_broadcast_buffers if tmp_broadcast_buffers is not None else broadcast_buffers
    # move model to gpu
    temp_model = move_models_to_gpu(temp_model, device_params, max_card=max_card,
                                    find_unused_parameters=final_find_unused_parameters,
                                    broadcast_buffers=final_broadcast_buffers)
    if final_optimizer_args is not None:
        temp_optimizer = build_model_defined_optimizer(temp_model, final_optimizer_args)
        temp_scheduler = build_scheduler(temp_optimizer, final_lr_scheduler_args)
    else:
        temp_optimizer = None
        temp_scheduler = None

    return temp_model, temp_optimizer, temp_scheduler


def parse_args_for_models(model_args):
    model_args = copy.deepcopy(model_args)
    #
    shared_optimizer_params = model_args.get('optimizer', None)
    if 'optimizer' in model_args:
        model_args.pop('optimizer')
    shared_lr_scheduler_param = model_args.get('lr_scheduler', None)
    if 'lr_scheduler' in model_args:
        model_args.pop('lr_scheduler')
    #
    # global find_unused_parameters setting
    find_unused_parameters = model_args.get('find_unused_parameters', False)
    if find_unused_parameters in model_args:
        model_args.pop('find_unused_parameters')
    broadcast_buffers = model_args.get('broadcast_buffers',
                                       False)  # set default value to False, which is also adopted in mmcls/mmseg/mmdet
    if broadcast_buffers in model_args:
        model_args.pop('broadcast_buffers')
    # global sync_bn setting
    sync_bn = model_args.get('sync_bn', None)
    if 'sync_bn' in model_args:
        model_args.pop('sync_bn')
    #
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
        temp_res = parse_args_for_one_model(model_args[key], shared_optimizer_params, shared_lr_scheduler_param,
                                            find_unused_parameters=find_unused_parameters, max_card=max_need_card,
                                            sync_bn=sync_bn)
        model_dict[key] = temp_res[0]
        if temp_res[1] is not None:
            optimizer_dict[key] = temp_res[1]
            scheduler_dict[key] = temp_res[2]
    return model_dict, optimizer_dict, scheduler_dict
