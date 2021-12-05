# Author: Vincent Zhang
# Mail: zhyx12@gmail.com
# ----------------------------------------------
from mmcv.utils import Registry
from torch.utils.data import DataLoader
import numpy as np
import random
from mmcv.parallel import collate
from functools import partial
from mmcv.runner import get_dist_info
from copy import deepcopy
from torch.utils import data
from torch.utils.data import RandomSampler
from torch.utils.data._utils.pin_memory import pin_memory
from torch.utils.data.distributed import DistributedSampler
from mmcv.utils import build_from_cfg
from fastda.utils import get_root_logger

DATASETS = Registry('fastda_datasets')
DATABUILDERS = Registry('fastda_databuilders')


@DATABUILDERS.register_module(name='default')
class DefaultDataBuilder(object):
    def __init__(self, dataset, samples_per_gpu, num_workers, shuffle, pin_memory, drop_last, seed, **kwargs):
        sampler = self.build_sampler(dataset, shuffle, samples_per_gpu, seed)
        collate_fn = self.build_collate_fn(samples_per_gpu)
        worker_init_fn = self.build_init_fn(num_workers, seed)
        self.dataloader = DataLoader(dataset, batch_size=samples_per_gpu, num_workers=num_workers, shuffle=False,
                                     sampler=sampler, pin_memory=pin_memory,
                                     drop_last=drop_last, collate_fn=collate_fn,
                                     worker_init_fn=worker_init_fn)

    def build_sampler(self, dataset, shuffle, samples_per_gpu=None, seed=None):
        return DistributedSampler(dataset, shuffle=shuffle)

    def build_collate_fn(self, samples_per_gpu):
        return partial(collate, samples_per_gpu=samples_per_gpu)

    def build_init_fn(self, num_workers, seed):
        rank, world_size = get_dist_info()
        return partial(self.worker_init_fn, num_workers=num_workers, rank=rank,
                       seed=seed) if seed is not None else None

    def get_dataloader(self):
        return self.dataloader

    def worker_init_fn(self, worker_id, num_workers, rank, seed):
        # The seed of each worker equals to
        # num_worker * rank + worker_id + user_seed
        worker_seed = num_workers * rank + worker_id + seed
        np.random.seed(worker_seed)
        random.seed(worker_seed)


def process_one_dataset(args, pipelines, samplers_per_gpu, n_workers, shuffle,
                        drop_last=True, data_root=None, random_seed=None,
                        debug=False, sample_num=None):
    dataset_params = deepcopy(args)
    #
    if 'pipeline' not in args:
        dataset_params['pipeline'] = pipelines
    #
    if 'data_root' not in dataset_params:
        dataset_params['data_root'] = data_root
    #
    if 'samples_per_gpu' in dataset_params:
        temp_samples_per_gpu = dataset_params['samples_per_gpu']
        dataset_params.pop('samples_per_gpu')
    else:
        temp_samples_per_gpu = samplers_per_gpu
    #
    dataset = build_from_cfg(dataset_params, DATASETS)
    #
    dataloader_params = dict(
        dataset=dataset,
        samples_per_gpu=temp_samples_per_gpu,
        num_workers=n_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        drop_last=drop_last,
        seed=random_seed,
    )
    if 'task_specific' in DATABUILDERS:
        type_param = {'type': 'task_specific'}
    else:
        type_param = {'type': 'default'}
    dataloader_params.update(type_param)
    loader = build_from_cfg(dataloader_params, DATABUILDERS).get_dataloader()
    return loader


def parse_args_for_multiple_datasets(dataset_args, random_seed=None, data_root=None, debug=False,
                                     train_debug_sample_num=None, test_debug_sample_num=None):
    """

    :param data_root:
    :param random_seed:
    :param dataset_args:
    :param debug:
    :param train_debug_sample_num:
    :param test_debug_sample_num:
    :return: 返回一个list
    """
    logger = get_root_logger()
    if debug:
        print("YOU ARE IN DEBUG MODE!!!!!!!!!!!!!!!!!!!")
    # Setup Augmentations
    trainset_args = dataset_args['train']
    testset_args = dataset_args['test']
    train_augmentations = trainset_args.get('pipeline', None)
    test_augmentations = testset_args.get('pipeline', None)
    # Setup Dataloader
    # 其它参数
    train_samples_per_gpu = trainset_args.get('samples_per_gpu', None)
    test_samples_per_gpu = testset_args.get('samples_per_gpu', None)
    n_workers = dataset_args['n_workers']
    drop_last = dataset_args.get('drop_last', True)

    # 训练集
    train_loaders = []
    for i in range(1, 100):
        if i in trainset_args.keys():
            temp_train_aug = trainset_args[i].get('augmentation', None)
            temp_train_aug = train_augmentations if temp_train_aug is None else temp_train_aug
            temp_train_loader = process_one_dataset(trainset_args[i], pipelines=temp_train_aug,
                                                    samplers_per_gpu=train_samples_per_gpu, n_workers=n_workers,
                                                    debug=debug, shuffle=True,
                                                    sample_num=train_debug_sample_num, drop_last=drop_last,
                                                    data_root=data_root,
                                                    random_seed=random_seed)
            train_loaders.append(temp_train_loader)
        else:
            break

    # 测试集
    test_loaders = []
    for i in range(1, 100):
        if i in testset_args.keys():
            temp_test_aug = testset_args[i].get('augmentation', None)
            temp_test_aug = test_augmentations if temp_test_aug is None else temp_test_aug
            temp_test_loader = process_one_dataset(testset_args[i], pipelines=temp_test_aug,
                                                   samplers_per_gpu=test_samples_per_gpu,
                                                   n_workers=n_workers,
                                                   shuffle=False, debug=debug,
                                                   sample_num=test_debug_sample_num,
                                                   drop_last=False, data_root=data_root,
                                                   random_seed=random_seed,
                                                   )
            test_loaders.append(temp_test_loader)
        else:
            break
    #
    for i, loader in enumerate(train_loaders):
        logger.info('{} train loader has {} images'.format(i, len(loader.dataset)))
    return train_loaders, test_loaders
