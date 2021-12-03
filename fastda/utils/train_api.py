# Author: Vincent Zhang
# Mail: zhyx12@gmail.com
# ----------------------------------------------
import os
import shutil
import torch
import random
import numpy as np
from . import get_root_logger, get_root_writer
from ..loaders import parse_args_for_multiple_datasets
from ..models import parse_args_for_models
from . import deal_with_val_interval
#
import time
from ..hooks import LrRecorder, TrainTimeRecoder, SaveModel, SchedulerStep
from mmcv import Config
from ..runner import build_trainer, build_validator
from mmcv.runner import get_dist_info
from .collect_env import collect_env

Allowable_Control_Key = ['log_interval', 'max_iters', 'val_interval', 'cudnn_deterministic', 'save_interval',
                         'max_save_num', 'seed', 'checkpoint', 'pretrained_model', 'save_init_model',
                         'find_unused_parameters', 'sync_bn', 'test_mode']


def train(args):
    torch.distributed.init_process_group(backend="nccl")
    local_rank, world_size = get_dist_info()
    #
    cfg = Config.fromfile(args.config)
    predefined_keys = ['datasets', 'models', 'control', 'train', 'test']
    old_keys = list(cfg._cfg_dict.keys())
    for key in old_keys:
        if not key in predefined_keys:
            del cfg._cfg_dict[key]
    # check control keys are allowable
    control_cfg = cfg['control']
    for key in control_cfg.keys():
        assert key in Allowable_Control_Key, '{} is not allowed appeared in control keys'.format(key)
    #
    run_id = random.randint(1, 100000)
    run_id_tensor = torch.ones((1,), device='cuda:{}'.format(local_rank)) * run_id
    torch.distributed.broadcast(run_id_tensor, 0)
    run_id = int(run_id_tensor.cpu().item())
    logdir = os.path.join('runs', os.path.basename(args.config)[:-3],
                          'job_' + args.job_id + '_exp_' + str(run_id))
    #
    if local_rank == 0:
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        #
        shutil.copy(args.config, logdir)  #
        shutil.copytree('./{}'.format(args.source_code_name), os.path.join(logdir, 'source_code'))
        #
        cfg_save_path = os.path.join(logdir, 'config.py')
        cfg.dump(cfg_save_path)
        #
    tb_writer = get_root_writer(log_dir=logdir)
    timestamp = time.strftime('runs_%Y_%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(logdir, f'rank_{local_rank}_{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=args.log_level)
    logger.info('log dir is {}'.format(logdir))
    logger.info('Let the games begin')
    logger.info('Experiment identifier is {}'.format(args.job_id))
    #
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    # Setup random seeds
    if 'seed' in control_cfg:
        random_seed = control_cfg.get('seed', None)
    else:
        random_seed = random.randint(1000, 2000)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    logger.info("Random Seed is {}".format(random_seed))
    # debug mode: set dataset sample number
    debug_flag = args.debug
    train_debug_sample_num = args.train_debug_sample_num
    test_debug_sample_num = args.test_debug_sample_num
    # debug mode: change log_interval和val_interval
    if debug_flag:
        control_cfg['log_interval'] = args.debug_log_interval
        control_cfg['val_interval'] = args.debug_val_interval
    #
    # build dataloader
    train_loaders, test_loaders = parse_args_for_multiple_datasets(cfg['datasets'], debug=debug_flag,
                                                                   train_debug_sample_num=train_debug_sample_num,
                                                                   test_debug_sample_num=test_debug_sample_num,
                                                                   random_seed=random_seed, data_root=args.data_root)
    if local_rank == 0:
        for i, loader in enumerate(train_loaders):
            logger.info('{} train loader has {} images'.format(i, len(loader.dataset)))
        # build model and corresponding optimizer, scheduler
        logger.info('Trainer class is {}'.format(args.trainer))
    #
    find_unused_parameters = control_cfg.get('find_unused_parameters', False)
    sync_bn = control_cfg.get('sync_bn', False)
    model_related_results = parse_args_for_models(cfg['models'], find_unused_parameters=find_unused_parameters,
                                                  sync_bn=sync_bn)
    model_dict, optimizer_dict, scheduler_dict = model_related_results
    #
    if control_cfg.get('save_init_model', None):
        tmp_path = os.path.join(logdir, 'init_model.pth')
        tmp_res = {}
        for key, item in model_dict.items():
            tmp_res[key] = item.state_dict()
        torch.save(tmp_res, tmp_path)
    # cudnn settings
    torch.backends.cudnn.enabled = True
    if control_cfg['cudnn_deterministic']:
        if local_rank == 0:
            logger.info('Using cudnn deterministic model')
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
    #
    # gather trainer args
    training_args = cfg['train']
    training_args.update({
        'type': args.trainer,
        'local_rank': args.local_rank,
        'model_dict': model_dict,
        'optimizer_dict': optimizer_dict,
        'scheduler_dict': scheduler_dict,
        'train_loaders': train_loaders,
        'logdir': logdir,
        'log_interval': control_cfg['log_interval']
    })
    #
    pretrained_model = control_cfg.get('pretrained_model', None)
    checkpoint_file = control_cfg.get('checkpoint', None)
    # build trainer
    trainer = build_trainer(training_args)
    trained_iteration = 0
    # load pretrained weights
    if pretrained_model is not None:
        if '~' in pretrained_model:
            pretrained_model = os.path.expanduser(pretrained_model)
        assert os.path.isfile(pretrained_model), '{} is not a weight file'.format(pretrained_model)
        if local_rank == 0:
            logger.info('Load pretrained model in {}'.format(pretrained_model))
        trainer.load_pretrained_model(pretrained_model)
    # resume training from checkpoint
    if checkpoint_file is not None:
        if '~' in checkpoint_file:
            checkpoint_file = os.path.expanduser(checkpoint_file)
        trainer.resume_training(checkpoint_file)
        trained_iteration = trainer.get_trained_iteration_from_scheduler()
    #
    # build validator
    test_args = cfg['test']
    test_args.update(
        {
            'type': args.validator,
            'local_rank': args.local_rank,
            'model_dict': model_dict,
            'test_loaders': test_loaders,
            'logdir': logdir,
            'trainer': trainer,
        }
    )
    validator = build_validator(test_args)
    ########################################
    log_interval = control_cfg['log_interval']
    updater_iter = control_cfg.get('update_iter', 1)
    scheduler_step = SchedulerStep(updater_iter)
    trainer.register_hook(scheduler_step, priority='VERY_LOW')
    # 注册训练的hook
    if args.local_rank == 0:
        lr_recoder = LrRecorder(log_interval)
        train_time_recoder = TrainTimeRecoder(log_interval)
        trainer.register_hook(lr_recoder, priority='HIGH')
        trainer.register_hook(train_time_recoder)
        save_model_hook = SaveModel(control_cfg['max_save_num'], save_interval=control_cfg['save_interval'])
        trainer.register_hook(save_model_hook,
                              priority='LOWEST')  # save model after scheduler step to get the right iteration number
    # test mode: only conduct test process
    test_mode = control_cfg.get('test_mode', False)
    if test_mode:
        validator(trainer.iteration)
        exit(0)
    # 处理val_interval
    val_point_list = deal_with_val_interval(control_cfg['val_interval'], max_iters=control_cfg['max_iters'],
                                            trained_iteration=trained_iteration)
    # 训练和测试交替的流程
    last_val_point = trained_iteration
    for val_point in val_point_list:
        # 训练
        trainer(train_iteration=val_point - last_val_point)
        time.sleep(2)
        # 测试
        save_flag, early_stop_flag = validator(trainer.iteration)
        #
        if save_flag:
            save_path = os.path.join(trainer.logdir, "best_model.pth".format(trainer.iteration))
            torch.save(trainer.state_dict(), save_path)
        #
        if early_stop_flag:
            logger.info("Early stop as iteration {}".format(val_point))
            break
        #
        last_val_point = val_point
    #
    tb_writer.close()