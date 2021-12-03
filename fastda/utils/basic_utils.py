# Author: Vincent Zhang
# Mail: zhyx12@gmail.com
# ----------------------------------------------
import torch
from collections.abc import Sequence
from mmcv.runner import get_dist_info
from mmcv.parallel import MMDistributedDataParallel


def move_data_to_gpu(cpu_data, gpu_id):
    relocated_data = cpu_data
    if isinstance(cpu_data, Sequence):
        for ind, item in enumerate(cpu_data):
            relocated_data[ind] = move_data_to_gpu(item, gpu_id)
    elif isinstance(cpu_data, dict):
        for key, item in cpu_data.items():
            relocated_data[key] = move_data_to_gpu(item, gpu_id)
    elif isinstance(cpu_data, torch.Tensor):
        if cpu_data.device == torch.device('cpu'):
            return cpu_data.to(gpu_id)
    return relocated_data


def move_models_to_gpu(model, device, max_card=0, find_unused_parameters=False):
    #
    rank, world_size = get_dist_info()
    #
    tmp_rank = rank * max_card + device
    model = model.to('cuda:{}'.format(tmp_rank))
    model = MMDistributedDataParallel(model, device_ids=[tmp_rank],
                                    output_device=tmp_rank,
                                    find_unused_parameters=find_unused_parameters)
    return model


def deal_with_val_interval(val_interval, max_iters, trained_iteration=0):
    fine_grained_val_checkpoint = []

    def reduce_trained_iteration(val_checkpoint):
        new_val_checkpoint = []
        start_flag = False
        for tmp_checkpoint in val_checkpoint:
            if start_flag:
                new_val_checkpoint.append(tmp_checkpoint)
            else:
                if tmp_checkpoint >= trained_iteration:
                    if tmp_checkpoint > trained_iteration:
                        new_val_checkpoint.append(tmp_checkpoint)
                    start_flag = True
        return new_val_checkpoint

    if isinstance(val_interval, (int, float)):
        val_times = int(max_iters / val_interval)
        for i in range(1, val_times + 1):
            fine_grained_val_checkpoint.append(i * int(val_interval))
        if fine_grained_val_checkpoint[-1] != max_iters:
            fine_grained_val_checkpoint.append(max_iters)
        return reduce_trained_iteration(fine_grained_val_checkpoint)
    elif isinstance(val_interval, dict):
        current_checkpoint = 0
        milestone_list = sorted(val_interval.keys())
        assert milestone_list[0] > 0 and milestone_list[-1] <= max_iters, 'check val interval keys'
        # 如果最后一个不是max_iter，则按最后的interval计算
        if milestone_list[-1] != max_iters:
            val_interval[max_iters] = val_interval[milestone_list[-1]]
            milestone_list.append(max_iters)
        last_milestone = 0
        for milestone in milestone_list:
            tmp_interval = val_interval[milestone]
            tmp_val_times = int((milestone - last_milestone) / tmp_interval)
            for i in range(tmp_val_times):
                fine_grained_val_checkpoint.append(current_checkpoint + int(tmp_interval))
                current_checkpoint += int(tmp_interval)
            if fine_grained_val_checkpoint[-1] != milestone:
                fine_grained_val_checkpoint.append(milestone)
                current_checkpoint = milestone
            last_milestone = current_checkpoint
        return reduce_trained_iteration(fine_grained_val_checkpoint)
    else:
        raise RuntimeError('only single value or dict is acceptable for val interval')


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output


if __name__ == "__main__":
    pass
