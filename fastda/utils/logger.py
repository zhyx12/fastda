# Author: Vincent Zhang
# Mail: zhyx12@gmail.com
# ----------------------------------------------
import logging
from mmcv.utils import get_logger
from mmcv.utils.logging import logger_initialized


def get_root_logger(log_file=None, log_level=logging.INFO):
    if log_file is None:
        assert 'FastDA' in logger_initialized, 'logger not initialized'
    return get_logger('FastDA', log_file, log_level)
