from _init import *

from ranger.modules.common_const import *

from datetime import datetime
import time, torch

###########################################################################################

def logging(msg: str, option=LOG_OPTION.STDOUT):
    if check_option(option, LOG_OPTION.STDOUT):
        print(msg)
    if check_option(option, LOG_OPTION.STDERR):
        print(msg, file=sys.stderr)


def logging_error(call_path: str, e: Exception):
    logging(f"### (ERROR) {call_path} error : {e}\n", LOG_OPTION.STDERR)


def check_option(option1: int, option2: int):
    if option1 == option2:
        return True
    elif (option1 & option2) != 0:
        return True
    else:
        return False


def rm_option(option1: int, option2: int):
    return (option1 | option2) ^ option2


def get_time_ms():
    return int(round(time.time_ns() / 1000000))


def get_datetime_now(expression: str="%Y-%m-%d %H:%M:%S"):
    date_now = datetime.now()
    return date_now.strftime(expression)


def check_gpu_memory(devices=[0], prefix='', do_print=True):
    if not torch.cuda.is_available():
        if do_print:
            logging(f'# common_util.check_gpu_memory() [GPU] CUDA not available')

        return -1, -1
    else:
        allocated_all, reserved_all = 0, 0

        for device in devices:
            allocated = torch.cuda.memory_allocated(device)
            reserved  = torch.cuda.memory_reserved(device)
            allocated_all += allocated
            reserved_all += reserved

            if do_print:
                logging(f'# common_util.check_gpu_memory() {prefix} [GPU:{device}] Allocated: {allocated/1e9:.2f} GB | Reserved: {reserved/1e9:.2f} GB')
        if do_print:
            if len(devices) > 1:
                logging(f'# common_util.check_gpu_memory() {prefix} [GPU:Total] Allocated: {allocated_all/1e9:.2f} GB | Reserved: {reserved_all/1e9:.2f} GB\n')
            else:
                logging('')
        
        return allocated_all, reserved_all

