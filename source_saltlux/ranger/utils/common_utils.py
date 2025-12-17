import time, torch, pynvml
from datetime import datetime

from ranger.utils.common_const import *


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


def get_time_ms():
    return int(round(time.time_ns() / 1000000))


def get_datetime_now(expression: str="%Y-%m-%d %H:%M:%S"):
    date_now = datetime.now()
    return date_now.strftime(expression)


def get_elapsed_time_ms(start, end=None):
    if end is None:
        end = get_time_ms()

    elapsed_ms = end - start

    elapsed = elapsed_ms // 1000
    h = elapsed // 3600
    m = (elapsed % 3600) // 60
    s = elapsed % 60

    elapsed_str = f'{h:02d}:{m:02d}:{s:02d}'

    return elapsed_ms, elapsed_str


def check_gpu_memory(devices: list=None, do_torch_clear=True, do_print=True, msg=''):
    if not torch.cuda.is_available():
        if do_print:
            logging(f'# common_util.check_gpu_memory() CUDA not available')
        
        return -1, -1, -1
    else:
        if do_print:
            logging('')

        if do_torch_clear:
            torch.cuda.empty_cache()
        
        pynvml.nvmlInit()

        if devices is None:
            device_count = pynvml.nvmlDeviceGetCount()
            devices = list(range(device_count))

        total_all, used_all, free_all = 0, 0, 0

        for device in devices:
            handle = pynvml.nvmlDeviceGetHandleByIndex(device)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

            total = mem_info.total
            used = mem_info.used
            free = mem_info.free

            total_all += total
            used_all += used
            free_all += free

            if do_print:
                logging(f'# common_util.check_gpu_memory() {msg} [GPU:{device}] total: {total/1e9:.2f} GB, used: {used/1e9:.2f} GB, free: {free/1e9:.2f} GB')
        
        if do_print:
            if len(devices) > 1:
                logging(f'# common_util.check_gpu_memory() {msg} [GPU:Total] total: {total_all/1e9:.2f} GB, used: {used_all/1e9:.2f} GB, free: {free_all/1e9:.2f} GB\n')
            else:
                logging('')
        
        pynvml.nvmlShutdown()

        return total_all, used_all, free_all

