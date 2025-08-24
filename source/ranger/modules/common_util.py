from _init import *

from ranger.modules.common_const import *

from datetime import datetime
import time, torch, pynvml

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


def check_gpu_memory_of_torch(devices=[0], prefix='', do_print=True):
    if not torch.cuda.is_available():
        if do_print:
            logging(f'# common_util.check_gpu_memory_of_torch() [GPU] CUDA not available')

        return -1, -1
    else:
        allocated_all, reserved_all = 0, 0

        for device in devices:
            allocated = torch.cuda.memory_allocated(device)
            reserved  = torch.cuda.memory_reserved(device)
            allocated_all += allocated
            reserved_all += reserved

            if do_print:
                logging(f'# common_util.check_gpu_memory_of_torch() {prefix} [GPU:{device}] Allocated: {allocated/1e9:.2f} GB | Reserved: {reserved/1e9:.2f} GB')
        if do_print:
            if len(devices) > 1:
                logging(f'# common_util.check_gpu_memory_of_torch() {prefix} [GPU:Total] Allocated: {allocated_all/1e9:.2f} GB | Reserved: {reserved_all/1e9:.2f} GB\n')
            else:
                logging('')
        
        return allocated_all, reserved_all


'''
    위 함수처럼 torch.cuda.memory_allocated()를 사용하면,  PyTorch의 메모리 할당기(Caching Allocator)가 관리하는 텐서들이 사용하는 메모리만 집계
    
    vLLM은 PagedAttention 기법을 위해 자체적으로 거대한 메모리 풀(KV Cache용)을 GPU에 직접 할당
    이 때,  메모리 할당은 PyTorch의 관리 범위를 벗어나서 이루어짐
    따라서 torch.cuda.memory_allocated() 함수는 vLLM이 사용하는 메모리 중 모델 가중치(weights) 등 PyTorch가 직접 관리하는 일부만 보여주고,
    가장 큰 부분을 차지하는 KV Cache 메모리 사용량은 놓치게됨

    # 해결법
    'pip install pynvml'으로 라이브러리 설치하고 아래처럼 코드 변경
'''
def check_gpu_memory(devices=[0], prefix='', do_print=True):
    if not torch.cuda.is_available():
        if do_print:
            logging(f'# common_util.check_gpu_memory() [GPU] CUDA not available')

        return -1, -1, -1
    else:
        pynvml.nvmlInit()
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
                logging(f'# common_util.check_gpu_memory() {prefix} [GPU:{device}] total: {total/1e9:.2f} GB, used: {used/1e9:.2f} GB, free: {free/1e9:.2f} GB')
        if do_print:
            if len(devices) > 1:
                logging(f'# common_util.check_gpu_memory() {prefix} [GPU:Total] total: {total_all/1e9:.2f} GB, used: {used_all/1e9:.2f} GB, free: {free_all/1e9:.2f} GB\n')
            else:
                logging('')
        
        pynvml.nvmlShutdown()
        return total_all, used_all, free_all

