from _init import *

import time


class ChainGenerateTime:
    def __init__(self):
        self._prev_time: int
        self._cur_time: int

        self._chain_generate_times: list
        self._chain_generate_sizes: list
        self._vllm_called_cnts: list
        self.reset()
    

    def reset(self):
        self._prev_time = -1
        self._cur_time = -1

        self._chain_generate_times = []
        self._chain_generate_sizes = []
        self._vllm_called_cnts = []


    def check_time(self, chain_size=0, called_cnt=0):
        self._cur_time = time.time()

        if self._prev_time == -1:
            self._prev_time = self._cur_time

        if chain_size > 0:
            chain_generate_time = self._cur_time - self._prev_time
            self._chain_generate_times.append(round(chain_generate_time, 2))
            self._chain_generate_sizes.append(chain_size)
            self._prev_time = self._cur_time
        
        if called_cnt > 0:
            if len(self._vllm_called_cnts) == 0:
                self._vllm_called_cnts.append(called_cnt)
            else:
                self._vllm_called_cnts.append(called_cnt - sum(self._vllm_called_cnts))


    def print_time(self, print_num=5):
        print(f'\n# [CHAIN_TIME] ChainGenerateTime.print_time()\n')

        batch_len = len(self._chain_generate_times)
        batch_start = max(1, batch_len - print_num + 1)

        for i, chain_time in enumerate(self._chain_generate_times[-print_num:]):
            chain_size = self._chain_generate_sizes[-print_num:][i]
            called_cnt = self._vllm_called_cnts[-print_num:][i]
            chain_time_avg = chain_time / chain_size

            batch_idx = batch_start + i
            print(f'\t[{batch_idx} batch] chain_size : {chain_size}, chain_time : {chain_time} (s), (avg : {chain_time_avg:.2f}), vllm_called : {called_cnt}')
        
        chain_size_all = sum(self._chain_generate_sizes)
        chain_time_all = sum(self._chain_generate_times)
        called_cnt_all = sum(self._vllm_called_cnts)
        print(f'\n\t[all] chain_size : {chain_size_all}, chain_time : {chain_time_all:.2f} (s), vllm_called : {called_cnt_all}')
        print(f'\t[avg] chain_time : {(chain_time_all / chain_size_all):.2f} (s), vllm_called : {(called_cnt_all / chain_size_all):.2f}\n')

