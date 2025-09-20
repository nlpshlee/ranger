from _init import *

import logging, time
from typing import List
from datasets import load_dataset

from ranger.modules import common_util, container_util, json_util
from ranger.vllm.vllm_agent import VllmAgent
from ranger.vllm.vllm_client import VllmClient
from ranger.vllm.vllm_engine import VllmEngine
from ranger.corag_agent import QueryResult, ChainResult, CoRagAgent


logging.getLogger("httpx").setLevel(logging.WARNING)


def download_datas(path, name, split, out_file_path):
    datas = load_dataset(path, name, split=split)
    datas = datas.to_list()
    json_util.write_file(datas, out_file_path)


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
            vllm_called_cnt_size = len(self._vllm_called_cnts)
            if vllm_called_cnt_size > 0:
                called_cnt = called_cnt - self._vllm_called_cnts[vllm_called_cnt_size-1]
            self._vllm_called_cnts.append(called_cnt)


    def print_time(self, print_num=5):
        print(f'# ChainGenerateTime.print_time()\n')

        batch_len = len(self._chain_generate_times)
        batch_start = max(1, batch_len - print_num + 1)

        for i, chain_time in enumerate(self._chain_generate_times[-print_num:]):
            chain_size = self._chain_generate_sizes[-print_num:][i]
            called_cnt = self._vllm_called_cnts[-print_num:][i]
            chain_time_avg = chain_time / chain_size

            batch_idx = batch_start + i
            print(f'\t[{batch_idx} batch] chain_size : {chain_size}, chain_time : {chain_time} (avg : {chain_time_avg:.2f}), vllm_called : {called_cnt}')
        
        chain_size_all = sum(self._chain_generate_sizes)
        chain_time_all = sum(self._chain_generate_times)
        called_cnt_all = sum(self._vllm_called_cnts)
        print(f'\n\t[all] chain_size : {chain_size_all}, chain_time : {chain_time_all:.2f}, vllm_called : {called_cnt_all}')
        print(f'\t[avg] chain_time : {(chain_time_all / chain_size_all):.2f}, vllm_called : {(called_cnt_all / chain_size_all):.2f}\n')





class ChainGenerator:
    def __init__(self, vllm_config: dict, use_gpu_ids=[0, 1]):
        self._vllm_config = vllm_config
        self._use_gpu_ids = use_gpu_ids

        self._vllm_agent: VllmAgent = None
        self._corag_agent: CoRagAgent = None
        self._chain_generate_time = ChainGenerateTime()

        self._init_agents()
        common_util.check_gpu_memory(self._use_gpu_ids, '[Chain Generator Init]')


    def _init_agents(self):
        # 서버 통신용
        if all(key in self._vllm_config for key in ['model_name', 'host', 'port', 'api_key']):
            self._vllm_agent = VllmClient(
                model_name=self._vllm_config['model_name'],
                host=self._vllm_config['host'],
                port=self._vllm_config['port'],
                api_key=self._vllm_config['api_key']
            )
            print(f'\n# ChainGenerator._init_agents() vllm_agent is [VllmClient]. vllm_config : {self._vllm_config}\n')

        # 내부 엔진용
        elif all(key in self._vllm_config for key in ['model_name', 'device', 'gpu_memory_utilization', 'dtype']):
            self._vllm_agent = VllmEngine(
                model_name=self._vllm_config['model_name'],
                device=self._vllm_config['device'],
                gpu_memory_utilization=self._vllm_config['gpu_memory_utilization'],
                dtype=self._vllm_config['dtype'],
                max_model_len=self._vllm_config['max_model_len'],
                n_logprob=self._vllm_config['n_logprob']
            )
            print(f'\n# ChainGenerator._init_agents() vllm_agent is [VllmEngine]. vllm_config : {self._vllm_config}\n')

        else:
            print(f'\n# ChainGenerator._init_agents() vllm_config error : {self._vllm_config}\n')

        if self._vllm_agent is not None:
            self._corag_agent = CoRagAgent(
                self._vllm_agent,
                self._vllm_config['max_model_len'],
                self._vllm_config['max_token_gen'],
                self._vllm_config['temperature'],
                self._vllm_config['top_k_query'],
                self._vllm_config['top_k_sub_query']
            )


    def reset(self):
        self._corag_agent.reset()
        self._chain_generate_time.reset()


    def chain_generate(self, datas, batch_size, n_chains, chain_depth, adapter_path='', do_print=False):
        print(f'\n# ChainGenerator.chain_generate() [start] data_size : {len(datas)}, batch_size : {batch_size}, n_chains : {n_chains}, chain_depth : {chain_depth}\n')
        results = []
        data_size = len(datas)

        for i, datas_batch in enumerate(container_util.chunks(datas, batch_size)):
            start = i*batch_size
            end = min((i+1)*batch_size-1, data_size-1)
            print(f'# ChainGenerator.chain_generate() batch {i+1} : datas size : {len(datas_batch)}({start} ~ {end})\n')

            self._chain_generate_time.check_time()

            query_results: List[QueryResult] = self._corag_agent.generate_batch(
                task_desc=self._vllm_config['task_desc'],
                datas=datas_batch,
                n_chains=n_chains,
                chain_depth=chain_depth,
                adapter_path=adapter_path
            )

            self._chain_generate_time.check_time(
                len(datas_batch) * n_chains,
                self._corag_agent._vllm_agent._called_cnt
            )

            if do_print:
                for query_result in query_results:
                    print(f'query_id : {query_result._query_id}, query : {query_result._query}, answer : {query_result._answer}\n')

                    chain_results: List[ChainResult] = query_result._chain_results
                    for chain_idx, chain_result in enumerate(chain_results):
                        print(f'chain_idx : {chain_idx}')
                        chain_result.print_chain()
                print()
            
            results.append(query_results)
        
        # 체인 생성 경과 시간 출력
        print(f'\n# ChainGenerator.chain_generate() [end] data_size : {len(datas)}, batch_size : {batch_size}, n_chains : {n_chains}, chain_depth : {chain_depth}\n')
        self._chain_generate_time.print_time()

        return results





def get_vllm_config_client():
    return {
        "model_name": "meta-llama/Meta-Llama-3-8B-Instruct",
        "task_desc": "answer multi-hop questions",
        'host': 'localhost',
        'port': 7030,
        'api_key': 'token-123',
        'max_model_len': 4096,
        'max_token_gen': 128,
        'temperature': 0.7,
        'top_k_query': 20,
        'top_k_sub_query': 5
    }


def get_vllm_config_engine():
    return {
        "model_name": "meta-llama/Meta-Llama-3-8B-Instruct",
        "task_desc": "answer multi-hop questions",
        'device': 'cuda:1',
        'gpu_memory_utilization': 0.9,
        'dtype': 'float16',
        'max_model_len': 4096,
        'max_token_gen': 128,
        'temperature': 0.7,
        'top_k_query': 20,
        'top_k_sub_query': 5
    }


if __name__ == "__main__":
    work_dir = f'/home/nlpshlee/dev_env/git/repos/ranger'
    data_dir = f'{work_dir}/data/corag'

    # vllm_config = get_vllm_config_client()
    vllm_config = get_vllm_config_engine()

    chain_generator = ChainGenerator(vllm_config)

    # download_datas('corag/multihopqa', 'hotpotqa', 'train', f'{data_dir}/input/multihopqa_train.json')
    # download_datas('corag/multihopqa', 'hotpotqa', 'validation', f'{data_dir}/input/multihopqa_valid.json')

    datas = json_util.load_file(f'{data_dir}/input/multihopqa_valid.json')
    chain_generator.chain_generate(datas[:5], 2, 3, 5, do_print=True, do_reset=True)

    # chain_generator.chain_generate(datas[:100], 100, 10, 5, do_print=False, do_reset=True)
    # chain_generator.chain_generate(datas[:1000], 1000, 10, 5, do_print=False, do_reset=True)
    # chain_generator.chain_generate(datas[:1000], 500, 10, 5, do_print=False, do_reset=True)
    # chain_generator.chain_generate(datas[:1000], 100, 10, 5, do_print=False, do_reset=True)

