from _init import *

import logging, time
from datasets import load_dataset

from ranger.modules import container_util, json_util
from ranger.vllm.vllm_agent import VllmAgent
from ranger.vllm.vllm_client import VllmClient
from ranger.vllm.vllm_engine import VllmEngine
from ranger.corag_agent import CoRagAgent


logging.getLogger("httpx").setLevel(logging.WARNING)


def download_datas(path, name, split, out_file_path):
    datas = load_dataset(path, name, split=split)
    datas = datas.to_list()
    json_util.write_file(datas, out_file_path)


class ChainGenerateTime:
    def __init__(self):
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


    def print_time(self):
        print(f'# ChainGenerateTime.print_time()\n')

        for i, chain_time in enumerate(self._chain_generate_times):
            chain_size = self._chain_generate_sizes[i]
            called_cnt = self._vllm_called_cnts[i]
            print(f'\t[{i+1} batch] chain size : {chain_size}, chain time : {chain_time}, vllm called : {called_cnt}')
        
        chain_size_all = sum(self._chain_generate_sizes)
        chain_time_all = sum(self._chain_generate_times)
        called_cnt_all = sum(self._vllm_called_cnts)
        print(f'\t[all] chain size : {chain_size_all}, chain time : {chain_time_all:.2f}, vllm called : {called_cnt_all}')
        print(f'\t[avg] chain time : {(chain_time_all / chain_size_all):.2f}, vllm called : {(called_cnt_all / chain_size_all):.2f}\n')





class ChainGenerator:
    def __init__(self, vllm_config: dict):
        self._vllm_config = vllm_config

        self._vllm_agent: VllmAgent = None
        self._corag_agent: CoRagAgent = None
        self._chain_generate_time = ChainGenerateTime()

        self._init_agents()
    

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
                max_model_len=self._vllm_config['max_model_len']
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


    def chain_generate(self, datas, batch_size, n_chains, chain_depth, do_print=False):
        print(f'\nChainGenerator.chain_generate() data_size : {len(datas)}, batch_size : {batch_size}, n_chains : {n_chains}, chain_depth : {chain_depth}\n')

        for i, datas_batch in enumerate(container_util.chunks(datas, batch_size)):
            print(f'batch {i+1} : datas size : {len(datas_batch)}({i*batch_size} ~ {(i+1)*batch_size-1})\n')

            self._chain_generate_time.check_time()

            chains = self._corag_agent.generate_batch(
                task_desc=self._vllm_config['task_desc'],
                datas=datas_batch,
                n_chains=n_chains,
                chain_depth=chain_depth
            )

            self._chain_generate_time.check_time(
                len(datas_batch) * n_chains,
                self._corag_agent._vllm_agent._called_cnt
            )

            if do_print:
                for query_result in chains:
                    print(f'query_id : {query_result._query_id}, query : {query_result._query}, answer : {query_result._answer}\n')
                    for chain_idx, chain_result in enumerate(query_result._chain_results):
                        print(f'chain_idx : {chain_idx}')
                        chain_result.print_chain()
                print()
        
        # 체인 생성 경과 시간 출력
        self._chain_generate_time.print_time()





def get_vllm_config_client():
    return {
        "model_name": "meta-llama/Meta-Llama-3-8B-Instruct",
        "task_desc": "answer multi-hop questions",
        'host': 'localhost',
        'port': 7030,
        'api_key': 'token-123',
        'max_model_len': 4096,
        'max_token_gen': 32,
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
        'max_token_gen': 32,
        'temperature': 0.7,
        'top_k_query': 20,
        'top_k_sub_query': 5
    }


if __name__ == "__main__":
    work_dir = f'/home/nlpshlee/dev_env/git/repos/ranger'
    data_dir = f'{work_dir}/data/corag'

    vllm_config = get_vllm_config_client()
    # vllm_config = get_vllm_config_engine()

    chain_generator = ChainGenerator(vllm_config)

    # download_datas('corag/multihopqa', 'hotpotqa', 'train', f'{data_dir}/input/multihopqa_train.json')
    # download_datas('corag/multihopqa', 'hotpotqa', 'validation', f'{data_dir}/input/multihopqa_valid.json')

    datas = json_util.load_file(f'{data_dir}/input/multihopqa_valid.json')
    chain_generator.chain_generate(datas[:5], 2, 3, 5, do_print=True)

    # chain_generator.chain_generate(datas[:100], 100, 10, 5, do_print=True)
    # chain_generator.chain_generate(datas[:1000], 1000, 10, 5, do_print=True)
    # chain_generator.chain_generate(datas[:1000], 500, 10, 5, do_print=True)
    # chain_generator.chain_generate(datas[:1000], 100, 10, 5, do_print=True)

