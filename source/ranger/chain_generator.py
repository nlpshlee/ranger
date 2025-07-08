from _init import *

import os, json, logging, string, time, statistics
from typing import Dict, List
from tqdm import tqdm
from datasets import load_dataset
from datasets.arrow_dataset import Dataset

from ranger.modules import file_util, container_util, json_util
from ranger.corag.agent.agent_utils import RagPath
from ranger.corag.data_utils import format_documents_for_final_answer, parse_answer_logprobs
from ranger.corag.inference.qa_utils import _normalize_answer
from ranger.corag.prompts import get_generate_intermediate_answer_prompt
from ranger.corag.search.search_utils import search_by_http
from ranger.corag.agent.corag_agent_batch_for_queries import CoRagAgent
from ranger.corag.vllm_client import VllmClient


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
        self._vllm_client = VllmClient(
            model=self._vllm_config['model_name'],
            host=self._vllm_config['host'],
            port=self._vllm_config['port'],
            api_key=self._vllm_config['api_key']
        )

        self._corag_agent = CoRagAgent(vllm_client=self._vllm_client)

        self._chain_generate_time = ChainGenerateTime()


    def chain_generate(self, datas, batch_size, n_chains, chain_depth, do_print=False):
        for i, datas_batch in enumerate(container_util.chunks(datas, batch_size)):
            print(f'batch {i+1} : datas size : {len(datas_batch)}({i*batch_size} ~ {(i+1)*batch_size-1})\n')

            self._chain_generate_time.check_time()
            chains = self._corag_agent.batch_chain_generate(
                task_desc=self._vllm_config['task_desc'],
                datas=datas_batch,
                n_chains=n_chains,
                chain_depth=chain_depth
            )
            self._chain_generate_time.check_time(
                len(datas_batch) * n_chains,
                self._corag_agent._vllm_client.called_cnt
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


if __name__ == "__main__":
    work_dir = f'/home/nlpshlee/dev_env/git/repos/ranger'
    data_dir = f'{work_dir}/data/corag'

    vllm_config = {
        "model_name": "meta-llama/Meta-Llama-3-8B-Instruct",
        "task_desc": "answer multi-hop questions",
        'host': 'localhost',
        'port': 7030,
        'api_key': 'token-123'
    }
    chain_generator = ChainGenerator(vllm_config)

    # download_datas('corag/multihopqa', 'hotpotqa', 'train', f'{data_dir}/input/multihopqa_train.json')
    # download_datas('corag/multihopqa', 'hotpotqa', 'validation', f'{data_dir}/input/multihopqa_valid.json')

    datas = json_util.load_file(f'{data_dir}/input/multihopqa_valid.json')
    # chain_generator.chain_generate(datas[:10], 4, 3, 5, do_print=True)

    chain_generator.chain_generate(datas[:100], 100, 10, 5, do_print=True)
    chain_generator.chain_generate(datas[:1000], 1000, 10, 5, do_print=True)
    chain_generator.chain_generate(datas[:1000], 500, 10, 5, do_print=True)
    chain_generator.chain_generate(datas[:1000], 100, 10, 5, do_print=True)

