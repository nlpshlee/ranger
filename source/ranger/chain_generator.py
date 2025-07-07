from _init import *

import os, json, logging, string, time
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
    

    def chain_generate(self, datas, batch_size, n_chains, chain_depth):
        for i, datas_batch in enumerate(container_util.chunks(datas, batch_size)):
            print(f'batch {i+1} : datas size : {len(datas_batch)}({i*batch_size} ~ {(i+1)*batch_size-1})\n')

            print(f'batch_chain_generate() start')
            chains = self._corag_agent.batch_chain_generate(
                task_desc=self._vllm_config['task_desc'],
                datas=datas_batch,
                n_chains=n_chains,
                chain_depth=chain_depth
            )
            print(f'batch_chain_generate() end -> print start')

            for query_result in chains:
                print(f'query_id : {query_result._query_id}, query : {query_result._query}, answer : {query_result._answer}')
                for chain_idx, chain_result in enumerate(query_result._chain_results):
                    print(f'chain_idx : {chain_idx}')
                    chain_result.print_chain()
            print()
            print(f'print end')


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
    chain_generator.chain_generate(datas[:10], 2, 3, 5)

