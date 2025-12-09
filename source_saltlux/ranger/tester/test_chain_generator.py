from _init import *

import torch, random
from typing import List

from ranger.utils import json_utils
from ranger.vllm.vllm_engine import VllmEngine
from ranger.corag.corag_agent import CoRagAgent
from ranger.corag.corag_result import ChainResult, QueryResult


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)

seed = 42
set_seed(seed)


model_name = 'meta-llama/Llama-3.2-3B-Instruct'
cg_device = 0
dtype = 'float16'
max_model_len = 4096
max_token_gen = 128

vllm_config = {
    "model_name": model_name,
    'device': f'cuda:{cg_device}',
    'gpu_memory_utilization': 0.30,
    'dtype': dtype,
    'max_model_len': max_model_len,
    'max_token_gen': max_token_gen,
    'top_k_query': 20,                          # main query 검색 문서 수
    'top_k_sub_query': 5,                       # sub query  검색 문서 수
    'temperature': 0.0,                         # 체인이 다양하게 생성되어야 하기 때문에, 높은 값 할당
    'n_logprob': 20,                            # 정답에 대한 모델의 confidence 계산 시에 확인하려는 상위 N개의 토큰 수 (최대 20까지만 가능)
    "task_desc": "answer multi-hop questions"
}


vllm_engine = VllmEngine(
    model_name=vllm_config['model_name'],
    device=vllm_config['device'],
    gpu_memory_utilization=vllm_config['gpu_memory_utilization'],
    dtype=vllm_config['dtype'],
    max_model_len=vllm_config['max_model_len'],
    n_logprob=vllm_config['n_logprob']
)


corag_agent = CoRagAgent(
    vllm_engine,
    vllm_config['max_model_len'],
    vllm_config['max_token_gen'],
    vllm_config['temperature'],
    vllm_config['top_k_query'],
    vllm_config['top_k_sub_query']
)


def datas_shuffle(datas: list, seed: int):
    rng = random.Random(seed)
    rng.shuffle(datas)


def load_datas(train_data_path: str, test_data_path: str, seed: int, do_print=False):
    train_datas = json_utils.load_file(train_data_path)
    test_datas = json_utils.load_file(test_data_path)
    datas_shuffle(train_datas, seed)
    datas_shuffle(test_datas, seed)

    if do_print:
        print(f'\n# ranger_runner.load_datas() train_datas[0] : {json_utils.to_str(train_datas[0])}')
        print(f'# ranger_runner.load_datas() test_datas[0] : {json_utils.to_str(test_datas[0])}\n')
    
    return train_datas, test_datas


def test_generate_batch(datas, n_chains, chain_depth, adapter_path=''):
    query_results: List[QueryResult] = corag_agent.generate_batch(
        task_desc=vllm_config['task_desc'],
        datas=datas,
        n_chains=n_chains,
        chain_depth=chain_depth,
        adapter_path=adapter_path
    )

    return query_results


work_dir = f'/home/nlpshlee/dev_env/git/repos/ranger'
data_dir = f'{work_dir}/data'
out_dir = f'{work_dir}/output'

train_data_path = f'{data_dir}/custom_musique_train_5000_final.jsonl'
test_data_path = f'{data_dir}/custom_multihopqa_eval_1000.jsonl'
train_datas, test_datas = load_datas(train_data_path, test_data_path, seed, do_print=False)


n_chains, chain_depth = 5, 5

query_results = test_generate_batch(train_datas[:1], n_chains, chain_depth)


def print_query_results(query_results: List[QueryResult]):
    print(f'query_results size : {len(query_results)}\n')

    for query_result in query_results:
        print(f'query_id : {query_result._query_id}')
        print(f'query : {query_result._query}')
        print(f'answers : {query_result._answers}')
        print(f'doc_ids : {query_result._doc_ids}')
        print(f'documents :')
        for i, document in enumerate(query_result._documents):
            document = document.replace('\n', ' ')
            print(f'[{i+1}] : {document}')

        chain_results: List[ChainResult] = query_result._chain_results
        print(f'\n\tchain_results size : {len(chain_results)}\n')

        for chain_idx, chain_result in enumerate(chain_results):
            print(f'\tchain_idx : {chain_idx+1}')
            print(f'\tsub_querys : {chain_result._sub_querys}')
            print(f'\tsub_answers : {chain_result._sub_answers}')
            print(f'\tdoc_ids_list : {chain_result._doc_ids_list}')
            print(f'\tdocuments_list :\n\t[depth][document_idx]')
            for i, documents in enumerate(chain_result._documents_list):
                for j, document in enumerate(documents):
                    document = document.replace('\n', ' ')
                    print(f'\t[{i+1}][{j+1}] : {document}')

            print(f'\tfinal_answers : {chain_result._final_answers}')
            print(f'\tcompletion_outputs : {chain_result._completion_outputs}\n')



# 이 결과를 파일로 저장해서, diff 비교
print_query_results(query_results)