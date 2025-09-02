from _init import *

import torch, random, unsloth

from ranger.modules import common_util, json_util
from ranger.chain_generator import ChainGenerator
from ranger.reward_calculator import RewardCalculator
from ranger.ranger_trainer import RANGERTrainer

'''
    - 현재 0번 GPU를 다른 곳에서 사용하고 있어서, 1번으로만 실행시켜야 되는 상황
        - 하지만, unsloth에서 import 단계에서 서버의 모든 GPU를 초기화 함
        - 이 과정에서 이미 0번 GPU가 full로 사용 중이라 초기화하다가 OOM 발생

            - Python 명령어를 다음과 같이 변경
                - 기존 : python -u -m ranger_runner > ../../logs/ranger_test.log
                - 변경 : CUDA_VISIBLE_DEVICES=1 python -u -m ranger_runner > ../../logs/ranger_test.log
        
        - 이렇게 1번 GPU만 보이게 설정해주는건데, 그렇게 되면 파이썬에서는 1번 GPU 1개만 있는 걸로 인식하기 때문에
        - 'cuda:0'과 같이 코드를 0번 GPU로 설정해줘야 함

            - 추가로, 1,3번을 사용한다고 하면 'CUDA_VISIBLE_DEVICES=1,3'으로 설정해주고, 코드 레벨에서는 'cuda:0', 'cuda:1'로 설정
        
        - 하지만, 아래 USE_GPU_IDS 변수는 common_util.check_gpu_memory() 에서 사용되는데, 실제 GPU ID를 모두 넘겨줘야 함
            - 아마도 사용되는 라이브러리(pynvml)가 물리적으로 접근하는 것 같음
'''
USE_GPU_IDS = [0, 1]


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)


def datas_shuffle(datas: list, seed: int):
    rng = random.Random(seed)
    rng.shuffle(datas)


def load_datas(train_data_path: str, test_data_path: str, seed: int, do_print=False):
    train_datas = json_util.load_file(train_data_path)
    test_datas = json_util.load_file(test_data_path)
    datas_shuffle(train_datas, seed)
    datas_shuffle(test_datas, seed)

    if do_print:
        print(f'\n# ranger_runner.load_datas() train_datas[0] : {json_util.to_str(train_datas[0])}')
        print(f'# ranger_runner.load_datas() test_datas[0] : {json_util.to_str(test_datas[0])}\n')
    
    return train_datas, test_datas


def run_base(data_dir: str, out_dir: str, model_name: str, train_datas: list, test_datas: list):
    # 기본(공통) Config
    dtype = 'float16'
    max_model_len = 4096
    max_token_gen = 128

    # 1. ChainGenerator 생성 (vllm engine)
    vllm_config = {
        "model_name": model_name,
        "task_desc": "answer multi-hop questions",
        'device': 'cuda:0',
        'gpu_memory_utilization': 0.30,
        'dtype': dtype,
        'max_model_len': max_model_len,
        'max_token_gen': max_token_gen,
        'top_k_query': 20,                          # main query 검색 문서 수
        'top_k_sub_query': 5,                       # sub query  검색 문서 수
        'temperature': 0
    }

    chain_generator = ChainGenerator(vllm_config, USE_GPU_IDS)

    # 2. RewardCalculator 생성
    reward_model_params_path = f'{data_dir}/reward/classifier_parmas_v2.json'
    reward_calculator = RewardCalculator(reward_model_params_path, USE_GPU_IDS)

    # 3. RangerTrainer 생성
    model_config = {
        "model_name": model_name,
        'device': 'cuda:0',
        'gpu_memory_utilization': 0.30,
        'dtype': dtype,
        'max_model_len': max_model_len,
        'max_token_gen': max_token_gen,
        'load_in_4bit': False,
        'trust_remote_code': False,
        'lora_r': 8,
        'lora_target_modules': ["q_proj", "k_proj", "v_proj", "o_proj"],
        'lora_alpha': 8,
        'use_gradient_checkpointing': False
    }

    grpo_config = {
        'learning_rate': 1e-6,                       # learning rate for `AdamW` optimizer
        'epsilon': 0.2,                              # CLIP epsilon
        'beta': 0.04                                 # KL coefficient
    }

    ranger_trainer = RANGERTrainer(chain_generator,
                                   reward_calculator,
                                   model_config,
                                   grpo_config,
                                   out_dir,
                                   USE_GPU_IDS)

    print(f'\n# ranger_runner.run_base() start time : {common_util.get_datetime_now()}\n')

    epochs, batch_size, n_chains, chain_depth = 100, 2, 3, 5
    ranger_trainer.train(train_datas[:5], batch_size, n_chains, chain_depth)

    print(f'\n# ranger_runner.run_base() end time : {common_util.get_datetime_now()}\n')


if __name__ == "__main__":
    work_dir = f'/home/nlpshlee/dev_env/git/repos/ranger'
    data_dir = f'{work_dir}/data'
    out_dir = f'{work_dir}/output'

    seed = 42
    set_seed(seed)

    train_data_path = f'{data_dir}/corag/input/multihopqa_train.json'
    test_data_path = f'{data_dir}/corag/input/multihopqa_valid.json'
    train_datas, test_datas = load_datas(train_data_path, test_data_path, seed, do_print=False)

    # model_name = 'meta-llama/Meta-Llama-3-8B-Instruct'
    model_name = 'meta-llama/Llama-3.2-3B-Instruct'
    run_base(data_dir, f'{out_dir}/test', model_name, train_datas, test_datas)

