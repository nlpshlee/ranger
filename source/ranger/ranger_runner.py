from _init import *

import torch, random, unsloth

from ranger.grpo.grpo_config import GRPOConfig

from ranger.modules import json_util
from ranger.chain_generator import ChainGenerator
from ranger.reward_calculator import RewardCalculator
from ranger.ranger_trainer import RANGERTrainer


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
    # 1. ChainGenerator 생성 (vllm engine)
    vllm_config = {
        "model_name": model_name,
        "task_desc": "answer multi-hop questions",
        'device': 'cuda:1',
        'gpu_memory_utilization': 0.8,
        'dtype': 'float16',
        'max_model_len': 4096,
        'max_token_gen': 128,
        'temperature': 0.7,
        'top_k_query': 20,
        'top_k_sub_query': 5
    }

    chain_generator = ChainGenerator(vllm_config, USE_GPU_IDS)

    # 2. RewardCalculator 생성
    reward_model_params_path = f'{data_dir}/reward/classifier_parmas_v2.json'
    reward_calculator = RewardCalculator(reward_model_params_path, USE_GPU_IDS)

    # 3. RangerTrainer 생성
    model_config = {
        "model_name": model_name,
        'device': 'cuda:0',
        'gpu_memory_utilization': 0.7,
        'dtype': 'float16',
        'max_model_len': 4096,
        'load_in_4bit': False,
        'trust_remote_code': False,
        'lora_r': 8,
        'lora_target_modules': ["q_proj", "k_proj", "v_proj", "o_proj"],
        'lora_alpha': 8,
        'use_gradient_checkpointing': False
    }

    grpo_config = GRPOConfig(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 1,
        num_generations = 4,                        # n_chains
        max_prompt_length = 4096,                   # max_model_len
        max_completion_length = 128,                # max_token_gen
        learning_rate = 1e-6,                       # learning rate for `AdamW` optimizer
        epsilon = 0.2,                              # CLIP epsilon
        beta = 0.04,                                # KL coefficient
        max_steps = 1,
        logging_steps = 1,
        output_dir = out_dir,
        report_to = "none"
    )

    ranger_trainer = RANGERTrainer(chain_generator,
                                   reward_calculator,
                                   model_config,
                                   grpo_config,
                                   USE_GPU_IDS)

    batch_size, n_chains, chain_depth = 2, 3, 5
    ranger_trainer.train(train_datas[:5], batch_size, n_chains, chain_depth)


if __name__ == "__main__":
    work_dir = f'/home/nlpshlee/dev_env/git/repos/ranger'
    data_dir = f'{work_dir}/data'
    out_dir = f'{work_dir}/outputs'

    seed = 42
    set_seed(seed)

    train_data_path = f'{data_dir}/corag/input/multihopqa_train.json'
    test_data_path = f'{data_dir}/corag/input/multihopqa_valid.json'
    train_datas, test_datas = load_datas(train_data_path, test_data_path, seed, do_print=False)

    model_name = 'meta-llama/Meta-Llama-3-8B-Instruct'
    run_base(data_dir, f'{out_dir}/test', model_name, train_datas, test_datas)

