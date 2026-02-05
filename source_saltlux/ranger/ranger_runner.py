from _init import *

import random, wandb

from ranger.utils import common_utils, json_utils
from ranger.reward.reward_calculator import RewardCalculator
from ranger.train.ranger_trainer import RangerTrainer


seed = COMMON_CONFIG['seed']
common_utils.set_seed(seed)


def datas_shuffle(datas: list, seed: int):
    rng = random.Random(seed)
    rng.shuffle(datas)


def load_datas(train_data_path: str, test_data_path: str, seed: int):
    train_datas = json_utils.load_file(train_data_path)
    test_datas = json_utils.load_file(test_data_path)
    datas_shuffle(train_datas, seed)
    datas_shuffle(test_datas, seed)
    
    return train_datas, test_datas


work_dir = f'/home/nlpshlee/dev_env/git/repos/ranger'
data_dir = f'{work_dir}/data'
out_dir = f'{work_dir}/outputs/test'

train_data_path = f'{data_dir}/custom_musique_train_5000_final.jsonl'
test_data_path = f'{data_dir}/custom_multihopqa_eval_1000.jsonl'
train_datas, test_datas = load_datas(train_data_path, test_data_path, seed)


reward_calculator = RewardCalculator(REWARD_CONFIG['reward_option'])

ranger_trainer = RangerTrainer(
    MODEL_CONFIG,
    reward_calculator,
    out_dir
)

epochs, batch_size, n_chains, chain_depth = 10, 1, 5, 5

wandb.init(
    project=f'RANGER-Training-260113-1',
    config={
        'model_name': VLLM_CONFIG['model_name'],
        'max_seq_length': VLLM_CONFIG['max_seq_length'],
        'max_new_tokens': VLLM_CONFIG['max_new_tokens'],
        'temperature': VLLM_CONFIG['temperature'],
        'top_k_query': CORAG_CONFIG['top_k_query'],
        'top_k_sub_query': CORAG_CONFIG['top_k_sub_query'],
        'lora_r': MODEL_CONFIG['lora_r'],
        'lora_target_modules': MODEL_CONFIG['lora_target_modules'],
        'lora_alpha': MODEL_CONFIG['lora_alpha'],
        'gradient_accumulation_steps': MODEL_CONFIG['gradient_accumulation_steps'],
        'epochs': epochs,
        'batch_size': batch_size,
        'n_chains': n_chains,
        'chain_depth': chain_depth
    }
)


ranger_trainer.train(
    train_datas, test_datas,
    epochs, batch_size, n_chains, chain_depth
)



'''
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

    CUDA_VISIBLE_DEVICES=1 python -u ranger_runner.py > ./logs/ranger_runner.log

    CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 accelerate launch  --num_processes=1  --num_machines=1  --mixed_precision=no  --dynamo_backend=no  ranger_runner.py > ./logs/ranger_runner.log

    CUDA_VISIBLE_DEVICES=0,1 accelerate launch  --num_processes=2  --num_machines=1  --mixed_precision=no  --dynamo_backend=no  ranger_runner.py > ./logs/ranger_runner.log
'''

'''
    $ CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes=2 ranger_runner.py > ./logs/ranger_runner.log

    time_sleep = 60
    print(f'\n\n{"="*100}Sleeping {time_sleep}(s)...')
    import time
    time.sleep(time_sleep)
'''

