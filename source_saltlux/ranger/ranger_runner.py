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


work_dir = f'/raid/ai/home/jsyang/dev_env/git/repos/ranger'
data_dir = f'{work_dir}/data'
date_version = '260909-3'
out_dir = f'{work_dir}/outputs/rl/{date_version}'

train_data_path = f'{data_dir}/custom_musique_train_5000_final.jsonl'
test_data_path = f'{data_dir}/custom_multihopqa_eval_1000.jsonl'
train_datas, test_datas = load_datas(train_data_path, test_data_path, seed)


'''
    [임시] 전 구간(vLLM -> 리트리버 -> 리워드 -> 역전파 -> 어댑터 재로드) 점검용 축소 실행
        - 전체 데이터로 돌리려면 아래 3줄과 IS_SCOPED_RUN 분기만 제거하면 됨
'''
IS_SCOPED_RUN = False

if IS_SCOPED_RUN:
    train_datas = train_datas[:200]
    test_datas = test_datas[:50]


reward_calculator = RewardCalculator(REWARD_CONFIG['reward_option'])

ranger_trainer = RangerTrainer(
    MODEL_CONFIG,
    reward_calculator,
    out_dir
)

'''
    batch_size : 한 번의 롤아웃 요청에 담기는 쿼리 수 (= vLLM 병렬성)
        - vLLM 은 (batch_size x n_chains) 개 시퀀스를 한 번에 생성함
        - batch_size=1 이면 한 번에 5개만 처리해서 서빙 GPU가 대부분 놀게 됨
        - 유효 배치(batch_size x GRADIENT_ACCUMULATION_STEPS)는 8로 유지
'''
'''
    epochs : RL 은 같은 쿼리를 여러 번 보면 리워드 해킹/과적합 위험이 커짐
             프롬프트가 5000개나 되므로 2 에폭이면 충분 (1250 optimizer step)
'''
epochs, batch_size, n_chains, chain_depth = (1 if IS_SCOPED_RUN else 2), 8, 5, 5

wandb.init(
    project=f'RANGER-{date_version}',
    name=f'scoped-{len(train_datas)}' if IS_SCOPED_RUN else None,
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

