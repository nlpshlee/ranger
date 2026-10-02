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
version = '260923-8B'
out_dir = f'{work_dir}/outputs/rl/{version}'

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
    epochs : 1 에폭으로 고정
        직전 실행(260909-3)에서 2 에폭은 KL 이 1.6 -> 3.6 으로 수렴 없이 발산하고
        학습 reward 는 정체(0.250 -> 0.233)했으며, 평가 성능이 크게 하락함
        -> 1 에폭(625 optimizer step)에서 종료
'''
epochs, n_chains, chain_depth = 1, 5, 5

'''
    [멀티 GPU] 유효 배치(= optimizer step 1회에 쓰이는 쿼리 수)를 GPU 개수와 무관하게 8로 고정
        - 랭크마다 batch_size 개를 처리하므로, 랭크당 batch_size = 8 / GPU 수
        - 3B RL 실험과 같은 학습 조건(스텝당 쿼리 8개, 1 에폭 625 스텝)을 유지해야 비교가 공정함
'''
EFFECTIVE_BATCH_SIZE = 8
num_processes = ranger_trainer._accelerator.num_processes
assert EFFECTIVE_BATCH_SIZE % num_processes == 0, f'EFFECTIVE_BATCH_SIZE({EFFECTIVE_BATCH_SIZE}) 는 GPU 수({num_processes})로 나누어떨어져야 함'
batch_size = EFFECTIVE_BATCH_SIZE // num_processes

# wandb 는 메인 프로세스만 기록 (다른 랭크까지 init 하면 빈 run 이 따로 생김)
wandb.init(
    mode=None if ranger_trainer._accelerator.is_main_process else 'disabled',
    project=f'RANGER-{version}',
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
        'effective_batch_size': EFFECTIVE_BATCH_SIZE,
        'num_processes': num_processes,
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

