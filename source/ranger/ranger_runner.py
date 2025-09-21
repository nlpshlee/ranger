from _init import *

import argparse, random, torch, wandb, unsloth

from ranger.modules import common_util, json_util
from ranger.chain_generator import ChainGenerator
from ranger.reward_calculator import RewardCalculator
from ranger.reward.models.reward_model import RewardOption
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


'''
    - vllm_config, model_config 모두에서 gpu device 를 설정하도록 되어 있었는데,
        
        - accelerator 를 사용하면서, model_config 의 device 를 python 명령어 실행 시점에서 제어하도록 변경됨

            - CUDA_VISIBLE_DEVICES=0,1 accelerate launch --gpu_ids '1' ranger_runner.py
                - [0,1]번 GPU를 사용하되, accelerator 를 이용한 학습은 '1'번 GPU에서 수행
            
            - CUDA_VISIBLE_DEVICES=1 accelerate launch --gpu_ids '1' ranger_runner.py
                - 프로그램이 바라보는 GPU는 [1]번 뿐이고, 내부적으로는 '0'으로 인덱싱함
                
                - 그 상황에서 accelerator 가 '1'번을 참조하면 에러가 발생해야되는데, 내부적으로 '1'이 없으니까 '0'으로 자동 연결되는 줄 알았으나...

                    - '1'로 설정해야 물리적인 '1'번 GPU에 할당되는게 맞음... 'unsloth' 때문이라고 추측됨...


            - accelerator 에서도 여러 GPU를 사용하고 싶은 경우
                - accelerate launch --gpu_ids '0,1'


        - 이에 맞춰서, vllm_config(ChainGenerator) 도 python 명령어 실행 시점에서 제어하도록 변경

            - CUDA_VISIBLE_DEVICES=0,1 accelerate launch --gpu_ids '1' ranger_runner.py --cg-device 0
                - [0,1]번 GPU를 사용하고, ChainGenerator 는 0번 GPU에서, RANGERTrainer 는 1번 GPU에서 작업됨

                    - 이게 돌아갈 것 같았지만, 데드락에 걸림

                        - accelerate launch는 CUDA_VISIBLE_DEVICES=0,1 설정을 보고 사용 가능한 GPU가 2개라고 판단
                        - 별도의 설정이 없으면, accelerate는 GPU 개수만큼 프로세스를 실행하여 분산 학습을 시도
                        - 즉, ranger_runner.py 스크립트 전체를 실행하는 프로세스가 2개 생김
                        - 프로세스 A는 0번 GPU에, 프로세스 B는 1번 GPU에 할당됨
                        - 두 프로세스 모두 --cg-device 0 인자를 전달받아 ChainGenerator를 0번 GPU에서 초기화하려고 시도
                        - 결과적으로 두 개의 무거운 프로세스가 동시에 0번 GPU의 리소스를 차지하려고 경쟁하다가 시스템이 멈춰버림

            - 해결 방안 1 : --num_processes=1 추가

                - CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 1 --gpu_ids '1' ranger_runner.py --cg-device 0

                    - 이렇게 해서 데드락은 풀렸지만, accelerator 가 설정한 GPU 환경을 unsloth 가 무조건 따르고 있어서, ChainGenerator 가 0번 GPU에 할당되지 않고 있음

            - 해결 방안 2 : accelerate launch 사용 X

                - accelerate launch가 실행 환경을 너무 강력하게 제어하여 unsloth가 다른 모든 설정을 무시하게 만드는 것이 원인
                    - 일반 python 명령어로 스크립트를 실행하고, 코드 내에서 Accelerator가 어떤 GPU를 사용해야하는지 인자로 전달

            - 해결 방안 3 : 다시 accelerate launch 사용하고, --gpu_ids 를 CUDA_VISIBLE_DEVICES 와 동일하게 설정

                - CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 1 --gpu_ids '0,1' ranger_runner.py --cg-device 0 --rt-device 1
                    - rt_device 는 현재 상태에선 사용되지 않음
                    - 해결 안됨 : 결국 ChainGenerator 랑 RANGERTrainer 를 완전히 분리해야 함...


        - 단일 GPU를 사용할 때는 어떤 GPU 를 할당해줘도 내부적으로는 '0'번 하나만 인식되므로 별도의 옵션을 주지 않아도 됨 (디폴트 설정되어 있음)

            - CUDA_VISIBLE_DEVICES=0 accelerate launch --gpu_ids '0' ranger_runner.py --cg-device 0

            - CUDA_VISIBLE_DEVICES=1 accelerate launch --gpu_ids '0' ranger_runner.py --cg-device 0

                - '1'번 GPU에 할당되어야 하는데, '0'번 GPU에 할당됨... accelerator 가 물리적인 번호를 보고 있는 것 같음...
                - 헷갈리니까 아래처럼 옵션 안 주는걸로...

            - CUDA_VISIBLE_DEVICES=0 python ranger_runner.py
            - CUDA_VISIBLE_DEVICES=1 python ranger_runner.py


    - 최종 사용 명령어

        - 멀티 GPU (0,1번 GPU 사용)
            - 해결 안됨
        
        - 싱글 GPU (1번 GPU만 사용)
            - CUDA_VISIBLE_DEVICES=1 python ranger_runner.py
'''
def run_base(data_dir: str, out_dir: str, model_name: str, train_datas: list, test_datas: list, cg_device='0', rt_device='0'):
    # 기본(공통) Config
    dtype = 'float16'
    max_model_len = 4096
    max_token_gen = 128

    # 1. ChainGenerator 생성 (vllm engine)
    vllm_config = {
        "model_name": model_name,
        'device': f'cuda:{cg_device}',
        'gpu_memory_utilization': 0.40,
        'dtype': dtype,
        'max_model_len': max_model_len,
        'max_token_gen': max_token_gen,
        'top_k_query': 20,                          # main query 검색 문서 수
        'top_k_sub_query': 5,                       # sub query  검색 문서 수
        'temperature': 0.7,                         # 체인이 다양하게 생성되어야 하기 때문에, 높은 값 할당
        'n_logprob': 20,                            # 정답에 대한 모델의 confidence 계산 시에 확인하려는 상위 N개의 토큰 수 (최대 20까지만 가능)
        "task_desc": "answer multi-hop questions"
    }

    chain_generator = ChainGenerator(vllm_config, USE_GPU_IDS)

    # 2. RewardCalculator 생성
    reward_config = {
        'reward_option': RewardOption.ALL,
        'penalty_short': 0.2,
        'penalty_long': 0.3,
        'missing_prob': 0.001                       # 정답 토큰이 상위 N개 토큰에 없는 경우의 패널티 확률 (내부적으론 log로 변환해서 사용됨)
    }

    reward_calculator = RewardCalculator(reward_config, USE_GPU_IDS)

    # 3. RangerTrainer 생성
    model_config = {
        "model_name": model_name,
        'device': f'cuda:{rt_device}',
        'gpu_memory_utilization': 0.40,
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
        'gradient_accumulation_steps': 20,           # Gradient Accumulation (실제 loss가 반영되는 가상의 배치 사이즈)
        'learning_rate': 1e-5,                       # learning rate for `AdamW` optimizer
        'epsilon': 0.2,                              # CLIP epsilon
        'beta': 0.04                                 # KL coefficient
    }

    ranger_trainer = RANGERTrainer(chain_generator,
                                   reward_calculator,
                                   model_config,
                                   grpo_config,
                                   out_dir,
                                   USE_GPU_IDS)

    '''
        - batch_size 는 '1'로 고정하고, 'Accelerate'를 이용하여 'Gradient Accumulation' 적용
            - batch_size 를 키우면, OOM 발생
    '''
    epochs, batch_size, n_chains, chain_depth = 10, 1, 5, 5

    wandb.init(
        # project=f'RANGER-GRPO-Training [{common_util.get_datetime_now("%Y%m%d %H %M %S")}]',
        project=f'RANGER-GRPO-Training-250921-2',
        config={
            'model_name': model_name,
            'max_model_len': max_model_len,
            'max_token_gen': max_token_gen,
            'search_top_k_query': vllm_config['top_k_query'],
            'search_top_k_sub_query': vllm_config['top_k_sub_query'],
            'temperature': vllm_config['temperature'],
            'lora_r': model_config['lora_r'],
            'lora_target_modules': model_config['lora_target_modules'],
            'lora_alpha': model_config['lora_alpha'],
            'gradient_accumulation_steps': grpo_config['gradient_accumulation_steps'],
            'epochs': epochs,
            'batch_size': batch_size,
            'n_chains': n_chains,
            'chain_depth': chain_depth
        }
    )

    print(f'\n# ranger_runner.run_base() start datetime : {common_util.get_datetime_now()}\n')
    ranger_trainer.train(train_datas[:100], test_datas[:10], epochs, batch_size, n_chains, chain_depth)
    print(f'\n# ranger_runner.run_base() end datetime : {common_util.get_datetime_now()}\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cg-device', type=str, default='0', help='Device for ChainGenerator (VLLM)')
    parser.add_argument('--rt-device', type=str, default='0', help='Device for RANGERTrainer')
    args = parser.parse_args()

    work_dir = f'/home/nlpshlee/dev_env/git/repos/ranger'
    data_dir = f'{work_dir}/data'
    out_dir = f'{work_dir}/output'

    seed = 42
    set_seed(seed)

    train_data_path = f'{data_dir}/corag/input/v1/custom_musique_train_5000_final.jsonl'
    test_data_path = f'{data_dir}/corag/input/v1/custom_multihopqa_eval_1000.jsonl'
    train_datas, test_datas = load_datas(train_data_path, test_data_path, seed, do_print=False)

    # model_name = 'meta-llama/Meta-Llama-3-8B-Instruct'
    model_name = 'meta-llama/Llama-3.2-3B-Instruct'
    run_base(data_dir, f'{out_dir}/test', model_name, train_datas, test_datas, args.cg_device, args.rt_device)

