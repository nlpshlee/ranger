import os

'''
    [중요] vLLM V1 은 EngineCore 를 별도 프로세스로 띄우는데, 기본 시작 방식이 'fork' 임
        부모 프로세스에서 CUDA 컨텍스트가 한 번이라도 만들어지면 자식이 CUDA 를 다시 초기화할 수 없어
        "Cannot re-initialize CUDA in forked subprocess" 로 죽음

        spawn 으로 강제하면 자식이 깨끗한 상태로 시작하므로 이 문제가 사라짐
        (엔진 종류/추론 동작은 그대로, 프로세스 시작 방식만 바뀜)

        반드시 vllm 이 import 되기 '전에' 설정되어야 하므로 파일 최상단에 둠
'''
os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')

from _init import *

'''
    임의의 모델(공개 모델 / 학습된 체크포인트 / LoRA 어댑터)을 평가 데이터로 돌려
    source(벤치마크) 별 EM/F1 을 집계하는 독립 실행 스크립트

        - 체인 서버(HTTP)를 거치지 않고 프로세스 안에서 vLLM 엔진을 직접 띄움
          -> 학습이 돌고 있는 중에도, 남는 GPU 에서 다른 모델을 동시에 평가할 수 있음
          -> globals.yml 을 건드리지 않으므로 학습 설정과 충돌하지 않음

        - 평가 데이터가 4개 벤치마크 혼합셋이라 단일 숫자로는 공개 논문 수치와 비교할 수 없음
          반드시 source 별로 쪼갠 값을 사용할 것

    실행 예)
        # CoRAG-8B (공개 모델) - greedy
        CUDA_VISIBLE_DEVICES=1 python -u eval_model_by_source.py \
            --model corag/CoRAG-Llama3.1-8B-MultihopQA --tag corag-8b --greedy \
            > ./logs/eval_corag8b_greedy.log 2>&1

        # 우리 3B SFT 베이스라인 - best-of-5 (학습 때와 동일 프로토콜)
        CUDA_VISIBLE_DEVICES=1 python -u eval_model_by_source.py \
            --model /raid/.../outputs/sft/Llama-3.2-3B/merged-573 --tag sft-3b \
            > ./logs/eval_sft3b_bo5.log 2>&1

        # 우리 3B + RANGER (LoRA 어댑터)
        CUDA_VISIBLE_DEVICES=1 python -u eval_model_by_source.py \
            --model /raid/.../outputs/sft/Llama-3.2-3B/merged-573 \
            --adapter /raid/.../outputs/rl/260909-1/lora_adapter_XXX_epoch_1 --tag ranger-3b-ep1 \
            > ./logs/eval_ranger3b_ep1.log 2>&1
'''

import argparse
from typing import List

from ranger.utils import common_utils, json_utils, evaluation_utils
from ranger.reward.reward_calculator import RewardCalculator

'''
    ChainGenerator / corag_agent 는 import 시점에 KILT 코퍼스(3,500만 건)를 로드함
    spawn 방식에서는 자식 프로세스가 __main__ 을 다시 import 하므로,
    최상단에 두면 엔진 서브프로세스에서도 코퍼스를 또 읽게 됨 -> main() 안에서 지연 import
'''


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, help='HF 모델 이름 또는 로컬 경로 (완전한 모델이어야 함)')
    p.add_argument('--adapter', default='', help='LoRA 어댑터 경로 (선택)')
    p.add_argument('--tag', default='model', help='로그 출력에 붙일 식별자')
    p.add_argument('--data', default='/raid/ai/home/jsyang/dev_env/git/repos/ranger/data/custom_multihopqa_eval_1000.jsonl')

    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--n-chains', type=int, default=5)
    p.add_argument('--chain-depth', type=int, default=5)

    p.add_argument('--greedy', action='store_true', help='n_chains=1 greedy 디코딩으로 평가')
    p.add_argument('--temperature', type=float, default=-9, help='-9 이면 VLLM_CONFIG 기본값')
    p.add_argument('--top-p', type=float, default=-9)
    p.add_argument('--top-k', type=int, default=-9)

    '''
        아래 3개는 VLLM_CONFIG 기본값을 그대로 쓰되, 비교 대상 모델의 공개 설정에 맞출 수 있도록 노출
            - max_new_tokens : 생성 길이 상한. VllmEngine 생성자에서만 받고 호출 단위 오버라이드가 없으므로 여기서 지정
            - n_log_prob     : best-of-n 체인 선택에 쓰는 top-k logprob 개수 (최대 20)
                               샘플링된 토큰이 top-k 밖이면 그 토큰은 집계에서 빠지므로, 선택 품질에 영향을 줌
    '''
    p.add_argument('--max-seq-length', type=int, default=-1, help='-1 이면 VLLM_CONFIG 기본값 (8B 등 다른 모델은 조정 필요할 수 있음)')
    p.add_argument('--max-new-tokens', type=int, default=-1, help='-1 이면 VLLM_CONFIG 기본값')
    p.add_argument('--n-log-prob', type=int, default=-1, help='-1 이면 VLLM_CONFIG 기본값 (최대 20)')
    p.add_argument('--gpu-memory-utilization', type=float, default=-1)
    '''
        [토큰 비용 공정 비교] 조기 종료 능력이 없는 베이스라인(순정 LLM, CoRAG 등)에 사용
            매 스텝 final_answer 를 생성하면, 그 모델의 실제 추론 절차에 없는 계산까지 과금되어
            제안 방법이 부당하게 유리해짐 -> 마지막 스텝에서만 final_answer 를 생성하도록 제한
    '''
    p.add_argument('--final-answer-last-only', action='store_true',
                   help='마지막 스텝에서만 최종 답변 생성 (조기 종료가 없는 베이스라인 전용)')
    p.add_argument('--limit', type=int, default=-1, help='디버그용, 앞에서 N건만 평가')
    p.add_argument('--out', default='', help='결과 JSON 저장 경로 (선택)')
    return p.parse_args()


def build_vllm_config(args):
    vllm_config = dict(VLLM_CONFIG)
    vllm_config['model_name'] = args.model

    # CUDA_VISIBLE_DEVICES 로 GPU 를 지정하므로, 프로세스 안에서는 항상 0번
    vllm_config['device'] = 'cuda:0'

    if 0 < args.max_seq_length:
        vllm_config['max_seq_length'] = args.max_seq_length
    if 0 < args.max_new_tokens:
        vllm_config['max_new_tokens'] = args.max_new_tokens
    if 0 < args.n_log_prob:
        vllm_config['n_log_prob'] = args.n_log_prob
    if 0 < args.gpu_memory_utilization:
        vllm_config['gpu_memory_utilization'] = args.gpu_memory_utilization

    return vllm_config


def main():
    from ranger.chain_generate.chain_generator import ChainGenerator
    from ranger.corag.corag_result import QueryResult

    args = parse_args()
    common_utils.set_seed(COMMON_CONFIG['seed'])

    if args.greedy:
        n_chains, temperature, top_p, top_k = 1, 0.0, 1.0, 1
        decoding = 'greedy (n=1)'
    else:
        n_chains, temperature, top_p, top_k = args.n_chains, args.temperature, args.top_p, args.top_k
        decoding = f'best-of-{n_chains} (temperature={temperature}, top_p={top_p})'

    datas = json_utils.load_file(args.data)
    if 0 < args.limit:
        datas = datas[:args.limit]

    prefix = f'# [EVAL][{args.tag}]'
    print(f'\n{prefix} model : {args.model}')
    print(f'{prefix} adapter : [{args.adapter}]')
    print(f'{prefix} data : {args.data} ({len(datas)}건)')
    vllm_config = build_vllm_config(args)
    print(f'{prefix} decoding : {decoding}, chain_depth : {args.chain_depth}, batch_size : {args.batch_size}')
    print(f'{prefix} final_answer : {"마지막 스텝에서만 (조기종료 없는 베이스라인)" if args.final_answer_last_only else "매 스텝 (조기종료 판단용)"}')
    print(f'{prefix} max_seq_length : {vllm_config["max_seq_length"]}, max_new_tokens : {vllm_config["max_new_tokens"]}, n_log_prob : {vllm_config["n_log_prob"]}')
    print(f'{prefix} start : {common_utils.get_datetime_now()}\n')
    start = common_utils.get_time_ms()

    # 체인 생성기를 프로세스 안에서 직접 생성 (체인 서버 HTTP 미사용)
    chain_generator = ChainGenerator(vllm_config, CORAG_CONFIG)

    def generate_fn(datas_batch, batch_size, n_chains_, chain_depth_, adapter_path,
                    temperature_=-9, top_p_=-9, top_k_=-9, is_eval=False) -> List[QueryResult]:
        # ChainGenerator.generate() 는 배치별 리스트의 리스트를 반환하므로 평탄화
        results = chain_generator.generate(
            datas_batch, batch_size, n_chains_, chain_depth_, adapter_path,
            temperature_, top_p_, top_k_, is_eval,
            final_answer_last_only=args.final_answer_last_only
        )
        return [qr for batch in results for qr in batch]

    ems, f1s, rewards, advantages, sources, token_stats = evaluation_utils.evaluate(
        f'[{args.tag}]',
        datas,
        args.batch_size,
        n_chains,
        args.chain_depth,
        args.adapter,
        temperature,
        top_p,
        top_k,
        RewardCalculator(REWARD_CONFIG['reward_option']),
        do_print=True,
        generate_fn=generate_fn
    )

    source_scores = evaluation_utils.aggregate_by_source(sources, ems, f1s, **token_stats)
    evaluation_utils.print_source_scores(f'{prefix} [최종]', source_scores)

    if args.out:
        json_utils.write_file({
            'tag': args.tag,
            'model': args.model,
            'adapter': args.adapter,
            'data': args.data,
            'decoding': decoding,
            'n_chains': n_chains,
            'chain_depth': args.chain_depth,
            'max_seq_length': vllm_config['max_seq_length'],
            'max_new_tokens': vllm_config['max_new_tokens'],
            'n_log_prob': vllm_config['n_log_prob'],
            'final_answer_last_only': args.final_answer_last_only,
            'source_scores': source_scores
        }, args.out)
        print(f'{prefix} saved : {args.out}')

    _, elapsed = common_utils.get_elapsed_time_ms(start)
    print(f'{prefix} end : {common_utils.get_datetime_now()}, elapsed : {elapsed}\n')


if __name__ == '__main__':
    main()
