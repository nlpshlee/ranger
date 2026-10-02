from _init import *

'''
    SFT 학습용 체인 생성을 여러 체인 서버(GPU)로 나눠서 병렬 실행

        chain_generate_for_sft.ipynb 의 생성 로직(chain_generate_for_sft)과 동일한 결과를 만들고,
        실행 방식만 아래처럼 바꿈
            - 전체 데이터를 샤드로 나눠, 샤드마다 서로 다른 체인 서버(포트)로 요청
            - 이미 온전하게 생성된 쿼리는 건너뜀 (중간에 끊겨도 이어서 실행 가능)
            - 결과 파일은 임시 폴더에 쓴 뒤 옮김 (반쯤 쓰인 파일이 결과 폴더에 남지 않음)
            - 배치마다 결과를 검사해서, 잘못된 모드의 서버에 붙으면 즉시 중단

        샤드 i 는 전체 데이터의 i, i+N, i+2N ... 번째 쿼리를 맡음
        -> 샤드끼리 겹치지 않고, 합치면 전체와 같음 (실행 시점과 무관하게 항상 같은 분할)

    실행 예) 3개 GPU, 서버 포트 7200/7201/7202
        python -u chain_generate_for_sft_shard.py --shard 0 --num-shards 3 --port 7200
        python -u chain_generate_for_sft_shard.py --shard 1 --num-shards 3 --port 7201
        python -u chain_generate_for_sft_shard.py --shard 2 --num-shards 3 --port 7202
        (전부 끝난 뒤 한 번)
        python -u chain_generate_for_sft_shard.py --merge
'''

import argparse, json, shutil
from typing import List

from ranger.utils import common_utils, json_utils, file_utils, container_utils
from ranger.corag.corag_result import QueryResult
from ranger.chain_generate.chain_generate_client import request_chain_generate
from ranger.reward.reward_calculator import RewardCalculator


WORK_DIR = '/raid/ai/home/jsyang/dev_env/git/repos/ranger'
TRAIN_DATA_PATH = f'{WORK_DIR}/data/custom_musique_train_5000_final.jsonl'
DECIDE = 'IMPORTANT: You must start your response with a special token'


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--num-shards', type=int, default=1)
    ap.add_argument('--port', type=int, default=CHAIN_SERVER_CONFIG['port'], help='이 샤드가 요청을 보낼 체인 서버 포트')
    ap.add_argument('--n-chains', type=int, default=32)          # 노트북과 동일
    ap.add_argument('--chain-depth', type=int, default=6)        # 노트북과 동일
    ap.add_argument('--batch-size', type=int, default=100)       # 노트북과 동일
    ap.add_argument('--out-root', default=f'{WORK_DIR}/data/sft')
    ap.add_argument('--limit', type=int, default=-1, help='디버그용 : 이 샤드에서 앞 N 개만')
    ap.add_argument('--merge', action='store_true', help='모든 샤드가 끝난 뒤 한 번 : 결과를 _merged.jsonl 로 합침')
    return ap.parse_args()


def is_done(path, n_chains):
    # 온전하게 생성된 파일인지 (없거나, 깨졌거나, 체인 수가 다르면 다시 생성)
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return len(data['query_result']['chain_results']) == n_chains
    except Exception:
        return False


def validate(query_results: List[QueryResult], datas_batch):
    '''
        SFT 데이터 생성 모드로 만들어진 결과인지 배치마다 확인
            - 잘못된 서버(구버전 코드 / 다른 모드)에 붙어서 생성되면, 오염된 데이터가 조용히 쌓이는 것을 막기 위함
            - 최종 답변 프롬프트에 <STOP> 지시문이 없어야 하고, 종료된 체인은 모두 정답이어야 함 (오라클 종료)
    '''
    if query_results is None:
        raise RuntimeError('체인 서버 요청 실패 (응답 None) : 서버 상태 / 포트를 확인하세요')
    if len(query_results) != len(datas_batch):
        raise RuntimeError(f'응답 쿼리 수 불일치 : 요청 {len(datas_batch)} / 응답 {len(query_results)}')

    for query_result in query_results:
        for chain_result in query_result._chain_results:
            if any(DECIDE in prompt[0]['content'] for prompt in chain_result._final_answer_prompts):
                raise RuntimeError(f'[{query_result._query_id}] 최종 답변 프롬프트에 <STOP> 지시문이 있음 -> SFT 데이터 생성 모드가 아닌 서버')
            if chain_result._is_stop and chain_result._final_answers[-1] not in query_result._answer_set:
                raise RuntimeError(f'[{query_result._query_id}] 오답인데 종료된 체인 -> 오라클 종료가 아닌 서버')


def write_atomic(data, out_path, tmp_dir):
    # 임시 폴더(결과 폴더 밖)에 쓴 뒤 옮김 -> 결과 폴더에는 항상 완성된 파일만 존재
    tmp_path = f'{tmp_dir}/{os.path.basename(out_path)}'
    with open(tmp_path, 'w', encoding='utf-8') as out_file:
        json.dump(data, out_file, ensure_ascii=False, indent=4)
    os.replace(tmp_path, out_path)


def chain_generate_for_sft(datas, batch_size, n_chains, chain_depth, out_dir, tmp_dir, reward_calculator, prefix):
    # chain_generate_for_sft.ipynb 의 chain_generate_for_sft() 와 동일한 생성/저장 로직 (+ 검사, 원자적 저장)
    done_cnt = 0

    for datas_batch in container_utils.chunks(datas, batch_size):
        batch_start = common_utils.get_time_ms()

        query_results: List[QueryResult] = request_chain_generate(datas_batch, batch_size, n_chains, chain_depth, sft_data_mode=True)
        validate(query_results, datas_batch)

        reward_calculator.calculate_reward_and_advantage(query_results)

        for data, query_result in zip(datas_batch, query_results):
            query_id = data['query_id']

            if query_id == query_result._query_id:
                data['query_result'] = query_result.to_dict()
                write_atomic(data, f'{out_dir}/{query_id}.json', tmp_dir)
            else:
                print(f'# [error] chain_generate_for_sft() query_id is diff : {query_id} - {query_result._query_id}')

        done_cnt += len(datas_batch)
        _, elapsed = common_utils.get_elapsed_time_ms(batch_start)
        print(f'{prefix} {done_cnt}/{len(datas)} 완료, 배치 소요 {elapsed}, {common_utils.get_datetime_now()}', flush=True)


def merge_to_jsonl(in_dir, out_file_path):
    # chain_generate_for_sft.ipynb 의 merge_to_jsonl() 과 동일
    in_file_paths = file_utils.get_file_paths(in_dir)
    print(f'{in_dir} : {len(in_file_paths)} files merge')

    file_utils.make_parent(out_file_path)
    with open(out_file_path, 'w', encoding='utf-8') as out_file:
        for in_file_path in in_file_paths:
            with open(in_file_path, 'r', encoding='utf-8') as in_file:
                data = json.load(in_file)
                json_line = json.dumps(data, ensure_ascii=False)
                out_file.write(json_line + '\n')


def main():
    args = parse_args()
    common_utils.set_seed(42)

    out_path = f'{args.out_root}/train_5000_n_chains-{args.n_chains}_chain_depth-{args.chain_depth}'
    train_datas = json_utils.load_file(TRAIN_DATA_PATH)     # 노트북과 동일하게 셔플하지 않음

    if args.merge:
        missing = [d['query_id'] for d in train_datas if not is_done(f'{out_path}/{d["query_id"]}.json', args.n_chains)]
        if missing:
            print(f'# [merge] 아직 생성되지 않은 쿼리 {len(missing)}개 (예: {missing[:3]}) -> 모든 샤드가 끝난 뒤 다시 실행하세요')
            return
        merge_to_jsonl(out_path, f'{out_path}_merged.jsonl')
        return

    assert 0 <= args.shard < args.num_shards, f'--shard 는 0 ~ {args.num_shards - 1}'
    CHAIN_SERVER_CONFIG['url_full_generate'] = f'http://{CHAIN_SERVER_CONFIG["host"]}:{args.port}{CHAIN_SERVER_CONFIG["url_generate"]}'

    mine = train_datas[args.shard::args.num_shards]
    if 0 < args.limit:
        mine = mine[:args.limit]
    pending = [d for d in mine if not is_done(f'{out_path}/{d["query_id"]}.json', args.n_chains)]

    tmp_dir = f'{out_path}.tmp_shard{args.shard}'
    os.makedirs(out_path, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)

    prefix = f'# [shard {args.shard}/{args.num_shards}, port {args.port}]'
    print(f'{prefix} 담당 {len(mine)}개 / 이미 완료 {len(mine) - len(pending)}개 / 남은 {len(pending)}개')
    print(f'{prefix} n_chains {args.n_chains}, chain_depth {args.chain_depth}, batch_size {args.batch_size}, out {out_path}', flush=True)

    chain_generate_for_sft(pending, args.batch_size, args.n_chains, args.chain_depth, out_path, tmp_dir,
                           RewardCalculator(REWARD_CONFIG['reward_option']), prefix)

    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f'{prefix} 종료 : {common_utils.get_datetime_now()}', flush=True)


if __name__ == '__main__':
    main()
