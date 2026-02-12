from _init import *

from typing import List
import numpy as np

from ranger.utils import common_utils, container_utils
from ranger.corag.corag_result import QueryResult
from ranger.chain_generate.chain_generate_client import request_chain_generate
from ranger.reward.reward_calculator import RewardCalculator


def evaluate(prefix, datas, batch_size, n_chains, chain_depth, adapter_path='',
             temperature=-9, top_p=-9, top_k=-9,
             reward_calculator: RewardCalculator=None,
             do_print=True):

    prefix = f'# evaluation_utils.evaluate() {prefix}'.strip()
    data_size = len(datas)

    if DEBUG.EVAL and do_print:
        print(f'{prefix} data_size : {data_size}, batch_size : {batch_size}, n_chains : {n_chains}, chain_depth : {chain_depth}, adapter_path : [{adapter_path}]')
        print(f'{prefix} temperature : {temperature}, top_p : {top_p}, top_k : {top_k}')
        print(f'{prefix} start : {common_utils.get_datetime_now()}')
        eval_start = common_utils.get_time_ms()
    
    ems, f1s = [], []
    rewards, advantages = [], []

    for batch_idx, datas_batch in enumerate(container_utils.chunks(datas, batch_size)):
        if DEBUG.EVAL and do_print:
            print(f'{prefix} {batch_idx+1} batch start\t: {common_utils.get_datetime_now()}')
            batch_start = common_utils.get_time_ms()
        
        query_results: List[QueryResult] = request_chain_generate(
            datas_batch,
            batch_size,
            n_chains,
            chain_depth,
            adapter_path,
            temperature,
            top_p,
            top_k,
            is_eval=True
        )

        if reward_calculator is None:
            for query_result in query_results:
                query_result.compute_metrics()
        else:
            reward_calculator.calculate_reward_and_advantage(query_results)

        for query_result in query_results:
            '''
                query_result.compute_metrics() 는 모든 체인을 정답과 비교하여 가장 높은 점수를 query 의 점수로 사용

                greedy(n_chains==1)는 어차피 chain 의 수가 '1'이기 때문에, 상관 없음
                
                단, best-of-n 정답을 보고 가장 높은 체인을 선택하면 안됨
                정답 없이 최선의 체인을 선택하고, 그 체인의 점수를 query 의 점수로 사용해야 함
                CoRAG에서는 모델의 생성 확률을 보고 체인을 선택함

                greedy, best-of-n 둘다 생성 확률이 가장 높은 체인의 스코어를 사용하면 됨
            '''
            log_probs = [chain_result._log_probs[-1] for chain_result in query_result._chain_results]
            max_idx = np.argmax(log_probs)

            ems.append(query_result._chain_results[max_idx]._em)
            f1s.append(query_result._chain_results[max_idx]._f1)
            rewards.append(query_result._chain_results[max_idx]._reward)
            advantages.append(query_result._chain_results[max_idx]._advantage)
        
        if DEBUG.EVAL and do_print:
            _, batch_elapsed_str = common_utils.get_elapsed_time_ms(batch_start)
            print(f'{prefix} {batch_idx+1} batch end\t: {common_utils.get_datetime_now()}, elapsed : {batch_elapsed_str}')
    if DEBUG.EVAL and do_print:
        _, eval_elapsed_str = common_utils.get_elapsed_time_ms(eval_start)
        print(f'{prefix} end : {common_utils.get_datetime_now()}, elapsed : {eval_elapsed_str}')

    return ems, f1s, rewards, advantages

