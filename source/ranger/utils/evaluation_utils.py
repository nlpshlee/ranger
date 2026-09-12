from _init import *

from typing import List
import numpy as np

import torch
from transformers import PreTrainedTokenizerFast, AutoModelForCausalLM

from ranger.utils import common_utils, container_utils, tokenizer_utils
from ranger.corag.corag_result import QueryResult, _compute_f1
from ranger.chain_generate.chain_generate_client import request_chain_generate
from ranger.reward.reward_calculator import RewardCalculator
from ranger.train.sft_dataset import SftDataset
from ranger.train.sft_trainer import IGNORE_INDEX


SOURCE_UNKNOWN = 'unknown'


def aggregate_by_source(sources, ems, f1s, prompt_tokens=None, gen_tokens=None, sel_tokens=None, depths=None):
    '''
        source(hotpotqa / musique / 2wikimultihopqa / bamboogle) 별로 EM/F1 집계

        평가 데이터가 4개 벤치마크를 섞은 혼합셋이라, 단일 숫자는 공개 논문 수치와 직접 비교가 불가능함
        (혼합 비율에 따라 값이 달라지므로 반드시 벤치마크 단위로 쪼개서 비교해야 함)
    '''
    n_query = len(ems)
    prompt_tokens = prompt_tokens or [0]*n_query
    gen_tokens = gen_tokens or [0]*n_query
    sel_tokens = sel_tokens or [0]*n_query
    depths = depths or [0]*n_query

    keys = ('em', 'f1', 'prompt_tokens', 'gen_tokens', 'sel_tokens', 'depth')
    agg = {}

    for source, em, f1, pt, gt, st, dp in zip(sources, ems, f1s, prompt_tokens, gen_tokens, sel_tokens, depths):
        source = source or SOURCE_UNKNOWN
        row = agg.setdefault(source, {'n': 0, **{f'{k}_sum': 0.0 for k in keys}})
        row['n'] += 1
        for k, v in zip(keys, (em, f1, pt, gt, st, dp)):
            row[f'{k}_sum'] += v

    def _finalize(row):
        out = {'n': row['n']}
        for k in keys:
            out[k] = row[f'{k}_sum'] / row['n']
        # 쿼리 1건을 처리하는 데 실제로 든 총 토큰 (prefill + decode, 모든 체인 합)
        out['total_tokens'] = out['prompt_tokens'] + out['gen_tokens']
        out[f'{"sum"}_total_tokens'] = row['prompt_tokens_sum'] + row['gen_tokens_sum']
        return out

    results = {source: _finalize(row) for source, row in agg.items()}

    # 전체(혼합셋) 값도 같은 형식으로 함께 제공
    if n_query:
        all_row = {'n': n_query}
        for k, vals in zip(keys, (ems, f1s, prompt_tokens, gen_tokens, sel_tokens, depths)):
            all_row[f'{k}_sum'] = float(sum(vals))
        results['ALL'] = _finalize(all_row)

    return results


def print_source_scores(prefix, source_scores: dict):
    order = [k for k in ['hotpotqa', 'musique', '2wikimultihopqa', 'bamboogle'] if k in source_scores]
    order += [k for k in sorted(source_scores) if k not in order and k != 'ALL']
    if 'ALL' in source_scores:
        order.append('ALL')

    print(f'\n{prefix} source 별 성능 / 토큰 소비량')
    print(f'{"source":<18}{"N":>6}{"EM":>9}{"F1":>9}{"depth":>8}'
          f'{"prompt/q":>11}{"gen/q":>9}{"total/q":>10}{"sel(total)/q":>14}{"total(sum)":>14}')
    print('-' * 108)
    for source in order:
        r = source_scores[source]
        print(f'{source:<18}{r["n"]:>6}{r["em"]*100:>8.2f}%{r["f1"]*100:>8.2f}%{r["depth"]:>8.2f}'
              f'{r["prompt_tokens"]:>11,.0f}{r["gen_tokens"]:>9,.1f}{r["total_tokens"]:>10,.0f}'
              f'{r["sel_tokens"]:>14,.0f}{r["sum_total_tokens"]:>14,.0f}')
    print(f'''
    depth         : 선택된 체인의 평균 스텝 수 (조기 종료 여부를 보여줌)
    prompt/q      : 쿼리 1건당 프롬프트(prefill) 토큰 - 검색 문서 포함, 실제 비용의 대부분
    gen/q         : 쿼리 1건당 생성(decode) 토큰
    total/q       : prompt + gen, n_chains 개 체인을 '모두' 생성한 실제 비용
    sel(total)/q  : 그 중 최종 선택된 체인 1개가 쓴 토큰 (greedy 면 total/q 와 동일)
    total(sum)    : 해당 source 전체 합계''')
    print()


def evaluate(prefix, datas, batch_size, n_chains, chain_depth, adapter_path='',
             temperature=-9, top_p=-9, top_k=-9,
             reward_calculator: RewardCalculator=None,
             do_print=True,
             generate_fn=None):
    '''
        generate_fn : 체인 생성 함수를 교체하기 위한 훅 (기본값은 체인 서버 HTTP 호출)
            - 학습 중에도 다른 GPU 에서 별도 모델을 평가할 수 있도록,
              체인 서버를 거치지 않는 in-process 생성기를 주입할 수 있게 함
    '''

    prefix = f'# evaluation_utils.evaluate() {prefix}'.strip()
    data_size = len(datas)

    if DEBUG.EVAL and do_print:
        print(f'{prefix} data_size : {data_size}, batch_size : {batch_size}, n_chains : {n_chains}, chain_depth : {chain_depth}, adapter_path : [{adapter_path}]')
        print(f'{prefix} temperature : {temperature}, top_p : {top_p}, top_k : {top_k}')
        print(f'{prefix} start : {common_utils.get_datetime_now()}')
        eval_start = common_utils.get_time_ms()
    
    ems, f1s = [], []
    rewards, advantages = [], []
    sources = []
    prompt_tokens, gen_tokens, sel_tokens = [], [], []
    depths = []

    # query_id -> source 매핑 (순서에 의존하지 않도록)
    query_id_to_source = {data['query_id']: data.get('source', SOURCE_UNKNOWN) for data in datas if 'query_id' in data}

    if generate_fn is None:
        generate_fn = request_chain_generate

    for batch_idx, datas_batch in enumerate(container_utils.chunks(datas, batch_size)):
        if DEBUG.EVAL and do_print:
            print(f'{prefix} {batch_idx+1} batch start\t: {common_utils.get_datetime_now()}')
            batch_start = common_utils.get_time_ms()
        
        query_results: List[QueryResult] = generate_fn(
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
            sources.append(query_id_to_source.get(query_result._query_id, SOURCE_UNKNOWN))

            '''
                토큰 소비량은 '실제로 생성한 모든 체인'을 합산해야 함
                best-of-n 은 n 개를 다 만들어야 1개를 고를 수 있으므로,
                선택된 체인만 세면 실제 추론 비용을 n 배 과소평가하게 됨
            '''
            prompt_tokens.append(sum(cr._prompt_tokens for cr in query_result._chain_results))
            gen_tokens.append(sum(cr._gen_tokens for cr in query_result._chain_results))
            sel_tokens.append(query_result._chain_results[max_idx]._prompt_tokens
                              + query_result._chain_results[max_idx]._gen_tokens)

            # 실제로 몇 스텝을 밟았는지 (final_answer 생성 시점 설정과 무관하게 sub_query 개수로 측정)
            depths.append(len(query_result._chain_results[max_idx]._sub_querys))
        
        if DEBUG.EVAL and do_print:
            _, batch_elapsed_str = common_utils.get_elapsed_time_ms(batch_start)
            print(f'{prefix} {batch_idx+1} batch end\t: {common_utils.get_datetime_now()}, elapsed : {batch_elapsed_str}')
    if DEBUG.EVAL and do_print:
        _, eval_elapsed_str = common_utils.get_elapsed_time_ms(eval_start)
        print(f'{prefix} end : {common_utils.get_datetime_now()}, elapsed : {eval_elapsed_str}')

    token_stats = {'prompt_tokens': prompt_tokens, 'gen_tokens': gen_tokens, 'sel_tokens': sel_tokens, 'depths': depths}

    if DEBUG.EVAL and do_print:
        print_source_scores(prefix, aggregate_by_source(sources, ems, f1s, **token_stats))

    return ems, f1s, rewards, advantages, sources, token_stats


def evaluate_sft(model_name_or_path, dtype,
                 eval_datas, max_seq_length, max_new_tokens, ignore_index=IGNORE_INDEX,
                 debug_cnt=-1):

    total_cnt = len(eval_datas)

    if DEBUG.EVAL:
        prefix = '# evaluation_utils.evaluate_sft()'
        print(f'{prefix} model_name(or checkpoint_path) : {model_name_or_path}')
        print(f'{prefix} eval_datas size : {total_cnt}')
        print(f'{prefix} start : {common_utils.get_datetime_now()}\n')
        eval_start = common_utils.get_time_ms()

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=getattr(torch, dtype),
        device_map='auto',
        trust_remote_code=False,
        attn_implementation='flash_attention_2'
    )
    model.eval()

    # 평가/추론 시에는 반드시 'left' 패딩
    tokenizer: PreTrainedTokenizerFast = tokenizer_utils.load_tokenizer(model_name_or_path, 'left')

    # SftDataset 클래스로 변환해줘야 함
    eval_dataset = SftDataset(eval_datas, tokenizer, max_seq_length, ignore_index)

    em_cnt = 0
    start_cnt = 0
    diff_cnt = 0
    f1_scores = []

    with torch.no_grad():
        for i, eval_data in enumerate(eval_dataset):
            input_ids = eval_data['input_ids']
            labels = eval_data['labels']

            source_len = sum(1 for x in labels if x == ignore_index)
            source_ids = input_ids[:source_len]
            target_ids = [x for x in labels if x != ignore_index]

            source_tensor = torch.tensor([source_ids]).to(model.device)
            attention_mask_tensor = torch.tensor([[1] * source_len]).to(model.device)

            outputs = model.generate(
                input_ids=source_tensor,
                attention_mask=attention_mask_tensor,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False, # Greedy 디코딩 (일관된 평가를 위해)
                temperature=None,
                top_p=None
            )

            generated_ids = outputs[0][source_len:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            target_text = tokenizer.decode(target_ids, skip_special_tokens=True).strip()

            generated_text = generated_text.lower()
            target_text = target_text.lower()

            if generated_text == target_text:
                em_cnt += 1
            elif generated_text.startswith(target_text):
                start_cnt += 1
            else:
                diff_cnt += 1

                if DEBUG.EVAL and diff_cnt <= debug_cnt:
                    print(f'{prefix} gen_text : {generated_text}')
                    print(f'{prefix} tgt_text : {target_text}\n')

            f1_score = _compute_f1(target_text, generated_text)
            f1_scores.append(f1_score)

            if DEBUG.EVAL:
                if (i+1) % int((total_cnt / 10)) == 0:
                    print(f'{prefix} {i+1} evaluate complet.')
        print(f'{prefix} {total_cnt} evaluate complet.')

    em_accuracy = (em_cnt / total_cnt) * 100
    print(f'\n{model_name_or_path} EM : {em_accuracy:.2f}% ({em_cnt}/{total_cnt})')

    start_accuracy = (start_cnt / total_cnt) * 100
    print(f'{model_name_or_path} STARTSWITH : {start_accuracy:.2f}% ({start_cnt}/{total_cnt})')

    full_cnt = em_cnt + start_cnt
    full_accuracy = (full_cnt / total_cnt) * 100
    print(f'{model_name_or_path} FULL : {full_accuracy:.2f}% ({full_cnt}/{total_cnt})')

    f1_avg = np.mean(f1_scores)
    print(f'{model_name_or_path} F1 : {(f1_avg*100):.2f}%\n')

    if DEBUG.EVAL:
        _, eval_elapsed_str = common_utils.get_elapsed_time_ms(eval_start)
        print(f'{prefix} end : {common_utils.get_datetime_now()}, elapsed : {eval_elapsed_str}\n')

    del model
    del tokenizer
    common_utils.clear_gpu_memory()

    return em_accuracy, start_accuracy, full_accuracy, f1_avg

