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

    em_accuracy = (em_cnt / total_cnt) * 100
    print(f'\n{model_name_or_path} EM : {em_accuracy:.2f}% ({em_cnt}/{total_cnt})')

    start_accuracy = (start_cnt / total_cnt) * 100
    print(f'{model_name_or_path} STARTSWITH : {start_accuracy:.2f}% ({start_cnt}/{total_cnt})')

    full_cnt = em_cnt + start_cnt
    full_accuracy = (full_cnt / total_cnt) * 100
    print(f'{model_name_or_path} FULL : {full_accuracy:.2f}% ({full_cnt}/{total_cnt})')

    f1_avg = np.mean(f1_scores)
    print(f'{model_name_or_path} F1 : {f1_avg:.2f}%\n')

    if DEBUG.EVAL:
        _, eval_elapsed_str = common_utils.get_elapsed_time_ms(eval_start)
        print(f'{prefix} end : {common_utils.get_datetime_now()}, elapsed : {eval_elapsed_str}\n')

    del model
    del tokenizer
    common_utils.clear_gpu_memory()

    return em_accuracy, start_accuracy, full_accuracy, f1_avg

