from _init import *

from transformers import AutoTokenizer, PreTrainedTokenizerFast
from typing import List, Dict


def load_tokenizer(model_name: str, padding_side='left') -> PreTrainedTokenizerFast:
    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.padding_side = padding_side

    if tokenizer.pad_token is None:
        # <|end_of_text|>(128001) 또는 <|eot_id|>(128009) -> '128009'가 중간에 나타나서 다른 걸로 변경
        if tokenizer.eos_token_id == 128009:
            tokenizer.pad_token = '<|end_of_text|>'
        else:
            tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


'''
    # 싱글턴
    datas = [
        [{'role': 'user', 'content': '안녕하세요'}],
        [{'role': 'user', 'content': '반갑습니다'}],
    ]

    # 멀티턴
    datas = [
        [{'role': 'system', 'content': '첫 번째 채팅입니다'}, {'role': 'user', 'content': '안녕하세요'}],
        [{'role': 'system', 'content': '두 번째 채팅입니다'}, {'role': 'user', 'content': '반갑습니다'}],
    ]
'''
def __apply_chat_template_and_truncate(datas: List[List[Dict]], tokenizer: PreTrainedTokenizerFast, max_length: int, add_generation_prompt=True):
    prompts = [tokenizer.apply_chat_template(data, tokenize=False, add_generation_prompt=add_generation_prompt) for data in datas]

    truncated_prompts = []
    for prompt in prompts:
        prompt_ids = tokenizer(prompt, add_special_tokens=False)['input_ids']

        if len(prompt_ids) > max_length:
            if prompt_ids[0] == tokenizer.bos_token_id:
                truncated_ids = [prompt_ids[0]] + prompt_ids[-(max_length - 1):]
            else:
                truncated_ids = prompt_ids[-max_length:]

            truncated_prompt = tokenizer.decode(truncated_ids, skip_special_tokens=False)
            truncated_prompts.append(truncated_prompt)
        else:
            truncated_prompts.append(prompt)

    return truncated_prompts


def tokenize_apply_chat_template_and_truncate(
        datas: List[List[Dict]],
        tokenizer: PreTrainedTokenizerFast,
        max_length: int,
        add_generation_prompt=True
    ) -> List[List[int]]:

    prompt_ids_list = [
        tokenizer.apply_chat_template(
            data,
            tokenize=True,
            add_generation_prompt=add_generation_prompt
        ) for data in datas
    ]

    truncated_prompt_ids_list = []
    bos_token_id = tokenizer.bos_token_id

    for prompt_ids in prompt_ids_list:
        if max_length < len(prompt_ids):
            print(f'# [LOG] tokenizer_utils.tokenize_apply_chat_template_and_truncate() length : {len(prompt_ids)}')

            if bos_token_id is not None and prompt_ids[0] == bos_token_id:
                truncated_prompt_ids = [bos_token_id + prompt_ids[-(max_length - 1):]]
            else:
                truncated_prompt_ids = prompt_ids[-max_length:]
            
            truncated_prompt_ids_list.append(truncated_prompt_ids)
        else:
            truncated_prompt_ids_list.append(prompt_ids)
    
    return truncated_prompt_ids_list

