from _init import *

from transformers import AutoTokenizer, PreTrainedTokenizerFast
from typing import List, Dict, Union


def load_tokenizer(model_name: str, padding_side='left') -> PreTrainedTokenizerFast:
    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.padding_side = padding_side

    if tokenizer.pad_token is None:
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
def apply_chat_template_and_truncate(datas: List[List[Dict]], tokenizer: PreTrainedTokenizerFast, max_length: int, add_generation_prompt=True):
    prompts = [tokenizer.apply_chat_template(data, tokenize=False, add_generation_prompt=add_generation_prompt) for data in datas]
    print(f'prompts : \n\n{prompts}')

    truncated_prompts = []
    for prompt in prompts:
        tokenized_ids = tokenizer(prompt, add_special_tokens=False)['input_ids']

        if len(tokenized_ids) > max_length:
            if tokenized_ids[0] == tokenizer.bos_token_id:
                truncated_ids = [tokenized_ids[0]] + tokenized_ids[-(max_length - 1):]
            else:
                truncated_ids = tokenized_ids[-max_length:]

            truncated_prompt = tokenizer.decode(truncated_ids, skip_special_tokens=False)
            truncated_prompts.append(truncated_prompt)
        else:
            truncated_prompts.append(prompt)

    print(f'\n\ntruncated_prompts : \n\n{truncated_prompts}')
    return truncated_prompts

