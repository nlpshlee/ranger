from _init import *

from transformers import AutoTokenizer, PreTrainedTokenizerFast
from typing import List, Dict


def load_tokenizer(model_name: str, padding_side='left') -> PreTrainedTokenizerFast:
    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.padding_side = padding_side

    if tokenizer.pad_token is None:
        '''
            <|end_of_text|>(128001) 또는 <|eot_id|>(128009) -> '128009'가 중간에 나타나서 다른 걸로 변경

            또, 128009가 종료 토큰으로 되어 있는데, 이걸 패딩 토큰으로 사용하면
            학습에 반영되어야 하는 실제 마지막 종료 토큰까지 패딩으로 인식되기 때문에
            종료 토큰(128009) 대신에 다른 토큰(128001)을 패딩 토큰으로 사용한 것
        '''
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

    '''
        배치로 한 번에 처리되도록 변경
        - apply_chat_template() 반환 타입을 고려하여, 2단계로 분리해서 진행
            - apply_chat_template() -> tokenize()
            - 한 번에 하는 것과 속도 차이는 크게 없음
    '''
    datas_chat_template = tokenizer.apply_chat_template(
        datas,
        tokenize=False,
        add_generation_prompt=add_generation_prompt
    )

    prompt_ids_list = tokenizer(datas_chat_template, add_special_tokens=False)["input_ids"]

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

