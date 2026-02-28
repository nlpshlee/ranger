from _init import *

import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
from peft import PeftModel

from ranger.utils import tokenizer_utils


LOG_PREFIX = '# [LOG] model_utils'


def merge_and_save(model_name, dtype, adapter_path, save_path):
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=getattr(torch, dtype),
        device_map='auto',
        trust_remote_code=False,
        attn_implementation='flash_attention_2'
    )

    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload()
    model.save_pretrained(save_path, safe_serialization=True)

    # 패딩 방향(left/right)은 상관 없음
    tokenizer: PreTrainedTokenizerFast = tokenizer_utils.load_tokenizer(adapter_path)
    tokenizer.save_pretrained(save_path)

    if DEBUG.LOG:
        print(f'\n{LOG_PREFIX}.merge_and_save() model load : {model_name}')
        print(f'{LOG_PREFIX}.merge_and_save() adapter merge : {adapter_path}')
        print(f'{LOG_PREFIX}.merge_and_save() merged model save : {save_path}\n')

