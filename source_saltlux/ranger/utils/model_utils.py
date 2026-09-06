from _init import *

import torch, re
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
from peft import PeftModel

from ranger.utils import tokenizer_utils


LOG_PREFIX = '# [LOG] model_utils'
ERR_PREFIX = '# [ERROR] model_utils'
ATTN_IMP = 'flash_attention_2'


def get_model_name_or_path(model_name: str):
    if model_name.startswith(f'Llama'):
        model_name_or_path = f'meta-llama/{model_name}-Instruct'
    elif model_name.startswith('Qwen'):
        model_name_or_path = f'Qwen/{model_name}-Instruct'
    else:
        model_name_or_path = model_name

    return model_name_or_path


def get_model(model_name_or_path, dtype, device=None, device_map=None, attn_imp=None, is_eval=False):
    if (device is not None) and (device_map is not None):
        if DEBUG.ERROR:
            print(f'\n{ERR_PREFIX}.get_model() : device or device_map is not assigned.\n')
        return None

    if attn_imp is None:
        attn_imp = ATTN_IMP

    if device is not None:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=getattr(torch, dtype),
            trust_remote_code=False,
            attn_implementation=attn_imp
        )
        model = model.to(device)

    elif device_map is not None:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=getattr(torch, dtype),
            device_map=device_map, # accelerator 사용할 때만 ('auto')
            trust_remote_code=False,
            attn_implementation=attn_imp
        )

    if is_eval:
        model.eval()
    
    if DEBUG.LOG:
        print(f'\n{LOG_PREFIX}.get_model() : {model_name_or_path}\n')

    return model


def merge_and_save(model_name, dtype, adapter_path, save_path, device_map='auto', attn_imp=None):
    if attn_imp is None:
        attn_imp = ATTN_IMP

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=getattr(torch, dtype),
        device_map=device_map,
        trust_remote_code=False,
        attn_implementation=attn_imp
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


def make_inputs(tokenizer: PreTrainedTokenizerFast, device: str, prompts, max_seq_length,
                return_offsets_mapping=False, return_all=False):

    chat_prompts = tokenizer.apply_chat_template(
        prompts,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        chat_prompts,
        padding=True,
        truncation=True,
        max_length=max_seq_length,
        return_tensors='pt',
        return_offsets_mapping=return_offsets_mapping,
        add_special_tokens=False
    )

    if device:
        inputs = inputs.to(device)

    if return_all:
        return chat_prompts, inputs
    else:
        return inputs


def forward(model: AutoModelForCausalLM, tokenizer: PreTrainedTokenizerFast, device: str,
            prompts, max_seq_length, output_hidden_states=True):

    inputs = make_inputs(tokenizer, device, prompts, max_seq_length)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=output_hidden_states)
    
    return outputs


def generate(model: AutoModelForCausalLM, tokenizer: PreTrainedTokenizerFast, device: str,
             prompts: list, max_seq_length: int, max_new_tokens: int,
             do_sample=False, temperature=None, top_k=None, top_p=None,
             return_all=False):
    
    inputs = make_inputs(tokenizer, device, prompts, max_seq_length)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
        # outputs 은 .to(device) 할 필요 없음, input_ids 와 같은 device 에 자동 할당됨
    
    if return_all:
        return input_ids, outputs
    else:
        return outputs


def get_generated_texts(model: AutoModelForCausalLM, tokenizer: PreTrainedTokenizerFast, device: str, 
                        prompts: list, max_seq_length: int, max_new_tokens: int, 
                        do_sample=False, temperature=None, top_k=None, top_p=None):

    input_ids, outputs = generate(
        model, tokenizer, device,
        prompts, max_seq_length, max_new_tokens,
        do_sample, temperature, top_k, top_p,
        return_all=True
    )

    generated_texts = []
    for i, output in enumerate(outputs):
        prompt_length = input_ids[i].shape[0]
        generated_token_ids = output[prompt_length:]
        generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
        generated_texts.append(generated_text.strip())
    
    return generated_texts


def is_correct(generated_text: str, answer: str):
    generated_text = generated_text.lower().strip(' ,.!?~')
    answer = answer.lower().strip(' ,.!?~')

    is_exact = True if generated_text == answer else False

    if is_exact:
        return [True, True]

    check_pat =  re.compile(rf'\b{re.escape(answer)}\b')
    is_contains = bool(check_pat.search(generated_text))

    if is_contains:
        return [False, True]

    check_pat =  re.compile(rf'\b{re.escape(generated_text)}\b')
    is_contains = bool(check_pat.search(answer))

    return [False, is_contains]

