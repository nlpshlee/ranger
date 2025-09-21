from _init import *

import math
from typing import List, Dict, Union, Tuple, Any
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput, CompletionOutput
from vllm.sequence import Logprob
from vllm.lora.request import LoRARequest


class VllmEngine:
    def __init__(self, model_name: str, device: str, gpu_memory_utilization: float, dtype: str,
                 max_model_len: int, n_logprob: int):

        self._model_name = model_name
        self._device = device
        self._gpu_memory_utilization = gpu_memory_utilization
        self._dtype = dtype
        self._max_model_len = max_model_len
        self._n_logprob = n_logprob

        self._called_cnt = 0
        self._called_cnt_all = 0

        '''
            - max_loras
                - 어댑터의 '종류' 또는 '개수'
                    - 값이 '5'라면, 5개의 서로 다른 LoRA 어댑터를 GPU 메모리에 올려둘 공간을 미리 확보
                    - 동시에 처리되는 여러 요청들이 서로 다른 종류의 LoRA 어댑터를 요구할 때, 사용됨
                    - A 요청은 '요약 LoRA', B 요청은 '번역 LoRA' ...
        '''
        self._llm = LLM(
            model=self._model_name,
            device=self._device,
            gpu_memory_utilization=self._gpu_memory_utilization,
            dtype=self._dtype,
            max_model_len=self._max_model_len,                      # 없으면, 에러남
            enable_prefix_caching=True,                             # 프리픽스 캐싱 활성화(성능 향상)
            enable_lora=True
            # max_loras=1
        )
        self._tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(self._model_name)
        self._tok_id_end = self._tokenizer.eos_token_id
    

    def reset(self):
        self._called_cnt = 0


    def _get_generated_text(self, completion_output: CompletionOutput) -> str:
        return completion_output.text.strip()
    

    def get_generated_log_like(self, completion_output: CompletionOutput, text='', tok_ids=[], missing_log_prob=math.log(1e-9), return_all=False):
        # 'text'가 주어지면, 'text'를 토크나이징하고 likelihood 계산
        if len(text) > 0:
            tok_ids: list = self._tokenizer(text, add_special_tokens=False)['input_ids']
        # 'text'도 주어지지 않고, 'tok_ids'도 주어지지 않으면, 'completion_output'에 내장되어 있는 'tok_ids' 사용
        elif len(tok_ids) == 0:
            tok_ids: list = completion_output.token_ids
        
        if tok_ids[-1] != self._tok_id_end:
            tok_ids.append(self._tok_id_end)

        log_probs = []
        completion_output_log_prob_len = len(completion_output.logprobs)

        for tok_idx in range(len(tok_ids)):
            tok_id = tok_ids[tok_idx]

            # 모델이 생성한 텍스트보다 정답이 더 긴 경우
            if completion_output_log_prob_len <= tok_idx:
                log_probs.append(missing_log_prob)
                continue

            # 'tok_idx'번째 위치에서의 전체 토큰 확률 분포 (실제로는 top-k)
            all_tok_log_prob = completion_output.logprobs[tok_idx]

            if tok_id in all_tok_log_prob.keys():
                log_prob: Logprob = all_tok_log_prob[tok_id]
                log_prob = log_prob.logprob
            else:
                log_prob = missing_log_prob
            
            log_probs.append(log_prob)
        
        log_like = sum(log_probs) / len(log_probs)

        if return_all:
            toks = [self._tokenizer.decode(tok_id) for tok_id in tok_ids]
            return log_like, toks, tok_ids, log_probs
        else:
            return log_like


    def _make_generate_result(self, completion_output: CompletionOutput, return_completion_output: bool) -> Union[str, Tuple[str, Any]]:
        generated_text = self._get_generated_text(completion_output)

        if not return_completion_output:
            return generated_text
        else:
            return generated_text, completion_output


    def generate_batch(self, messages: List[List[Dict]], max_token_gen: int, temperature: int,
                       return_completion_output=False, adapter_path='', do_print=True) -> List[Union[str, Tuple[str, Any]]]:

        # LoRA Adapter 추가 코드
        if os.path.exists(adapter_path):
            lora_request = LoRARequest('RANGER_Policy', 1, adapter_path)
        else:
            lora_request = None
        
        sampling_params = SamplingParams(
            max_tokens=max_token_gen,
            temperature=temperature,
            logprobs=self._n_logprob if return_completion_output else None
        )

        # LoRA Adapter 추가 코드
        prompts = [self._tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]

        # LoRA Adapter 추가 코드
        '''
            chat() -> generate()
                messages -> prompts
                (+) lora_request
            
            chat() 함수에 lora_request 전달 불가능
                - vllm.LLM.chat() 메소드는 lora_request 파라미터를 지원하지 않도록 설계됨
                - LoRA를 동적으로 적용하려면 반드시 llm.generate() 사용
        '''
        req_outputs: List[RequestOutput] = self._llm.generate(
            prompts,
            sampling_params=sampling_params,
            lora_request=lora_request,
            use_tqdm=False
        )

        results = []
        for req_output in req_outputs:
            completion_output: CompletionOutput = req_output.outputs[0]
            results.append(self._make_generate_result(completion_output, return_completion_output))
        
        self._called_cnt += 1
        self._called_cnt_all += 1
        if do_print:
            if self._called_cnt_all % 100 == 0:
                print(f'# VllmEngine.generate_batch() [vllm] called_cnt : {self._called_cnt}, called_cnt_all : {self._called_cnt_all}')
        
        return results

