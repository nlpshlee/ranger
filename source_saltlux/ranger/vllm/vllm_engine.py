from _init import *

import math
from typing import List, Dict, Union, Tuple, Any
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput, CompletionOutput
from vllm.sequence import Logprob
from vllm.lora.request import LoRARequest


class VllmEngine:
    def __init__(self, model_name: str, device: str, dtype: str, max_seq_length: int, max_new_tokens: int,
                 temperature: float, gpu_memory_utilization: float, n_log_prob: int):

        self._model_name = model_name
        self._device = device
        self._dtype = dtype
        self._max_seq_length = max_seq_length
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._gpu_memory_utilization = gpu_memory_utilization
        self._n_log_prob = n_log_prob

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
            dtype=self._dtype,
            max_model_len=self._max_seq_length, # 없으면, 에러남
            gpu_memory_utilization=self._gpu_memory_utilization,
            enable_lora=True,
            enable_prefix_caching=False # 프리픽스 캐싱 활성화(속도 향상), 근데 한 번 어댑터 주면 캐싱되기 때문에 사용 X
            # max_loras=1,
            # enforce_eager=True # CUDA Graph 비활성화 (속도는 조금 느려짐, 재현성은 높아짐), 해결 안됨
        )

        self._tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(self._model_name)
        self._tok_id_end = self._tokenizer.eos_token_id

        # 검증 과정에서 재현이 필요한 경우에만 설정 (완벽한 재현은 안됨...)
        self._seed = -1


    def reset(self):
        self._called_cnt = 0


    def _truncate_prompts(self, prompts: List[str]) -> List[str]:
        truncated_prompts = []

        for prompt in prompts:
            token_ids = self._tokenizer(
                prompt,
                truncation=True,
                max_length=self._max_seq_length,
                add_special_tokens=True
            )['input_ids']

            if 'llama' in self._model_name.lower():
                if len(token_ids) > 1 and token_ids[0] == token_ids[1] == 128000:
                    token_ids = token_ids[1:]

            truncated_prompt = self._tokenizer.decode(token_ids, skip_special_tokens=False)
            truncated_prompts.append(truncated_prompt)
        
        return truncated_prompts


    def _get_generated_text(self, completion_output: CompletionOutput) -> str:
        return completion_output.text.strip()
    

    def get_generated_log_prob(self, completion_output: CompletionOutput, text='', tok_ids=[], missing_log_prob=math.log(1e-9), return_all=False):
        # 'text'가 주어지면, 'text'를 토크나이징하고 likelihood 계산
        if len(text) > 0:
            tok_ids: list = self._tokenizer(text, add_special_tokens=False)['input_ids']
        # 'text'도 주어지지 않고, 'tok_ids'도 주어지지 않으면, 'completion_output'에 내장되어 있는 'tok_ids' 사용
        elif len(tok_ids) == 0:
            tok_ids: list = completion_output.token_ids
        
        if tok_ids[-1] != self._tok_id_end:
            tok_ids.append(self._tok_id_end)

        tok_log_probs = []
        completion_output_log_prob_len = len(completion_output.logprobs)

        for tok_idx in range(len(tok_ids)):
            tok_id = tok_ids[tok_idx]

            # 모델이 생성한 텍스트보다 정답이 더 긴 경우
            if completion_output_log_prob_len <= tok_idx:
                tok_log_probs.append(missing_log_prob)
                continue

            # 'tok_idx'번째 위치에서의 전체 토큰 확률 분포 (실제로는 top-k)
            all_tok_log_prob = completion_output.logprobs[tok_idx]

            if tok_id in all_tok_log_prob.keys():
                tok_log_prob: Logprob = all_tok_log_prob[tok_id]
                tok_log_prob = tok_log_prob.logprob
            else:
                tok_log_prob = missing_log_prob
            
            tok_log_probs.append(tok_log_prob)
        
        log_prob = sum(tok_log_probs) / len(tok_log_probs)

        if return_all:
            toks = [self._tokenizer.decode(tok_id) for tok_id in tok_ids]
            return log_prob, toks, tok_ids, tok_log_probs
        else:
            return log_prob


    def _make_generate_result(self, completion_output: CompletionOutput, return_completion_output: bool) -> Union[str, Tuple[str, Any]]:
        generated_text = self._get_generated_text(completion_output)

        if not return_completion_output:
            return generated_text
        else:
            return generated_text, completion_output


    def generate_batch(self, datas: List[Dict], return_completion_output=False, adapter_path='') -> List[Union[str, Tuple[str, Any]]]:

        # LoRA Adapter 추가 코드
        if os.path.exists(adapter_path):
            lora_request = LoRARequest('RANGER_Policy', 1, adapter_path)
        else:
            lora_request = None
        
        sampling_params = SamplingParams(
            max_tokens=self._max_new_tokens,
            temperature=self._temperature,
            logprobs=self._n_log_prob if return_completion_output else None
        )

        if self._seed != -1:
            sampling_params.seed = self._seed

        prompts = [self._tokenizer.apply_chat_template([data], tokenize=False, add_generation_prompt=True) for data in datas]

        # vllm은 내부적으로 입력 길이 제한을 하지 않음 -> 직접 잘라서 넘겨줘야 함...
        truncated_prompts = self._truncate_prompts(prompts)

        # LoRA Adapter 추가 코드
        '''
            chat() -> generate()
                datas -> prompts
                (+) lora_request
            
            chat() 함수에 lora_request 전달 불가능
                - vllm.LLM.chat() 메소드는 lora_request 파라미터를 지원하지 않도록 설계됨
                - LoRA를 동적으로 적용하려면 반드시 llm.generate() 사용
        '''
        req_outputs: List[RequestOutput] = self._llm.generate(
            truncated_prompts,
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
        if DEBUG.VLLM:
            if self._called_cnt_all % 100 == 0:
                print(f'# [VLLM] VllmEngine.generate_batch() vllm called_cnt : {self._called_cnt}, called_cnt_all : {self._called_cnt_all}')
        
        return results

