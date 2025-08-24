from _init import *

from typing import List, Dict, Union, Tuple, Any
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput, CompletionOutput
from vllm.sequence import Logprob

from ranger.vllm.vllm_agent import VllmAgent


class VllmEngine(VllmAgent):
    def __init__(self, model_name: str, device: str, gpu_memory_utilization: float, dtype: str, max_model_len: int):
        super().__init__()

        self._model_name = model_name
        self._device = device
        self._gpu_memory_utilization = gpu_memory_utilization
        self._dtype = dtype
        self._max_model_len = max_model_len

        self._llm = LLM(
            model=self._model_name,
            device=self._device,
            gpu_memory_utilization=self._gpu_memory_utilization,
            dtype=self._dtype,
            max_model_len=self._max_model_len,          # 없으면, 에러남
            enable_prefix_caching=True                  # 프리픽스 캐싱 활성화(성능 향상)
        )
        self._tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(self._model_name)
        self._called_cnt = 0


    def _get_generated_text(self, completion: CompletionOutput) -> str:
        return completion.text.strip()


    def _get_generated_toks_log_probs(self, completion: CompletionOutput) -> Tuple[List, List]:
        toks, log_probs = [], []

        for i, tok_id in enumerate(completion.token_ids):
            tok = self._tokenizer.decode(tok_id).strip()
            if len(tok) == 0:
                continue

            log_prob_temp: Logprob = completion.logprobs[i][tok_id]
            log_prob = log_prob_temp.logprob

            toks.append(tok)
            log_probs.append(log_prob)

        return toks, log_probs


    def _make_generate_result(self, completion: CompletionOutput, return_toks_log_probs: bool, do_print: bool) -> Union[str, Tuple[str, Any]]:
        generated_text = self._get_generated_text(completion)

        if not return_toks_log_probs:
            return generated_text
        else:
            toks, log_probs = self._get_generated_toks_log_probs(completion)
            return generated_text, (toks, log_probs)


    def generate_batch(self, messages: List[List[Dict]], max_token_gen: int, temperature: int,
                       return_toks_log_probs=False, do_print=True) -> List[Union[str, Tuple[str, Any]]]:
        
        sampling_params = SamplingParams(
            max_tokens=max_token_gen,
            temperature=temperature,
            logprobs=1
        )

        req_outputs: List[RequestOutput] = self._llm.chat(
            messages,
            sampling_params=sampling_params,
            use_tqdm=False
        )

        results = []
        for req_output in req_outputs:
            completion: CompletionOutput = req_output.outputs[0]
            results.append(self._make_generate_result(completion, return_toks_log_probs, do_print))
        
        self._called_cnt += 1
        if do_print:
            if self._called_cnt % 100 == 0:
                print(f'# VllmEngine.generate_batch() vllm called_cnt : {self._called_cnt}')
        
        return results

