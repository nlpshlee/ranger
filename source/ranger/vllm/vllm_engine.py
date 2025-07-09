from _init import *

from typing import List, Dict, Union, Tuple, Any
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput, CompletionOutput
from vllm.sequence import Logprob

from ranger.vllm.vllm_agent import VllmAgent


class VllmEngine(VllmAgent):
    def __init__(self, model_name: str, device='cuda:0', gpu_memory_utilization=0.95, dtype='float16', max_model_len=4096):
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
            enable_prefix_caching=True, # 프리픽스 캐싱 활성화(성능 향상)
            max_model_len=self._max_model_len
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


    def generate(self, messages: List[Dict], return_toks_log_probs=False, do_print=True, **kwargs) -> Union[str, Tuple[str, Any]]:
        prompts = [msg["content"] for msg in messages]

        if 'max_tokens' in kwargs.keys():
            sampling_params = SamplingParams(temperature=kwargs['temperature'], logprobs=1, max_tokens=kwargs['max_tokens'])
        else:
            sampling_params = SamplingParams(temperature=kwargs['temperature'], logprobs=1)

        req_output: RequestOutput = self._llm.generate(
            prompts,
            sampling_params=sampling_params,
            use_tqdm=False
        )[0]

        completion: CompletionOutput = req_output.outputs[0]

        return self._return_generate(completion, return_toks_log_probs, do_print)


    def generate_batch(self, messages: List[List[Dict]], return_toks_log_probs=False, num_workers=4, do_print=True, **kwargs) -> List[Union[str, Tuple[str, Any]]]:
        results = []

        for m in messages:
            # vllm.LLM 은 싱글 스레드로만 쓸 수 있다는데..?
            results.append(self.generate(m, return_toks_log_probs, do_print, **kwargs))
        
        if do_print:
            if self._called_cnt % 1000 != 0:
                print(f'VllmEngine.generate_batch() called_cnt : {self._called_cnt}\n')
        
        return results

