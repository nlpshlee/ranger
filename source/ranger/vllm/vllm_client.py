from _init import *

from typing import List, Dict, Union, Tuple, Any
from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionTokenLogprob
from concurrent.futures import ThreadPoolExecutor

from ranger.vllm.vllm_agent import VllmAgent


class VllmClient(VllmAgent):
    def __init__(self, model_name: str, host='localhost', port=8000, api_key='token-123'):
        super().__init__()

        self._model_name = model_name
        self._base_url = f'http://{host}:{port}/v1'
        self._api_key = api_key
        
        self._client = OpenAI(base_url=self._base_url, api_key=self._api_key)
        self._called_cnt = 0
    

    def _get_generated_text(self, completion: ChatCompletion) -> str:
        return completion.choices[0].message.content
    
    
    def _get_generated_toks_log_probs(self, completion: ChatCompletion) -> Tuple[List, List]:
        completion_tok_logprob_list: List[ChatCompletionTokenLogprob] = completion.choices[0].logprobs.content

        toks, log_probs = [], []
        for completion_tok_logprob in completion_tok_logprob_list:
            toks.append(completion_tok_logprob.token)
            log_probs.append(completion_tok_logprob.logprob)
        
        return toks, log_probs
    

    def generate(self, messages: List[Dict], return_toks_log_probs=False, do_print=True, **kwargs) -> Union[str, Tuple[str, Any]]:
        completion: ChatCompletion = self._client.chat.completions.create(
            model=self._model_name,
            messages=messages,
            logprobs=1,
            **kwargs
        )
        
        return self._return_generate(completion, return_toks_log_probs, do_print)


    def generate_batch(self, messages: List[List[Dict]], return_toks_log_probs=False, num_workers=4, do_print=True, **kwargs) -> List[Union[str, Tuple[str, Any]]]:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(
                lambda m: self.generate(m, return_toks_log_probs, do_print, **kwargs),
                messages
            ))
        
            if do_print:
                if self._called_cnt % 1000 != 0:
                    print(f'VllmClient.generate_batch() called_cnt : {self._called_cnt}\n')
            
            return results

