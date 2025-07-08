from _init import *

from typing import List, Dict, Union, Tuple, Any
from openai import OpenAI
from openai.types.chat import ChatCompletion
from concurrent.futures import ThreadPoolExecutor

from ranger.vllm.vllm_interface import VllmInterface


class VllmClient(VllmInterface):
    def __init__(self, model_name: str, host='localhost', port=8000, api_key='token-123'):
        super().__init__()

        self._model_name = model_name
        self._base_url = f'http://{host}:{port}/v1'
        self._api_key = api_key
        
        self._client = OpenAI(base_url=self._base_url, api_key=self._api_key)
        self._called_cnt = 0
    

    def generate(self, messages: List[Dict], return_toks_log_probs=False, do_print=True, **kwargs) -> Union[str, Tuple[str, Any]]:
        completion: ChatCompletion = self._client.chat.completions.create(
            model=self._model_name,
            messages=messages,
            logprobs=1,
            **kwargs
        )

        self._called_cnt += 1
        if do_print:
            if self._called_cnt % 1000 == 0:
                print(f'VllmClient.generate() called_cnt : {self._called_cnt}')
        
        generated_text = self.get_generated_text(completion)

        if not return_toks_log_probs:
            return generated_text
        else:
            toks, log_probs = self.get_generated_toks_log_probs(completion)
            return generated_text, (toks, log_probs)


    def generate_batch(self, messages: List[List[Dict]], return_toks_log_probs=False, num_workers=4, do_print=True, **kwargs) -> List[Union[str, Tuple[str, Any]]]:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            return list(executor.map(
                lambda m: self.generate(m, return_toks_log_probs, do_print, **kwargs),
                messages
            ))
        
        if do_print:
            if self._called_cnt % 1000 != 0:
                print(f'VllmClient.generate_batch() called_cnt : {self._called_cnt}\n')

