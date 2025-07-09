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


    def _make_generate_result(self, completion: ChatCompletion, return_toks_log_probs: bool, do_print: bool) -> Union[str, Tuple[str, Any]]:
        generated_text = self._get_generated_text(completion)

        if not return_toks_log_probs:
            return generated_text
        else:
            toks, log_probs = self._get_generated_toks_log_probs(completion)
            return generated_text, (toks, log_probs)


    def _generate(self, messages: List[Dict], max_token_gen: int, temperature: int,
                 return_toks_log_probs=False, do_print=True) -> Union[str, Tuple[str, Any]]:

        completion: ChatCompletion = self._client.chat.completions.create(
            model=self._model_name,
            messages=messages,
            max_tokens=max_token_gen,
            temperature=temperature,
            logprobs=1
        )

        result = self._make_generate_result(completion, return_toks_log_probs, do_print)

        self._called_cnt += 1
        if do_print:
            if self._called_cnt % 1000 == 0:
                print(f'VllmClient vllm called_cnt : {self._called_cnt}')
        
        return result


    def generate_batch(self, messages: List[List[Dict]], max_token_gen: int, temperature: int,
                       return_toks_log_probs=False, num_workers=4, do_print=True) -> List[Union[str, Tuple[str, Any]]]:

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(
                lambda m: self._generate(m, max_token_gen, temperature, return_toks_log_probs, do_print), messages
            ))
        
            if do_print:
                if self._called_cnt % 1000 != 0:
                    print(f'VllmClient vllm called_cnt : {self._called_cnt}')
            
            return results

