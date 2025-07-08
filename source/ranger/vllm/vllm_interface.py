from _init import *

from abc import ABC, abstractmethod
from typing import List, Dict, Union, Tuple, Any
from openai.types.chat import ChatCompletion, ChatCompletionTokenLogprob

'''
    서버 통신형 클라이언트와 내부 엔진형 클래스의 공통 추상 인터페이스
'''
class VllmInterface(ABC):
    def __init__(self):
        super().__init__()

        self._model_name: str
        self._called_cnt: int


    @abstractmethod
    def generate(self, messages: List[Dict], return_toks_log_probs=False, **kwargs) -> Union[str, Tuple[str, Any]]:
        pass

    @abstractmethod
    def generate_batch(self, messages: List[List[Dict]], return_toks_log_probs=False, num_workers=4, **kwargs) -> List[Union[str, Tuple[str, Any]]]:
        pass


    def get_generated_text(self, completion: ChatCompletion) -> str:
        return completion.choices[0].message.content
    
    
    def get_generated_toks_log_probs(self, completion: ChatCompletion) -> Tuple[List, List]:
        completion_tok_logprob_list: List[ChatCompletionTokenLogprob] = completion.choices[0].logprobs.content

        toks, log_probs = [], []
        for completion_tok_logprob in completion_tok_logprob_list:
            toks.append(completion_tok_logprob.token)
            log_probs.append(completion_tok_logprob.logprob)
        
        return toks, log_probs

