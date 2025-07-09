from _init import *

from abc import ABC, abstractmethod
from typing import List, Dict, Union, Tuple, Any

'''
    서버 통신형 클라이언트와 내부 엔진형 클래스의 공통 추상 인터페이스
'''
class VllmAgent(ABC):
    def __init__(self):
        super().__init__()

        self._model_name: str
        self._called_cnt: int
    

    @abstractmethod
    def _get_generated_text(self, result: Any) -> str:
        pass
    
    
    @abstractmethod
    def _get_generated_toks_log_probs(self, result: Any) -> Tuple[List, List]:
        pass


    def _return_generate(self, completion: Any, return_toks_log_probs: bool, do_print: bool) -> Union[str, Tuple[str, Any]]:
        self._called_cnt += 1
        if do_print:
            if self._called_cnt % 1000 == 0:
                print(f'{self.__class__.__name__}._return_generate() called_cnt : {self._called_cnt}')

        generated_text = self._get_generated_text(completion)

        if not return_toks_log_probs:
            return generated_text
        else:
            toks, log_probs = self._get_generated_toks_log_probs(completion)
            return generated_text, (toks, log_probs)


    @abstractmethod
    def generate(self, messages: List[Dict], return_toks_log_probs=False, **kwargs) -> Union[str, Tuple[str, Any]]:
        pass

    @abstractmethod
    def generate_batch(self, messages: List[List[Dict]], return_toks_log_probs=False, num_workers=4, **kwargs) -> List[Union[str, Tuple[str, Any]]]:
        pass

