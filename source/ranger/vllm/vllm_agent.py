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
        self._max_model_len: int
        self._max_token_gen: int
        self._called_cnt: int
    

    @abstractmethod
    def _get_generated_text(self, result: Any) -> str:
        pass
    
    
    @abstractmethod
    def _get_generated_toks_log_probs(self, result: Any) -> Tuple[List, List]:
        pass


    @abstractmethod
    def _make_generate_result(self,
                              completion: Any,
                              return_toks_log_probs: bool) -> Union[str, Tuple[str, Any]]:
        pass


    # @abstractmethod
    # def generate(self,
    #              messages: List[Dict],
    #              max_token_gen: int,
    #              temperature: int,
    #              return_toks_log_probs=False) -> Union[str, Tuple[str, Any]]:
    #     pass

    @abstractmethod
    def generate_batch(self,
                       messages: List[List[Dict]],
                       max_token_gen: int,
                       temperature: int,
                       return_toks_log_probs=False) -> List[Union[str, Tuple[str, Any]]]:
        pass

